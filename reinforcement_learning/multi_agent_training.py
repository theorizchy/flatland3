import os
import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional, List

import numpy as np
import psutil
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.deadlockavoidance_with_decision_agent import DeadLockAvoidanceWithDecisionAgent
from reinforcement_learning.decision_point_agent import DecisionPointAgent
from reinforcement_learning.multi_decision_agent import MultiDecisionAgent
from reinforcement_learning.multi_policy import MultiPolicy
from reinforcement_learning.ppo_agent import FLATLandPPOPolicy
from utils.agent_action_config import get_flatland_full_action_size, get_action_size, map_actions, map_action, \
    set_action_size_reduced, set_action_size_full, convert_default_rail_env_action
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent
from utils.deadlock_check import find_and_punish_deadlock
from utils.flatland_observation import FlatlandObservation, FlatlandTreeObservation, FlatlandFastTreeObservation
from utils.shortest_path_walking_agent import ShortestPathWalkingAgent

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from utils.timer import Timer

try:
    import wandb

    wandb.init(sync_tensorboard=True)
except ImportError:
    print("Install wandb to log to Weights & Biases")

"""
This file shows how to train multiple agents using a reinforcement learning approach.
After training an agent, you can submit it straight away to the NeurIPS 2020 Flatland challenge!

Agent documentation: https://flatland.aicrowd.com/getting-started/rl/multi-agent.html
Submission documentation: https://flatland.aicrowd.com/getting-started/first-submission.html
"""


def create_rail_env(env_params, tree_observation):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )


def train_agent(train_params, train_env_params, eval_env_params, obs_params):
    # Environment parameters
    n_agents = train_env_params.n_agents
    x_dim = train_env_params.x_dim
    y_dim = train_env_params.y_dim
    n_cities = train_env_params.n_cities
    max_rails_between_cities = train_env_params.max_rails_between_cities
    max_rails_in_city = train_env_params.max_rails_in_city
    seed = train_env_params.seed

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Observation parameters
    observation_tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    observation_max_path_depth = obs_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    n_eval_episodes = train_params.n_evaluation_episodes
    restore_replay_buffer = train_params.restore_replay_buffer
    save_replay_buffer = train_params.save_replay_buffer
    skip_unfinished_agent = train_params.skip_unfinished_agent

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    print("------------------------------------- CREATE OBSERVATION -----------------------------------")
    if train_params.use_observation == 'TreeObs':
        print("Using FlatlandTreeObservation (standard)")
        tree_observation = FlatlandTreeObservation(max_depth=observation_tree_depth)
    elif train_params.use_observation == 'FastTreeObs':
        print("Using FlatlandFastTreeObservation")
        tree_observation = FlatlandFastTreeObservation()
    else:  # train_params.use_observation == 'FlatlandObs':
        print("Using FlatlandObservation")
        tree_observation = FlatlandObservation(max_depth=observation_tree_depth)
    # Get the state size
    state_size = tree_observation.observation_dim

    if train_params.policy == "DeadLockAvoidance":
        print("Using SimpleObservationBuilder")

        class SimpleObservationBuilder(ObservationBuilder):
            """
            DummyObservationBuilder class which returns dummy observations
            This is used in the evaluation service
            """

            def __init__(self):
                super().__init__()

            def reset(self):
                pass

            def get_many(self, handles: Optional[List[int]] = None):
                return super().get_many(handles)

            def get(self, handle: int = 0):
                return [handle]

        tree_observation = SimpleObservationBuilder()
        tree_observation.observation_dim = 1

    # Setup the environments
    train_env = create_rail_env(train_env_params, tree_observation)
    train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    eval_env = create_rail_env(eval_env_params, tree_observation)
    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

    action_count = [0] * get_flatland_full_action_size()
    action_dict = dict()

    # Smoothed values used as target for hyperparameter tuning
    smoothed_eval_normalized_score = -1.0
    smoothed_eval_completion = 0.0

    scores_window = deque(maxlen=checkpoint_interval)  # todo smooth when rendering instead
    completion_window = deque(maxlen=checkpoint_interval)
    deadlocked_window = deque(maxlen=checkpoint_interval)

    if train_params.action_size == "reduced":
        set_action_size_reduced()
    else:
        set_action_size_full()

    print("---------------------------------------- CREATE AGENT --------------------------------------")
    print('Using', train_params.policy)
    if train_params.policy == "DDDQN":
        policy = DDDQNPolicy(state_size, get_action_size(), train_params,
                             enable_delayed_transition_push_at_episode_end=False,
                             skip_unfinished_agent=skip_unfinished_agent)
    elif train_params.policy == "ShortestPathWalkingAgent":
        policy = ShortestPathWalkingAgent(train_env)
    elif train_params.policy == "PPO":
        policy = FLATLandPPOPolicy(state_size, get_action_size(),
                                   use_replay_buffer=train_params.buffer_size > 0,
                                   enable_replay_curiosity_sampling=False,
                                   in_parameters=train_params,
                                   skip_unfinished_agent=skip_unfinished_agent,
                                   K_epoch=train_params.K_epoch)
    elif train_params.policy == "PPORCS":
        policy = FLATLandPPOPolicy(state_size, get_action_size(),
                                   use_replay_buffer=train_params.buffer_size > 0,
                                   enable_replay_curiosity_sampling=True,
                                   in_parameters=train_params,
                                   skip_unfinished_agent=skip_unfinished_agent,
                                   K_epoch=train_params.K_epoch)
    elif train_params.policy == "DeadLockAvoidance":
        policy = DeadLockAvoidanceAgent(train_env, get_action_size(), enable_eps=False)
    elif train_params.policy == "DeadLockAvoidanceWithDecisionAgent":
        policy = DeadLockAvoidanceWithDecisionAgent(train_env, state_size, get_action_size(),
                                                    in_parameters=train_params)
    elif train_params.policy == "DecisionPointAgent":
        inter_policy = FLATLandPPOPolicy(state_size, get_action_size(),
                                         use_replay_buffer=train_params.buffer_size > 0,
                                         enable_replay_curiosity_sampling=True,
                                         in_parameters=train_params,
                                         skip_unfinished_agent=skip_unfinished_agent,
                                         K_epoch=train_params.K_epoch)
        policy = DecisionPointAgent(train_env, state_size, get_action_size(), inter_policy)
    elif train_params.policy == "DecisionPointAgent_DDDQN":
        inter_policy = DDDQNPolicy(state_size, get_action_size(), train_params,
                                   enable_delayed_transition_push_at_episode_end=False,
                                   skip_unfinished_agent=skip_unfinished_agent)
        policy = DecisionPointAgent(train_env, state_size, get_action_size(), inter_policy)
    elif train_params.policy == "MultiDecisionAgent":
        policy = MultiDecisionAgent(train_env, state_size, get_action_size(), train_params)
    elif train_params.policy == "MultiPolicy":
        policy = MultiPolicy(state_size, get_action_size())
    else:
        policy = FLATLandPPOPolicy(state_size, get_action_size(), use_replay_buffer=False, in_parameters=train_params)

    # make sure that at least one policy is set
    if policy is None:
        policy = DDDQNPolicy(state_size, get_action_size(), train_params)

    # Load existing policy
    if train_params.load_policy != "":
        policy.load(train_params.load_policy)

    # Loads existing replay buffer
    if restore_replay_buffer:
        try:
            policy.load_replay_buffer(restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print("\n🛑 Could't load replay buffer, were the experiences generated using the same tree depth?")
            print(e)
            exit(1)

    print("\n💾 Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), train_params.buffer_size))

    hdd = psutil.disk_usage('/')
    if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
        print("⚠️  Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left."
              .format(hdd.free / (2 ** 30)))

    # TensorBoard writer
    writer = SummaryWriter(comment="_" +
                                   train_params.policy + "_" +
                                   train_params.use_observation + "_" +
                                   train_params.action_size)

    training_timer = Timer()
    training_timer.start()

    print(
        "\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating {} trains on {} episodes every {} episodes. "
        "Training id '{}'.\n".format(
            train_env.get_num_agents(),
            x_dim, y_dim,
            n_episodes,
            eval_env.get_num_agents(),
            n_eval_episodes,
            checkpoint_interval,
            training_id
        ))

    for episode_idx in range(n_episodes + 1):
        reset_timer = Timer()
        policy_start_episode_timer = Timer()
        policy_start_step_timer = Timer()
        policy_act_timer = Timer()
        env_step_timer = Timer()
        policy_shape_reward_timer = Timer()
        policy_step_timer = Timer()
        policy_end_step_timer = Timer()
        policy_end_episode_timer = Timer()
        total_episode_timer = Timer()

        total_episode_timer.start()

        action_count = [0] * get_flatland_full_action_size()
        agent_prev_obs = [None] * n_agents
        agent_prev_action = [convert_default_rail_env_action(RailEnvActions.STOP_MOVING)] * n_agents
        update_values = [False] * n_agents

        # Reset environment
        reset_timer.start()
        if train_params.n_agent_fixed:
            number_of_agents = n_agents
        else:
            number_of_agents = int(min(n_agents, 1 + np.floor(max(0, episode_idx - 1) / 200)))
        if train_params.n_agent_iterate:
            train_env_params.n_agents = episode_idx % number_of_agents + 1
        else:
            train_env_params.n_agents = number_of_agents

        train_env = create_rail_env(train_env_params, tree_observation)
        agent_obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)
        policy.reset(train_env)
        reset_timer.end()

        if train_params.render:
            # Setup renderer
            env_renderer = RenderTool(train_env, gl="PGL",
                                      show_debug=True,
                                      agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS)

            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent_handle in train_env.get_agent_handles():
            agent_prev_obs[agent_handle] = agent_obs[agent_handle].copy()

        # Max number of steps per episode
        # This is the official formula used during evaluations
        # See details in flatland.envs.schedule_generators.sparse_schedule_generator
        # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
        max_steps = train_env._max_episode_steps

        # Run episode
        policy_start_episode_timer.start()
        policy.start_episode(train=True)
        policy_start_episode_timer.end()
        for step in range(max_steps - 1):
            # policy.start_step ---------------------------------------------------------------------------------------
            policy_start_step_timer.start()
            policy.start_step(train=True)
            policy_start_step_timer.end()

            # policy.act ----------------------------------------------------------------------------------------------
            policy_act_timer.start()
            action_dict = {}
            for agent_handle in policy.get_agent_handles(train_env):
                if info['action_required'][agent_handle]:
                    update_values[agent_handle] = True
                    action = policy.act(agent_handle, agent_obs[agent_handle], eps=eps_start)
                    action_count[map_action(action)] += 1
                    actions_taken.append(map_action(action))
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent_handle] = False
                    action = convert_default_rail_env_action(RailEnvActions.DO_NOTHING)

                action_dict.update({agent_handle: action})
            policy_act_timer.end()

            # policy.end_step -----------------------------------------------------------------------------------------
            policy_end_step_timer.start()
            policy.end_step(train=True)
            policy_end_step_timer.end()

            # Environment step ----------------------------------------------------------------------------------------
            env_step_timer.start()
            next_obs, all_rewards, dones, info = train_env.step(map_actions(action_dict))
            env_step_timer.end()

            # policy.shape_reward -------------------------------------------------------------------------------------
            policy_shape_reward_timer.start()
            # Deadlock
            deadlocked_agents, all_rewards, = find_and_punish_deadlock(train_env, all_rewards, -10.0)

            # The might requires a policy based transformation
            for agent_handle in train_env.get_agent_handles():
                all_rewards[agent_handle] = policy.shape_reward(agent_handle,
                                                                action_dict[agent_handle],
                                                                agent_obs[agent_handle],
                                                                all_rewards[agent_handle],
                                                                dones[agent_handle],
                                                                deadlocked_agents[agent_handle])

            policy_shape_reward_timer.end()

            # Render an episode at some interval
            if train_params.render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=True,
                    show_predictions=False
                )

            # Update replay buffer and train agent
            for agent_handle in train_env.get_agent_handles():
                if update_values[agent_handle] or dones['__all__'] or deadlocked_agents[agent_handle]:
                    # Only learn from timesteps where somethings happened
                    policy_step_timer.start()
                    policy.step(agent_handle,
                                agent_prev_obs[agent_handle],
                                agent_prev_action[agent_handle],
                                all_rewards[agent_handle],
                                agent_obs[agent_handle],
                                dones[agent_handle] or (deadlocked_agents[agent_handle] > 0))
                    policy_step_timer.end()

                    agent_prev_obs[agent_handle] = agent_obs[agent_handle].copy()
                    agent_prev_action[agent_handle] = action_dict[agent_handle]

                score += all_rewards[agent_handle]

                # update_observation (step)
                agent_obs[agent_handle] = next_obs[agent_handle].copy()

            nb_steps = step

            if dones['__all__']:
                break

            if deadlocked_agents['__all__']:
                if train_params.render_deadlocked is not None:
                    # Setup renderer
                    env_renderer = RenderTool(train_env,
                                              gl="PGL",
                                              show_debug=True,
                                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS,
                                              screen_width=2000,
                                              screen_height=1200)

                    env_renderer.set_new_rail()
                    env_renderer.render_env(
                        show=False,
                        frames=True,
                        show_observations=False,
                        show_predictions=False
                    )
                    env_renderer.gl.save_image("{}/flatland_{:04d}.png".format(
                        train_params.render_deadlocked,
                        episode_idx))
                    break

        # policy.end_episode
        policy_end_episode_timer.start()
        policy.end_episode(train=True)
        policy_end_episode_timer.end()

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        total_episode_timer.end()

        # Collect information about training
        tasks_finished = sum(dones[idx] for idx in train_env.get_agent_handles())
        tasks_deadlocked = sum(deadlocked_agents[idx] for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())
        deadlocked = tasks_deadlocked / max(1, train_env.get_num_agents())
        normalized_score = score / max(1, train_env.get_num_agents())
        action_probs = action_count / max(1, np.sum(action_count))

        scores_window.append(normalized_score)
        completion_window.append(completion)
        deadlocked_window.append(deadlocked)
        smoothed_normalized_score = np.mean(scores_window)
        smoothed_completion = np.mean(completion_window)
        smoothed_deadlocked = np.mean(deadlocked_window)

        if train_params.render:
            env_renderer.close_window()

        # Print logs
        if episode_idx % checkpoint_interval == 0 and episode_idx > 0:
            policy.save('./checkpoints/' + training_id + '-' + str(episode_idx) + '.pth')

            if save_replay_buffer:
                policy.save_replay_buffer('./replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            # reset action count
            action_count = [0] * get_flatland_full_action_size()

        print(
            '\r🚂 Episode {}'
            '\t 🚉 nAgents {:2}/{:2}'
            ' 🏆 Score: {:7.3f}'
            ' Avg: {:7.3f}'
            '\t 💯 Done: {:6.2f}%'
            ' Avg: {:6.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'.format(
                episode_idx,
                train_env_params.n_agents, number_of_agents,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                eps_start,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0 and episode_idx > 0:
            scores, completions, nb_steps_eval = eval_policy(eval_env,
                                                             tree_observation,
                                                             policy,
                                                             train_params,
                                                             obs_params)

            writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)

            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (
                    1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx)

        if episode_idx > 49:
            # Save logs to tensorboard
            writer.add_scalar("scene_done_training/completion_{}".format(train_env_params.n_agents),
                              np.mean(completion), episode_idx)
            writer.add_scalar("scene_dead_training/deadlocked_{}".format(train_env_params.n_agents),
                              np.mean(deadlocked), episode_idx)

            writer.add_scalar("training/score", normalized_score, episode_idx)
            writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
            writer.add_scalar("training/completion", np.mean(completion), episode_idx)
            writer.add_scalar("training/deadlocked", np.mean(deadlocked), episode_idx)
            writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
            writer.add_scalar("training/smoothed_deadlocked", np.mean(smoothed_deadlocked), episode_idx)
            writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
            writer.add_scalar("training/n_agents", train_env_params.n_agents, episode_idx)
            writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
            writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx)
            writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
            writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
            writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
            writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
            writer.add_scalar("training/epsilon", eps_start, episode_idx)
            writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
            writer.add_scalar("training/loss", policy.loss, episode_idx)

            writer.add_scalar("timer/00_reset", reset_timer.get(), episode_idx)
            writer.add_scalar("timer/01_policy_start_episode", policy_start_episode_timer.get(), episode_idx)
            writer.add_scalar("timer/02_policy_start_step", policy_start_step_timer.get(), episode_idx)
            writer.add_scalar("timer/03_policy_act", policy_act_timer.get(), episode_idx)
            writer.add_scalar("timer/04_env_step", env_step_timer.get(), episode_idx)
            writer.add_scalar("timer/05_policy_shape_reward", policy_shape_reward_timer.get(), episode_idx)
            writer.add_scalar("timer/06_policy_step", policy_step_timer.get(), episode_idx)
            writer.add_scalar("timer/07_policy_end_step", policy_end_step_timer.get(), episode_idx)
            writer.add_scalar("timer/08_policy_end_episode", policy_end_episode_timer.get(), episode_idx)
            writer.add_scalar("timer/09_total_episode", total_episode_timer.get_current(), episode_idx)
            writer.add_scalar("timer/10_total", training_timer.get_current(), episode_idx)

        writer.flush()


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, tree_observation, policy, train_params, obs_params):
    n_eval_episodes = train_params.n_evaluation_episodes
    max_steps = env._max_episode_steps

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        score = 0.0

        agent_obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        policy.reset(env)
        final_step = 0

        if train_params.eval_render:
            # Setup renderer
            env_renderer = RenderTool(env, gl="PGL",
                                      show_debug=True,
                                      agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS)
            env_renderer.set_new_rail()

        policy.start_episode(train=False)
        for step in range(max_steps - 1):
            policy.start_step(train=False)
            for agent in env.get_agent_handles():
                action = convert_default_rail_env_action(RailEnvActions.DO_NOTHING)
                if info['action_required'][agent]:
                    action = policy.act(agent, agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})
            policy.end_step(train=False)
            agent_obs, all_rewards, done, info = env.step(map_actions(action_dict))

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

            # Render an episode at some interval
            if train_params.eval_render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=True,
                    show_predictions=False
                )

        policy.end_episode(train=False)
        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

        if train_params.eval_render:
            env_renderer.close_window()

    print(" ✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=5000, type=int)
    parser.add_argument("--n_agent_fixed", help="hold the number of agent fixed", action='store_true')
    parser.add_argument("--n_agent_iterate", help="iterate the number of agent fixed", action='store_true')
    parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=1,
                        type=int)
    parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=3,
                        type=int)
    parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=10, type=int)
    parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=200, type=int)
    parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
    parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
    parser.add_argument("--eps_decay", help="exploration decay", default=0.99975, type=float)
    parser.add_argument("--buffer_size", help="replay buffer size", default=int(32_000), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--restore_replay_buffer", help="replay buffer to restore", default="", type=str)
    parser.add_argument("--save_replay_buffer", help="save replay buffer at each evaluation interval", default=False,
                        type=bool)
    parser.add_argument("--batch_size", help="minibatch size", default=1024, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
    parser.add_argument("--tau", help="soft update of target parameters", default=0.5e-3, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)
    parser.add_argument("--hidden_size", help="hidden size (2 fc layers)", default=128, type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=200, type=int)
    parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=4, type=int)

    parser.add_argument("--load_policy", help="policy filename (reference) to load", default="", type=str)
    parser.add_argument("--use_observation", help="observation name [TreeObs, FastTreeObs, FlatlandObs]",
                        default='FlatlandObs')
    parser.add_argument("--max_depth", help="max depth", default=2, type=int)
    parser.add_argument("--K_epoch", help="K_epoch", default=10, type=int)
    parser.add_argument("--skip_unfinished_agent", default=9999.0, type=float)
    parser.add_argument("--render", help="render while training", action='store_true')
    parser.add_argument("--eval_render", help="render evaluation", action='store_true')
    parser.add_argument("--render_deadlocked", default=None, type=str)
    parser.add_argument("--policy",
                        help="policy name [DDDQN, PPO, PPORCS, DecisionPointAgent, DecisionPointAgent_DDDQN,"
                             "DeadLockAvoidance, DeadLockAvoidanceWithDecisionAgent, MultiDecisionAgent, MultiPolicy]",
                        default="PPO")
    parser.add_argument("--action_size", help="define the action size [reduced,full]", default="full", type=str)

    training_params = parser.parse_args()
    env_params = [
        {
            # Test_0
            "n_agents": 1,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_1
            "n_agents": 2,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_2
            "n_agents": 5,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 0,
            "seed": 0
        },
        {
            # Test_3
            "n_agents": 10,
            "x_dim": 35,
            "y_dim": 35,
            "n_cities": 3,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {
            # Test_4
            "n_agents": 20,
            "x_dim": 40,
            "y_dim": 40,
            "n_cities": 5,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
    ]

obs_params = {
    "observation_tree_depth": training_params.max_depth,
    "observation_radius": 10,
    "observation_max_path_depth": 30
}


def check_env_config(id):
    if id >= len(env_params) or id < 0:
        print("\n🛑 Invalid environment configuration, only Test_0 to Test_{} are supported.".format(
            len(env_params) - 1))
        exit(1)


check_env_config(training_params.training_env_config)
check_env_config(training_params.evaluation_env_config)

training_env_params = env_params[training_params.training_env_config]
evaluation_env_params = env_params[training_params.evaluation_env_config]

# FIXME hard-coded for sweep search
# see https://wb-forum.slack.com/archives/CL4V2QE59/p1602931982236600 to implement properly
# training_params.use_fast_tree_observation = True

print("\nTraining parameters:")
pprint(vars(training_params))
print("\nTraining environment parameters (Test_{}):".format(training_params.training_env_config))
pprint(training_env_params)
print("\nEvaluation environment parameters (Test_{}):".format(training_params.evaluation_env_config))
pprint(evaluation_env_params)
print("\nObservation parameters:")
pprint(obs_params)

os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
train_agent(training_params, Namespace(**training_env_params), Namespace(**evaluation_env_params),
            Namespace(**obs_params))
