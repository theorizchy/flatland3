import math
import multiprocessing
import os
import sys
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from utils.deadlock_check import check_if_all_blocked
from utils.timer import Timer
from utils.observation_utils import normalize_observation
from reinforcement_learning.dddqn_policy import DDDQNPolicy


def eval_policy(env_params, checkpoint, n_eval_episodes, max_steps, action_size, state_size, seed, render, allow_skipping, allow_caching):
    # Evaluation is faster on CPU (except if you use a really huge policy)
    parameters = {
        'use_gpu': False
    }

    policy = DDDQNPolicy(state_size, action_size, Namespace(**parameters), evaluation_mode=True)
    policy.qnetwork_local = torch.load(checkpoint)

    env_params = Namespace(**env_params)

    # Environment parameters
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city

    # Malfunction and speed profiles
    # TODO pass these parameters properly from main!
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1. / 2000,  # Rate of malfunctions
        min_duration=20,  # Minimal duration
        max_duration=50  # Max duration
    )

    # Only fast trains in Round 1
    speed_profiles = {
        1.: 1.0,  # Fast passenger train
        1. / 2.: 0.0,  # Fast freight train
        1. / 3.: 0.0,  # Slow commuter train
        1. / 4.: 0.0  # Slow freight train
    }

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_radius = env_params.observation_radius
    observation_max_path_depth = env_params.observation_max_path_depth

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city,
        ),
        schedule_generator=sparse_schedule_generator(speed_profiles),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation
    )

    if render:
        env_renderer = RenderTool(env, gl="PGL")

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []
    inference_times = []
    preproc_times = []
    agent_times = []
    step_times = []

    for episode_idx in range(n_eval_episodes):
        seed += 1

        inference_timer = Timer()
        preproc_timer = Timer()
        agent_timer = Timer()
        step_timer = Timer()

        step_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
        step_timer.end()

        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        if render:
            env_renderer.set_new_rail()

        final_step = 0
        skipped = 0

        nb_hit = 0
        agent_last_obs = {}
        agent_last_action = {}

        for step in range(max_steps - 1):
            if allow_skipping and check_if_all_blocked(env):
                # FIXME why -1? bug where all agents are "done" after max_steps!
                skipped = max_steps - step - 1
                final_step = max_steps - 2
                n_unfinished_agents = sum(not done[idx] for idx in env.get_agent_handles())
                score -= skipped * n_unfinished_agents
                break

            agent_timer.start()
            for agent in env.get_agent_handles():
                if obs[agent] and info['action_required'][agent]:
                    if agent in agent_last_obs and np.all(agent_last_obs[agent] == obs[agent]):
                        nb_hit += 1
                        action = agent_last_action[agent]

                    else:
                        preproc_timer.start()
                        norm_obs = normalize_observation(obs[agent], tree_depth=observation_tree_depth, observation_radius=observation_radius)
                        preproc_timer.end()

                        inference_timer.start()
                        action = policy.act(norm_obs, eps=0.0)
                        inference_timer.end()

                    action_dict.update({agent: action})

                    if allow_caching:
                        agent_last_obs[agent] = obs[agent]
                        agent_last_action[agent] = action
            agent_timer.end()

            step_timer.start()
            obs, all_rewards, done, info = env.step(action_dict)
            step_timer.end()

            if render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

                if step % 100 == 0:
                    print("{}/{}".format(step, max_steps - 1))

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

        inference_times.append(inference_timer.get())
        preproc_times.append(preproc_timer.get())
        agent_times.append(agent_timer.get())
        step_times.append(step_timer.get())

        skipped_text = ""
        if skipped > 0:
            skipped_text = "\t⚡ Skipped {}".format(skipped)

        hit_text = ""
        if nb_hit > 0:
            hit_text = "\t⚡ Hit {} ({:.1f}%)".format(nb_hit, (100 * nb_hit) / (n_agents * final_step))

        print(
            "☑️  Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} "
            "\t🍭 Seed: {}"
            "\t🚉 Env: {:.3f}s  "
            "\t🤖 Agent: {:.3f}s (per step: {:.3f}s) \t[preproc: {:.3f}s \tinfer: {:.3f}s]"
            "{}{}".format(
                normalized_score,
                completion * 100.0,
                final_step,
                seed,
                step_timer.get(),
                agent_timer.get(),
                agent_timer.get() / final_step,
                preproc_timer.get(),
                inference_timer.get(),
                skipped_text,
                hit_text
            )
        )

    return scores, completions, nb_steps, agent_times, step_times


def evaluate_agents(file, n_evaluation_episodes, use_gpu, render, allow_skipping, allow_caching):
    nb_threads = 1
    eval_per_thread = n_evaluation_episodes

    if not render:
        nb_threads = multiprocessing.cpu_count()
        eval_per_thread = max(1, math.ceil(n_evaluation_episodes / nb_threads))

    total_nb_eval = eval_per_thread * nb_threads
    print("Will evaluate policy {} over {} episodes on {} threads.".format(file, total_nb_eval, nb_threads))

    if total_nb_eval != n_evaluation_episodes:
        print("(Rounding up from {} to fill all cores)".format(n_evaluation_episodes))

    # Observation parameters need to match the ones used during training!

    # small_v0
    small_v0_params = {
        # sample configuration
        "n_agents": 5,
        "x_dim": 25,
        "y_dim": 25,
        "n_cities": 4,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 20
    }

    # Test_0
    test0_params = {
        # sample configuration
        "n_agents": 5,
        "x_dim": 25,
        "y_dim": 25,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 20
    }

    # Test_1
    test1_params = {
        # environment
        "n_agents": 10,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 10
    }

    # Test_5
    test5_params = {
        # environment
        "n_agents": 80,
        "x_dim": 35,
        "y_dim": 35,
        "n_cities": 5,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 4,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 20
    }

    params = small_v0_params
    env_params = Namespace(**params)

    print("Environment parameters:")
    pprint(params)

    # Calculate space dimensions and max steps
    max_steps = int(4 * 2 * (env_params.x_dim + env_params.y_dim + (env_params.n_agents / env_params.n_cities)))
    action_size = 5
    tree_observation = TreeObsForRailEnv(max_depth=env_params.observation_tree_depth)
    tree_depth = env_params.observation_tree_depth
    num_features_per_node = tree_observation.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(tree_depth + 1)])
    state_size = num_features_per_node * n_nodes

    results = []
    if render:
        results.append(eval_policy(params, file, eval_per_thread, max_steps, action_size, state_size, 0, render, allow_skipping, allow_caching))

    else:
        with Pool() as p:
            results = p.starmap(eval_policy,
                                [(params, file, 1, max_steps, action_size, state_size, seed * nb_threads, render, allow_skipping, allow_caching)
                                 for seed in
                                 range(total_nb_eval)])

    scores = []
    completions = []
    nb_steps = []
    times = []
    step_times = []
    for s, c, n, t, st in results:
        scores.append(s)
        completions.append(c)
        nb_steps.append(n)
        times.append(t)
        step_times.append(st)

    print("-" * 200)

    print("✅ Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} \tAgent total: {:.3f}s (per step: {:.3f}s)".format(
        np.mean(scores),
        np.mean(completions) * 100.0,
        np.mean(nb_steps),
        np.mean(times),
        np.mean(times) / np.mean(nb_steps)
    ))

    print("⏲️  Agent sum: {:.3f}s \tEnv sum: {:.3f}s \tTotal sum: {:.3f}s".format(
        np.sum(times),
        np.sum(step_times),
        np.sum(times) + np.sum(step_times)
    ))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="checkpoint to load", required=True, type=str)
    parser.add_argument("-n", "--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)

    # TODO
    # parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=0, type=int)

    parser.add_argument("--use_gpu", dest="use_gpu", help="use GPU if available", action='store_true')
    parser.add_argument("--render", help="render a single episode", action='store_true')
    parser.add_argument("--allow_skipping", help="skips to the end of the episode if all agents are deadlocked", action='store_true')
    parser.add_argument("--allow_caching", help="caches the last observation-action pair", action='store_true')
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(1)
    evaluate_agents(file=args.file, n_evaluation_episodes=args.n_evaluation_episodes, use_gpu=args.use_gpu, render=args.render,
                    allow_skipping=args.allow_skipping, allow_caching=args.allow_caching)
