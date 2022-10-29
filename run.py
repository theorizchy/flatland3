"""
I did experiments in an early submission. Please note that the epsilon can have an
effects on the evaluation outcome :
DDDQNPolicy experiments - EPSILON impact analysis
----------------------------------------------------------------------------------------
checkpoint = "./checkpoints/201124171810-7800.pth"  # Training on AGENTS=10 with Depth=2
EPSILON = 0.000 # Sum Normalized Reward :  0.000000000000000 (primary score)
EPSILON = 0.002 # Sum Normalized Reward : 18.445875081269286 (primary score)
EPSILON = 0.005 # Sum Normalized Reward : 18.371733625865854 (primary score)
EPSILON = 0.010 # Sum Normalized Reward : 18.249244799876152 (primary score)
EPSILON = 0.020 # Sum Normalized Reward : 17.526987022691376 (primary score)
EPSILON = 0.030 # Sum Normalized Reward : 16.796885571003942 (primary score)
EPSILON = 0.040 # Sum Normalized Reward : 17.280787151431426 (primary score)
EPSILON = 0.050 # Sum Normalized Reward : 16.256945636647025 (primary score)
EPSILON = 0.100 # Sum Normalized Reward : 14.828347241759966 (primary score)
EPSILON = 0.200 # Sum Normalized Reward : 11.192330074898457 (primary score)
EPSILON = 0.300 # Sum Normalized Reward : 14.523067754608782 (primary score)
EPSILON = 0.400 # Sum Normalized Reward : 12.901508220410834 (primary score)
EPSILON = 0.500 # Sum Normalized Reward :  3.754660231871272 (primary score)
EPSILON = 1.000 # Sum Normalized Reward :  1.397180159192391 (primary score)
"""

import platform
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland.evaluators.client import FlatlandRemoteClient
from flatland.evaluators.client import TimeoutException
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.deadlockavoidance_with_decision_agent import DeadLockAvoidanceWithDecisionAgent
from reinforcement_learning.decision_point_agent import DecisionPointAgent
from reinforcement_learning.multi_decision_agent import MultiDecisionAgent
from reinforcement_learning.ppo_agent import PPOPolicy
from utils.agent_action_config import get_action_size, map_actions, set_action_size_full
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent, DeadlockAvoidanceObservation
from utils.deadlock_check import check_if_all_blocked
from utils.flatland_observation import FlatlandObservation, FlatlandFastTreeObservation, FlatlandTreeObservation

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

####################################################
# If rendering you might need to install freeglut
# sudo apt-get install freeglut3-dev

DO_RENDER = False

# Print per-step logs
VERBOSE = True

'''
    # -------------------------------------------------------------------------------------------------------
    # !! This is not a RL solution !!!!
    # -------------------------------------------------------------------------------------------------------
    # 116727 adrian_egli
    # graded	106.786	0.768	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/54
    # Sat, 23 Jan 2021 14:31:50
    set_action_size_reduced()
    load_policy = "DeadLockAvoidance"  # 21.037702071623333
    checkpoint = None
    EPSILON = 0.0
'''

#####################################################################
# EVALUATION PARAMETERS
set_action_size_full()
# load_policy = "DeadLockAvoidance"  # 19.921253063405388   #  50 agents
load_policy = "DecisionPointAgent"
load_second_policy = "PPO"
checkpoint = "./checkpoints/211208081311-14000.pth"  # 16.42901579305986 # 34 agents # training on 1:10
EPSILON = 0.0


#####################################################################
# Observation parameters (must match training parameters!)
print("Using FlatlandObservation")
observation_tree_depth = 2
flatland_observation = FlatlandFastTreeObservation()
flatland_observation = FlatlandTreeObservation(max_depth=observation_tree_depth)
flatland_observation = FlatlandObservation(max_depth=observation_tree_depth)
state_size = flatland_observation.observation_dim

# Use last action cache
USE_ACTION_CACHE = False

#####################################################################
DO_RENDER = DO_RENDER and platform.node() == 'K57261'

remote_client = FlatlandRemoteClient()

#####################################################################
# Main evaluation loop
#####################################################################
evaluation_number = 0

while True:
    evaluation_number += 1

    # We use a dummy observation and call TreeObsForRailEnv ourselves when needed.
    # This way we decide if we want to calculate the observations or not instead
    # of having them calculated every time we perform an env step.
    time_start = time.time()
    observation, info = remote_client.env_create(
        obs_builder_object=DummyObservationBuilder()
    )
    env_creation_time = time.time() - time_start

    if not observation:
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been
        # evaluated on all the required evaluation environments,
        # and hence it's safe to break out of the main evaluation loop.
        break

    print("Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)

    local_env = remote_client.env
    nb_agents = len(local_env.agents)
    max_nb_steps = local_env._max_episode_steps

    flatland_observation.set_env(local_env)
    flatland_observation.reset()

    # Creates the policy. No GPU on evaluation server.
    if load_policy == "DDDQN":
        policy = DDDQNPolicy(state_size, get_action_size(), Namespace(**{'use_gpu': False}), evaluation_mode=True)
    elif load_policy == "PPO":
        policy = PPOPolicy(state_size, get_action_size())
    elif load_policy == "DeadLockAvoidance":
        policy = DeadLockAvoidanceAgent(local_env, get_action_size(), enable_eps=False)
    elif load_policy == "DeadLockAvoidanceWithDecisionAgent":
        policy = DeadLockAvoidanceWithDecisionAgent(local_env, state_size, get_action_size())
    elif load_policy == "DecisionPointAgent":
        inter_policy = DDDQNPolicy(state_size, get_action_size(), Namespace(**{'use_gpu': False}), evaluation_mode=True)
        if load_second_policy == "PPO":
            inter_policy = PPOPolicy(state_size, get_action_size())
        policy = DecisionPointAgent(local_env, state_size, get_action_size(), inter_policy)
    elif load_policy == "MultiDecisionAgent":
        policy = MultiDecisionAgent(local_env, state_size, get_action_size(), Namespace(**{'use_gpu': False}),
                                    evaluation_mode=True)
    else:
        policy = PPOPolicy(state_size, get_action_size(), use_replay_buffer=False,
                           in_parameters=Namespace(**{'use_gpu': False}))

    policy.load(checkpoint)

    policy.reset(local_env)
    observation = flatland_observation.get_many(list(range(nb_agents)))

    if DO_RENDER:
        env_renderer = RenderTool(local_env, gl="PGL")
        env_renderer.set_new_rail()
    else:
        env_renderer = None

    print("Evaluation {}: {} agents in {}x{}".format(evaluation_number, nb_agents, local_env.width, local_env.height))

    # Now we enter into another infinite loop where we
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    steps = 0

    # Bookkeeping
    time_taken_by_controller = []
    time_taken_per_step = []

    # Action cache: keep track of last observation to avoid running the same inferrence multiple times.
    # This only makes sense for deterministic policies.
    agent_last_obs = {}
    agent_last_action = {}
    nb_hit = 0

    policy.start_episode(train=False)
    while True:
        try:
            #####################################################################
            # Evaluation of a single episode
            #####################################################################
            steps += 1
            obs_time, agent_time, step_time = 0.0, 0.0, 0.0
            no_ops_mode = False

            if not check_if_all_blocked(env=local_env):
                time_start = time.time()
                action_dict = {}
                policy.start_step(train=False)
                for agent_handle in range(nb_agents):
                    action = map_actions(RailEnvActions.DO_NOTHING)
                    if info['action_required'][agent_handle]:
                        if agent_handle in agent_last_obs and \
                                np.all(agent_last_obs[agent_handle] == observation[agent_handle]):
                            # cache hit
                            action = agent_last_action[agent_handle]
                            nb_hit += 1
                        else:
                            action = policy.act(agent_handle, observation[agent_handle], eps=EPSILON)

                    action_dict[agent_handle] = action

                    if USE_ACTION_CACHE:
                        agent_last_obs[agent_handle] = observation[agent_handle]
                        agent_last_action[agent_handle] = action

                policy.end_step(train=False)
                agent_time = time.time() - time_start
                time_taken_by_controller.append(agent_time)

                time_start = time.time()
                _, all_rewards, done, info = remote_client.env_step(map_actions(action_dict))
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

                time_start = time.time()
                observation = flatland_observation.get_many(list(range(nb_agents)))
                obs_time = time.time() - time_start

            else:
                # Fully deadlocked: perform no-ops
                no_ops_mode = True

                time_start = time.time()
                _, all_rewards, done, info = remote_client.env_step({})
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

            nb_agents_done = 0
            nb_agent_active = 0
            for i_agent, agent in enumerate(local_env.agents):
                # manage the boolean flag to check if all agents are indeed done (or done_removed)
                if agent.state in [TrainState.DONE]:
                    nb_agents_done += 1
                if agent.state in [TrainState.STOPPED, TrainState.MALFUNCTION, TrainState.MOVING]:
                    nb_agent_active += 1

            if DO_RENDER:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=True,
                    show_predictions=False
                )

            if VERBOSE or done['__all__']:
                print(
                    "Step {}/{}\tAgents active: {}\tdone: {}\t Obs time {:.3f}s\t Inference time {:.5f}s\t Step "
                    "time {:.3f}s".format(
                        str(steps).zfill(4),
                        max_nb_steps,
                        nb_agent_active,
                        nb_agents_done,
                        obs_time,
                        agent_time,
                        step_time
                    ), end="\r")

            if done['__all__']:
                # When done['__all__'] == True, then the evaluation of this
                # particular Env instantiation is complete, and we can break out
                # of this loop, and move onto the next Env evaluation
                print()
                break

        except TimeoutException as err:
            # A timeout occurs, won't get any reward for this episode :-(
            # Skip to next episode as further actions in this one will be ignored.
            # The whole evaluation will be stopped if there are 10 consecutive timeouts.
            print("Timeout! Will skip this episode and go to the next.", err)
            break

    policy.end_episode(train=False)

    if DO_RENDER:
        env_renderer.close_window()

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(),
          np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("=" * 100)

print("Evaluation of all environments complete!")
########################################################################
# Submit your Results
#
# Please do not forget to include this call, as this triggers the
# final computation of the score statistics, video generation, etc
# and is necessary to have your submission marked as successfully evaluated
########################################################################
print(remote_client.submit())
