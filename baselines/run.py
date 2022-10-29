import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import time

import torch
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.evaluators.client import TimeoutException

from utils.deadlock_check import check_if_all_blocked

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from utils.observation_utils import normalize_observation

####################################################
# EVALUATION PARAMETERS

# Print per-step logs
VERBOSE = False

# Checkpoint to use (remember to push it!)
checkpoint = "checkpoints/sample-checkpoint.pth"

# Use last action cache
USE_ACTION_CACHE = True

# Observation parameters (must match training parameters!)
observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

####################################################

remote_client = FlatlandRemoteClient()

# Observation builder
predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

# Calculates state and action sizes
n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
state_size = tree_observation.observation_dim * n_nodes
action_size = 5

# Creates the policy. No GPU on evaluation server.
policy = DDDQNPolicy(state_size, action_size, Namespace(**{'use_gpu': False}), evaluation_mode=True)

if os.path.isfile(checkpoint):
    policy.qnetwork_local = torch.load(checkpoint)
else:
    print("[WARNING] Checkpoint not found, using untrained policy! (path: {})".format(checkpoint))

#####################################################################
# Main evaluation loop
#####################################################################
evaluation_number = 0

while True:
    evaluation_number += 1
    print("[INFO] EPISODE_START : {}".format(evaluation_number))


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

    print("[INFO] Env Path : ", remote_client.current_env_path)
    print("[INFO] Env Creation Time : ", env_creation_time)

    local_env = remote_client.env
    nb_agents = len(local_env.agents)
    max_nb_steps = local_env._max_episode_steps

    tree_observation.set_env(local_env)
    tree_observation.reset()
    observation = tree_observation.get_many(list(range(nb_agents)))

    print("[INFO] Evaluation {}: {} agents in {}x{}".format(evaluation_number, nb_agents, local_env.width, local_env.height))

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
                for agent in range(nb_agents):
                    if observation[agent] and info['action_required'][agent]:
                        if agent in agent_last_obs and np.all(agent_last_obs[agent] == observation[agent]):
                            # cache hit
                            action = agent_last_action[agent]
                            nb_hit += 1
                        else:
                            # otherwise, run normalization and inference
                            norm_obs = normalize_observation(observation[agent], tree_depth=observation_tree_depth, observation_radius=observation_radius)
                            action = policy.act(norm_obs, eps=0.0)

                        action_dict[agent] = action

                        if USE_ACTION_CACHE:
                            agent_last_obs[agent] = observation[agent]
                            agent_last_action[agent] = action
                agent_time = time.time() - time_start
                time_taken_by_controller.append(agent_time)

                time_start = time.time()

                try:
                    _, all_rewards, done, info = remote_client.env_step(action_dict)
                except:
                    print("[ERR] DONE BUT step()_1 CALLED")

                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

                time_start = time.time()
                observation = tree_observation.get_many(list(range(nb_agents)))
                obs_time = time.time() - time_start

            else:
                # Fully deadlocked: perform no-ops
                no_ops_mode = True

                time_start = time.time()

                try:
                    _, all_rewards, done, info = remote_client.env_step({})
                except:
                    print("[ERR] DONE BUT step()_2 CALLED")                
                
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

            nb_agents_done = sum(done[idx] for idx in local_env.get_agent_handles())

            if VERBOSE or done['__all__']:
                print("[INFO] Step {}/{}\tAgents done: {}\t Obs time {:.3f}s\t Inference time {:.5f}s\t Step time {:.3f}s\t Cache hits {}\t No-ops? {}".format(
                    str(steps).zfill(4),
                    max_nb_steps,
                    nb_agents_done,
                    obs_time,
                    agent_time,
                    step_time,
                    nb_hit,
                    no_ops_mode
                ), end="\r")

            if done['__all__']:
                # When done['__all__'] == True, then the evaluation of this
                # particular Env instantiation is complete, and we can break out
                # of this loop, and move onto the next Env evaluation
                print("[INFO] EPISODE_DONE : ", evaluation_number)
                break

        except TimeoutException as err:
            # A timeout occurs, won't get any reward for this episode :-(
            # Skip to next episode as further actions in this one will be ignored.
            # The whole evaluation will be stopped if there are 10 consecutive timeouts.
            print("[ERR] Timeout! Will skip this episode and go to the next.", err)
            break

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
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
