import os
import sys
from argparse import  ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import time

import psutil
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from flatland.envs.step_utils.states import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

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
checkpoint = "checkpoints/221029180929-19900.pth"

# Observation parameters (must match training parameters!)
observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

####################################################

#######################################################
def create_rail_env(env_params, tree_observation):

    # Environment parameters
    n_agents = 5
    x_dim = 30
    y_dim = 30
    n_cities = 2
    max_rails_between_cities = 2
    max_rail_pairs_in_city = 1
    seed = 0

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/5,
        min_duration=20,
        max_duration=50
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )
#########################################################


parser = ArgumentParser()
parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=2500, type=int)
parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=0, type=int)
parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=0, type=int)
parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)
parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=100, type=int)
parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
parser.add_argument("--eps_decay", help="exploration decay", default=0.99, type=float)
parser.add_argument("--buffer_size", help="replay buffer size", default=int(1e5), type=int)
parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
parser.add_argument("--restore_replay_buffer", help="replay buffer to restore", default="", type=str)
parser.add_argument("--save_replay_buffer", help="save replay buffer at each evaluation interval", default=False, type=bool)
parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
parser.add_argument("--tau", help="soft update of target parameters", default=1e-3, type=float)
parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)
parser.add_argument("--hidden_size", help="hidden size (2 fc layers)", default=128, type=int)
parser.add_argument("--update_every", help="how often to update the network", default=8, type=int)
parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=1, type=int)
parser.add_argument("--render", help="render 1 episode in 100", default=False, type=bool)
train_params = parser.parse_args()

# Observation builder
predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

env = create_rail_env(train_params, tree_observation)

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

time_start = time.time()
observation, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
env_creation_time = time.time() - time_start

print("[INFO] Env Creation Time : ", env_creation_time)

local_env = env
nb_agents = len(env.agents)
max_nb_steps = local_env._max_episode_steps

tree_observation.set_env(local_env)
tree_observation.reset()
observation = tree_observation.get_many(list(range(nb_agents)))


steps = 0

# Bookkeeping
time_taken_by_controller = []
time_taken_per_step = []

# Action cache: keep track of last observation to avoid running the same inferrence multiple times.
# This only makes sense for deterministic policies.
agent_last_obs = {}
agent_last_action = {}
nb_hit = 0


env_renderer = RenderTool(local_env, gl="PGL")
for i in range(500):
    print(f"This is the {i}th iter")
    try:
        #####################################################################
        # Evaluation of a single episode
        #####################################################################

        env_renderer.render_env(
            show=True,
            frames=False,
            show_observations=True,
            show_predictions=False
        )
        #time.sleep(2)


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

            agent_time = time.time() - time_start
            time_taken_by_controller.append(agent_time)

            time_start = time.time()

            try:
                # TODO
                _, all_rewards, done, info = env.step(action_dict)
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
                _, all_rewards, done, info = env.step({})
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
            print("done")
            break

    except TimeoutException as err:
        # A timeout occurs, won't get any reward for this episode :-(
        # Skip to next episode as further actions in this one will be ignored.
        # The whole evaluation will be stopped if there are 10 consecutive timeouts.
        print("[ERR] Timeout! Will skip this episode and go to the next.", err)
        break

#np_time_taken_by_controller = np.array(time_taken_by_controller)
#np_time_taken_per_step = np.array(time_taken_per_step)
#print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
#print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
#print("=" * 100)

#print("Evaluation of all environments complete!")
########################################################################
# Submit your Results
#
# Please do not forget to include this call, as this triggers the
# final computation of the score statistics, video generation, etc
# and is necessary to have your submission marked as successfully evaluated
########################################################################
