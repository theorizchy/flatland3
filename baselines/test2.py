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
checkpoint = "tembusu_checkpoints/221029180929-19900.pth"

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
policy = DDDQNPolicy(state_size, action_size, Namespace(**{'use_gpu': True}), evaluation_mode=True)

if os.path.isfile(checkpoint):
    policy.qnetwork_local = torch.load(checkpoint)
else:
    print("[WARNING] Checkpoint not found, using untrained policy! (path: {})".format(checkpoint))

#####################################################################
# Main evaluation loop
#####################################################################

time_start = time.time()
obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
env_creation_time = time.time() - time_start

print("[INFO] Env Creation Time : ", env_creation_time)

nb_agents = len(env.agents)

tree_observation.set_env(env)
tree_observation.reset()
observation = tree_observation.get_many(list(range(nb_agents)))


steps = 0

env_renderer = RenderTool(env, gl="PGL")

max_steps = env._max_episode_steps
action_dict = dict()
agent_obs = [None] * env.get_num_agents()    

score = 0.0
final_step = 0
for step in range(max_steps):
    env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=True,
                    show_predictions=True
    )
    time.sleep(2)
    for agent in env.get_agent_handles():
        if obs[agent]:
            agent_obs[agent] = normalize_observation(obs[agent], tree_depth=observation_tree_depth, observation_radius=observation_radius)

        action = 0
        if info['action_required'][agent]:
            action = policy.act(agent_obs[agent], eps=0.0)
        action_dict.update({agent: action})

    obs, all_rewards, done, info = env.step(action_dict)

    for agent in env.get_agent_handles():
        score += all_rewards[agent]

    final_step = step

    if done['__all__']:
        break
