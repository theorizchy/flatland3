import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import time

from flatland.utils.rendertools import RenderTool
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from utils.observation_utils import normalize_observation

##################### TO CHANGE ####################
checkpoint = "checkpoints/sample-checkpoint.pth"
n_agents = 5
x_dim = 30
y_dim = 30
n_cities = 2
max_rails_between_cities = 2
max_rail_pairs_in_city = 1
seed = 0

malfunction_rate = 0
min_duration = 1
max_duration = 5

show_observations = True
show_predictions = True
#################### END OF STUFF TO CHANGE ###################




########################### CREATE ENV ##############################
def create_rail_env(tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=malfunction_rate,
        min_duration=min_duration,
        max_duration=max_duration
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        line_generator=sparse_line_generator(seed=seed),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )
#########################################################

########################### Observation builder ###########################

observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

use_gpu = True
if checkpoint == "checkpoints/sample-checkpoint.pth":
    use_gpu = False
if not os.path.isfile(checkpoint):
    use_gpu = False
    print("[WARNING] Checkpoint not found, using untrained policy! (path: {})".format(checkpoint))


predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

env = create_rail_env(tree_observation)

# Calculates state and action sizes
n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
state_size = tree_observation.observation_dim * n_nodes
action_size = 5

# Creates the policy. No GPU on evaluation server.
policy = DDDQNPolicy(state_size, action_size, Namespace(**{'use_gpu': use_gpu}), evaluation_mode=True)

if os.path.isfile(checkpoint):
    policy.qnetwork_local = torch.load(checkpoint)
###########################################################################


#####################################################################
# Main evaluation loop
#####################################################################



obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

nb_agents = len(env.agents)
tree_observation.set_env(env)
tree_observation.reset()
observation = tree_observation.get_many(list(range(nb_agents)))


steps = 0

env_renderer = RenderTool(env, gl="PGL", screen_height=2160, screen_width=3840)

max_steps = env._max_episode_steps
action_dict = dict()
agent_obs = [None] * env.get_num_agents()    

score = 0.0
final_step = 0
for step in range(max_steps):
    env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=show_observations,
                    show_predictions=show_predictions
    )
    time.sleep(0.5)
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
