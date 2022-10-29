import sys
import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from reinforcement_learning.ordered_policy import OrderedPolicy

"""
This file shows how to move agents in a sequential way: it moves the trains one by one, following a shortest path strategy.
This is obviously very slow, but it's a good way to get familiar with the different Flatland components: RailEnv, TreeObsForRailEnv, etc...

multi_agent_training.py is a better starting point to train your own solution!
"""

np.random.seed(2)

x_dim = np.random.randint(8, 20)
y_dim = np.random.randint(8, 20)
n_agents = np.random.randint(3, 8)
n_goals = n_agents + np.random.randint(0, 3)
min_dist = int(0.75 * min(x_dim, y_dim))

env = RailEnv(
    width=x_dim,
    height=y_dim,
    rail_generator=complex_rail_generator(
        nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
        max_dist=99999,
        seed=0
    ),
    schedule_generator=complex_schedule_generator(),
    obs_builder_object=TreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv()),
    number_of_agents=n_agents)
env.reset(True, True)

tree_depth = 1
observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())
env_renderer = RenderTool(env, gl="PGL", )
handle = env.get_agent_handles()
n_episodes = 10
max_steps = 100 * (env.height + env.width)
record_images = False
policy = OrderedPolicy()
action_dict = dict()

for trials in range(1, n_episodes + 1):
    # Reset environment
    obs, info = env.reset(True, True)
    done = env.dones
    env_renderer.reset()
    frame_step = 0

    # Run episode
    for step in range(max_steps):
        env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

        if record_images:
            env_renderer.gl.save_image("./Images/flatland_frame_{:04d}.bmp".format(frame_step))
            frame_step += 1

        # Action
        acting_agent = 0
        for a in range(env.get_num_agents()):
            if done[a]:
                acting_agent += 1
            if a == acting_agent:
                action = policy.act(obs[a])
            else:
                action = 4
            action_dict.update({a: action})

        # Environment step
        obs, all_rewards, done, _ = env.step(action_dict)

        if done['__all__']:
            break
