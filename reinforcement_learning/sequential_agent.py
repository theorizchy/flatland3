import sys
import PIL
import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
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

x_dim = np.random.randint(30, 35)
y_dim = np.random.randint(30, 35)
n_agents = np.random.randint(3, 8)

env = RailEnv(
    width=x_dim,
    height=y_dim,
    rail_generator=sparse_rail_generator(),
    line_generator=sparse_line_generator(),
    obs_builder_object=TreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv()),
    number_of_agents=n_agents)
env.reset(True, True)

env_renderer = RenderTool(env, gl="PGL", )
handle = env.get_agent_handles()
n_episodes = 1
max_steps = 100 * (env.height + env.width)
record_images = True
policy = OrderedPolicy()
action_dict = dict()
frame_list = []
for trials in range(1, n_episodes + 1):

    # Reset environment
    obs, info = env.reset(True, True)
    done = env.dones
    env_renderer.reset()
    frame_step = 0

    # Run episode
    for step in range(max_steps):
        env_renderer.render_env(show=False, show_observations=False, show_predictions=True)

        if record_images:
            frame_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))
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
            print(done)
            if record_images:
                frame_list[0].save(f"flatland_sequential_agent_{trials}.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)
                frame_list=[]
            break
