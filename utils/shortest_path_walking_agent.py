import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.policy import HeuristicPolicy
from utils.agent_action_config import convert_default_rail_env_action
from utils.fast_methods import fast_count_nonzero


class ShortestPathWalkingAgent(HeuristicPolicy):
    def __init__(self, env: RailEnv):
        print(">> ShortestPathWalkingAgent")
        self.env = env

    def reset(self, env):
        self.env = env

    def act(self, handle, state, eps=0.):
        agent = self.env.agents[handle]

        if agent.position is not None:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

        num_transitions = fast_count_nonzero(possible_transitions)
        if num_transitions == 1:
            return convert_default_rail_env_action(RailEnvActions.MOVE_FORWARD)

        return self.get_action(handle, possible_transitions)

    def get_action(self, handle, possible_transitions):
        agent = self.env.agents[handle]
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[direction]:
                new_position = get_new_position(agent.position, direction)
                min_distances.append(
                    self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
            else:
                min_distances.append(np.inf)

        distance_estimator = [0, 0, 0]
        distance_estimator[np.argmin(min_distances)] = 1

        return convert_default_rail_env_action(np.argmax(distance_estimator) + 1)
