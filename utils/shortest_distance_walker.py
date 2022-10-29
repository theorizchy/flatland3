import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv, RailEnvActions

from utils.fast_methods import fast_count_nonzero, fast_argmax


class ShortestDistanceWalker:
    def __init__(self, env: RailEnv):
        self.env = env

    def walk(self, handle, position, direction):
        min_distances = [np.inf, np.inf, np.inf, np.inf]
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        num_transitions = fast_count_nonzero(possible_transitions)
        if num_transitions == 1:
            new_direction = fast_argmax(possible_transitions)
            new_position = get_new_position(position, new_direction)
            dist = self.env.distance_map.get()[handle, new_position[0], new_position[1], new_direction]
            min_distances[new_direction] = dist
            return new_position, new_direction, dist, RailEnvActions.MOVE_FORWARD, possible_transitions, min_distances
        else:
            positions = []
            directions = []
            min_distances = []
            for new_direction in [(direction + i) % 4 for i in range(-1, 3)]:
                if possible_transitions[new_direction]:
                    new_position = get_new_position(position, new_direction)
                    min_dist = self.env.distance_map.get()[handle, new_position[0], new_position[1], new_direction]
                    if np.isinf(min_dist):
                        min_dist = 9999999
                    min_distances.append(min_dist)
                    positions.append(new_position)
                    directions.append(new_direction)
                else:
                    min_distances.append(np.inf)
                    positions.append(None)
                    directions.append(None)

        a = self.get_action(handle, min_distances)
        return positions[a], directions[a], min_distances[a], a + 1, possible_transitions, min_distances

    def get_action(self, handle, min_distances):
        return np.argmin(min_distances)

    def callback(self, handle, agent, position, direction, action, possible_transitions):
        return True

    def get_agent_position_and_direction(self, handle):
        agent = self.env.agents[handle]
        if agent.position is not None:
            position = agent.position
        else:
            position = agent.initial_position
        direction = agent.direction
        return position, direction

    def walk_to_target(self, handle, position=None, direction=None, break_max_step=500, break_at_switch=0):
        if position is None and direction is None:
            position, direction = self.get_agent_position_and_direction(handle)
        elif position is None:
            position, _ = self.get_agent_position_and_direction(handle)
        elif direction is None:
            _, direction = self.get_agent_position_and_direction(handle)

        agent = self.env.agents[handle]
        step = 0
        # a switch to break == switch cluster counting
        switch_cluster = 1
        while (position != agent.target) and (step < break_max_step):
            if break_at_switch >= 1:
                possible_transitions = self.env.rail.get_transitions(*position, direction)
                num_transitions = fast_count_nonzero(possible_transitions)
                if num_transitions > 1:
                    if break_at_switch == 1:
                        break
                    break_at_switch -= switch_cluster
                    switch_cluster = 0
                else:
                    switch_cluster = 1
            position, direction, dist, action, possible_transitions, min_distances = \
                self.walk(handle, position, direction)
            if position is None:
                break
            if not self.callback(handle, agent, position, direction, action,
                                 self.env.rail.get_transitions(*position, direction)):
                break
            step += 1

    def callback_one_step(self, handle, agent, position, direction, action, possible_transitions):
        return True

    def walk_one_step(self, handle):
        agent = self.env.agents[handle]
        if agent.position is not None:
            position = agent.position
        else:
            position = agent.initial_position
        direction = agent.direction
        possible_transitions = (0, 1, 0, 0)
        if (position != agent.target):
            new_position, new_direction, dist, action, possible_transitions, min_distances = \
                self.walk(handle, position, direction)
            if new_position is None:
                return position, direction, RailEnvActions.MOVE_FORWARD, possible_transitions
            if not self.callback_one_step(handle, agent, new_position, new_direction, action, possible_transitions):
                pass
            return new_position, new_direction, action, possible_transitions
        else:
            print('target', position, agent.target)
            return position, direction, RailEnvActions.STOP_MOVING, possible_transitions
