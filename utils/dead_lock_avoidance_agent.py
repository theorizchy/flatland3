from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.policy import HeuristicPolicy, DummyMemory
from utils.agent_action_config import convert_default_rail_env_action
from utils.agent_can_choose_helper import AgentCanChooseHelper
from utils.shortest_distance_walker import ShortestDistanceWalker


class DeadlockAvoidanceObservation(DummyObservationBuilder):
    def __init__(self):
        self.counter = 0

    def get_many(self, handles: Optional[List[int]] = None) -> bool:
        self.counter += 1
        obs = np.ones((len(handles), 2))
        for handle in handles:
            obs[handle][0] = handle
            obs[handle][1] = self.counter
        return obs


class DeadlockAvoidanceShortestDistanceWalker(ShortestDistanceWalker):
    def __init__(self, env: RailEnv, agent_positions, switches, stop_walking_when_opposite_agent_encountered=False):
        super().__init__(env)

        self.stop_walking_when_opposite_agent_encountered = stop_walking_when_opposite_agent_encountered

        # shortest_distance_agent_map is represented as grid with 1 when the cell is part of the shortest path,
        # 0 otherwise. It contains all cells on the along the shortest path except switch cells and all cell after
        # the walker encounters an opposite agent - this means it's a sub path which is opposite agent free
        self.shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                     self.env.height,
                                                     self.env.width),
                                                    dtype=int) - 1

        # shortest_distance_agent_map is represented as grid with one when the cell is part of the shortest path,
        # zero otherwise. No cell where skipped.
        self.full_shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                          self.env.height,
                                                          self.env.width),
                                                         dtype=int) - 1

        self.agent_positions = agent_positions

        self.opp_agent_map = {}
        self.same_agent_map = {}
        self.switches = switches

    def getData(self):
        return self.shortest_distance_agent_map, self.full_shortest_distance_agent_map

    def callback(self, handle, agent, position, direction, action, possible_transitions):
        ret_value = True
        opp_a = self.agent_positions[position]
        if opp_a != -1 and opp_a != handle:
            if self.env.agents[opp_a].direction != direction:
                d = self.opp_agent_map.get(handle, [])
                if opp_a not in d:
                    d.append(opp_a)
                self.opp_agent_map.update({handle: d})
            else:
                if len(self.opp_agent_map.get(handle, [])) == 0:
                    d = self.same_agent_map.get(handle, [])
                    if opp_a not in d:
                        d.append(opp_a)
                    self.same_agent_map.update({handle: d})

        if opp_a != handle:
            if len(self.opp_agent_map.get(handle, [])) == 0:
                if self.switches.get(position, None) is None:
                    self.shortest_distance_agent_map[(handle, position[0], position[1])] = 1
            else:
                if self.stop_walking_when_opposite_agent_encountered:
                    ret_value = False

        self.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 1
        return ret_value


class DeadLockAvoidanceAgent(HeuristicPolicy):
    def __init__(self, env: RailEnv, action_size, enable_eps=False, show_debug_plot=False, break_at_switch=0):
        print(">> DeadLockAvoidance")
        self.env = env
        self.memory = DummyMemory()
        self.loss = 0
        self.action_size = action_size
        self.agent_can_move = {}
        self.agent_can_move_value = {}
        self.switches = {}
        self.show_debug_plot = show_debug_plot
        self.enable_eps = enable_eps
        self.break_at_switch = break_at_switch
        self.agent_can_choose_helper: AgentCanChooseHelper = None

    def step(self, handle, state, action, reward, next_state, done):
        '''
        not used interface method
        '''
        pass

    def act(self, handle, state, eps=0.):
        '''
        Estimates the next agents action
        :param handle: agents reference
        :param state: not used (just interface parameter)
        :param eps: if eps enable the eps used for randomness
        :return: action
        '''
        # Epsilon-greedy action selection
        if self.enable_eps:
            if np.random.random() < eps:
                return np.random.choice(np.arange(self.action_size))

        agent_position, agent_direciton, agent_status, agent_target = \
            self.agent_can_choose_helper.get_agent_position_and_direction(handle)

        agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all = \
            self.agent_can_choose_helper.check_agent_decision(agent_position, agent_direciton)
        if agent_status in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION] and \
                not agents_on_switch and \
                not agents_near_to_switch:
            return convert_default_rail_env_action(RailEnvActions.MOVE_FORWARD)

        _, action = self.check_agent_can_move(handle)
        return convert_default_rail_env_action(action)

    def check_agent_can_move(self, handle):
        check = self.agent_can_move.get(handle, None)
        if check is None:
            return False, RailEnvActions.STOP_MOVING
        return True, check[3]

    def get_agent_can_move_value(self, handle):
        '''
        Returns the stored free cell min value for given agent
        :param handle: the handle to the agent
        :return: cell free move value
        '''
        return self.agent_can_move_value.get(handle, np.inf)

    def reset(self, env):
        '''
        Reset the deadlock avoidance agent and reset the environment
        :param env: RailEnv object
        '''
        self.env = env
        self.agent_positions = None
        self.shortest_distance_walker = None

        if self.agent_can_choose_helper is None:
            self.agent_can_choose_helper = AgentCanChooseHelper()
        self.agent_can_choose_helper.reset(self.env)

    def start_step(self, train):
        '''
        Computes for current situation the deadlock avoidance maps and estimates whether agents can walk or the have
        to stop
        '''
        self.build_agent_position_map()
        self.generate_shortest_path_agent_walking_maps()
        self.apply_deadlock_avoidance_heuristic(threshold=1.0, opp_agent_threshold_factor=1.0)
        self.agent_can_choose_helper.reset_switch_cluster_occupied()

    def end_step(self, train):
        '''
        not used interface method
        '''
        pass

    def get_actions(self):
        '''
        not used interface method
        '''
        pass

    def build_agent_position_map(self):
        '''
        build map with agent positions (only active agents)
        '''
        self.agent_positions = np.zeros((self.env.height, self.env.width), dtype=int) - 1
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                if agent.position is not None:
                    self.agent_positions[agent.position] = handle

    def generate_shortest_path_agent_walking_maps(self):
        '''
        This methods generates for all agents the shortest walk maps. The method uses the DeadlockAvoidanceShortestDistanceWalker
        to compute the paths.
        '''
        self.shortest_distance_walker = DeadlockAvoidanceShortestDistanceWalker(self.env,
                                                                                self.agent_positions,
                                                                                self.agent_can_choose_helper.switches)
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state <= TrainState.MOVING:
                self.shortest_distance_walker.walk_to_target(handle, break_at_switch=self.break_at_switch)

    def get_number_of_free_cells_on_agents_path(self, handle, agents_path_map, opp_agents):
        '''
        This method calculates for a given agent the number of free cells. The number of free cell gets extracted by
        the method calculate_map_differences

        :param handle:
        :param agents_path_map:
        :return: number of free cells
        '''
        _, full_shortest_distance_agent_map = self.shortest_distance_walker.getData()
        return self.calculate_map_differences(agents_path_map,
                                              opp_agents,
                                              full_shortest_distance_agent_map)

    def apply_deadlock_avoidance_heuristic_threshold(self, min_value, opp_agents, threshold,
                                                     opp_agent_threshold_factor):
        '''
        This method estimates whether an agent can move or it has to stop to avoid deadlock(s)

        :param min_value: number of free cell : see get_number_off_free_cells_on_agents_path
        :param opp_agents: an array of opposite traveling agents
        :param threshold: minimal number of free cells (min_value)
        :param opp_agent_threshold_factor: occupation ration for number of opp_agents
        :return: [true/false]
        '''
        return (min_value > (threshold + opp_agent_threshold_factor * len(opp_agents)))

    def apply_deadlock_avoidance_heuristic(self, threshold, opp_agent_threshold_factor=2.0):
        '''
        This method puts for each agent a flag, whether it can walk or has to stop (see: self.agent_can_move) and
        stores the min_value in a global dict (self.agent_can_move_value)

        :param threshold: minimal number of free cells (min_value)
        :param opp_agent_threshold_factor: occupation ration for number of opp_agents
        '''
        self.agent_can_move = {}
        shortest_distance_agent_map, full_shortest_distance_agent_map = self.shortest_distance_walker.getData()
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state < TrainState.DONE:
                opp_agents = self.shortest_distance_walker.opp_agent_map.get(handle, [])
                min_value = self.get_number_of_free_cells_on_agents_path(handle,
                                                                         shortest_distance_agent_map[handle],
                                                                         opp_agents)
                self.agent_can_move_value.update({handle: min_value})
                opp_agents = self.shortest_distance_walker.opp_agent_map.get(handle, [])
                if self.apply_deadlock_avoidance_heuristic_threshold(min_value,
                                                                     opp_agents,
                                                                     threshold,
                                                                     opp_agent_threshold_factor):
                    next_position, next_direction, action, _ = self.shortest_distance_walker.walk_one_step(handle)
                    self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})

        if self.show_debug_plot:
            a = np.floor(np.sqrt(self.env.get_num_agents()))
            b = np.ceil(self.env.get_num_agents() / a)
            for handle in range(self.env.get_num_agents()):
                plt.subplot(a, b, handle + 1)
                plt.imshow(full_shortest_distance_agent_map[handle] + shortest_distance_agent_map[handle])
            plt.show(block=False)
            plt.pause(0.01)

    def calculate_map_differences(self,
                                  my_shortest_walking_path,
                                  opp_agents,
                                  full_shortest_distance_agent_map):
        '''
        This methods computes a given path the number of free cells. A cell is free if it is part of the agents
        walking path and if no opposite traveling agents requires this cell.

        :param my_shortest_walking_path: map of cells where the agents walks on (1 if the cell is part of the path
        otherwise 0)
        :param opp_agents: array of handles of opposite agents
        :param full_shortest_distance_agent_map: all path maps
        :return: minimal number of free cells
        '''
        agent_positions_map = (self.agent_positions > -1).astype(int)
        min_value = np.inf
        for opp_a in opp_agents:
            opp_map = full_shortest_distance_agent_map[opp_a]
            delta = ((my_shortest_walking_path - opp_map - agent_positions_map) > 0).astype(int)
            min_value = min(min_value, np.sum(delta))
        return min_value

    def save(self, filename):
        '''
        not used interface method
        '''
        pass

    def load(self, filename):
        '''
        not used interface method
        '''
        pass
