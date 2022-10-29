from collections import namedtuple
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions, RailEnv
from utils.agent_action_config import get_action_size
from utils.agent_can_choose_helper import AgentCanChooseHelper
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent
from utils.deadlock_check import get_agent_targets, get_agent_positions_next_step
from utils.fast_methods import fast_count_nonzero, fast_argmax, fast_isclose
from utils.shortest_distance_walker import ShortestDistanceWalker
from utils.shortest_path_walking_agent import ShortestPathWalkingAgent

AgentMapCache = namedtuple("AgentMapCache", "position direction status shortest_path_map branching_path_map")

"""
LICENCE for the FastTreeObs Observation Builder  

The observation can be used freely and reused for further submissions. Only the author needs to be referred to
/mentioned in any submissions - if the entire observation or parts, or the main idea is used.

Author: Adrian Egli (adrian.egli@gmail.com)

[Linkedin](https://www.researchgate.net/profile/Adrian_Egli2)
[Researchgate](https://www.linkedin.com/in/adrian-egli-733a9544/)
"""

OBSERVATION_DIM = 56


def probabilistic_value_mixing(a, b):
    # return np.maximum(a, b)
    return 1.0 - ((1.0 - a) * (1.0 - b))


class ProjectPathWalker(ShortestDistanceWalker):
    def __init__(self, env: RailEnv, project_weighted_distance):
        super(ProjectPathWalker, self).__init__(env)
        self.project_weighted_distance = project_weighted_distance
        self.projected_density_map: np.ndarray = np.zeros((4, self.env.height, self.env.width))
        self.branch_entry_points = []
        self.distance_map = self.env.distance_map.get()
        self.weight = 1.0

    def callback(self, handle, agent, position, direction, action, possible_transitions):
        cur_dist = self.distance_map[handle, position[0], position[1], direction]
        if np.isinf(cur_dist):
            return False

        branching_dist = []
        local_branch_entry_points = []
        summed_branching_dist = 0
        if fast_count_nonzero(possible_transitions) > 1:
            for dir_loop in range(4):
                if possible_transitions[dir_loop] == 1:
                    next_position = get_new_position(position, dir_loop)
                    dist = self.distance_map[handle, next_position[0], next_position[1], dir_loop]
                    if not np.isinf(dist):
                        summed_branching_dist += dist
                        branching_dist.append(dist)
                        local_branch_entry_points.append((next_position, dir_loop, dist))

        for k in range(len(local_branch_entry_points)):
            dat = local_branch_entry_points[k]
            dist = dat[2]
            if cur_dist - 1 < dist:
                self.branch_entry_points.append((dat[0], dat[1], dat[2], summed_branching_dist))

        self.projected_density_map[direction, position[0], position[1]] += self.weight
        if self.project_weighted_distance and len(branching_dist) > 0:
            self.weight *= 1.0 - np.min(branching_dist) / summed_branching_dist
        return True

    def get_branch_entry_points(self):
        return self.branch_entry_points

    def get_projected_density_map(self):
        return self.projected_density_map


class FastTreeObs(ObservationBuilder):

    def __init__(self):
        self.observation_dim = OBSERVATION_DIM
        self.previous_observations = {}
        self.agent_can_choose_helper: AgentCanChooseHelper = None
        self.shortest_path_walking_agent: ShortestPathWalkingAgent = None
        self.dead_lock_avoidance_agent: DeadLockAvoidanceAgent = None
        self.enabled_dead_lock_avoidance_agent = True
        self.agents_path_maps_cache = {}

    def reset(self):
        self.previous_observations = {}
        if self.agent_can_choose_helper is None:
            self.agent_can_choose_helper = AgentCanChooseHelper()
        self.agent_can_choose_helper.reset(self.env)

        if self.shortest_path_walking_agent is None:
            self.shortest_path_walking_agent = ShortestPathWalkingAgent(self.env)
        else:
            self.shortest_path_walking_agent.end_episode(False)
        self.shortest_path_walking_agent.reset(self.env)
        self.shortest_path_walking_agent.start_episode(False)

        if self.enabled_dead_lock_avoidance_agent:
            if self.dead_lock_avoidance_agent is None:
                self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(self.env, get_action_size(),
                                                                        enable_eps=False, break_at_switch=True)
            else:
                self.dead_lock_avoidance_agent.end_episode(False)
            self.dead_lock_avoidance_agent.reset(self.env)
            self.dead_lock_avoidance_agent.start_episode(False)

        self.agents_path_maps_cache = {}

    def check_intersecting_possible_transitions(self,
                                                possible_transitions_agent_1, agent_dir1,
                                                possible_transitions_agent_2, agent_dir2):
        num_transitions_agent_1 = fast_count_nonzero(possible_transitions_agent_1)
        num_transitions_agent_2 = fast_count_nonzero(possible_transitions_agent_2)
        if num_transitions_agent_1 < 2 and num_transitions_agent_2 < 2:
            return agent_dir1 == agent_dir2
        if num_transitions_agent_1 == 1:
            if num_transitions_agent_2 > 1:
                return 0
            else:
                return agent_dir1 == agent_dir2
        if num_transitions_agent_2 == 1:
            if num_transitions_agent_1 > 1:
                return 0
            else:
                return agent_dir1 == agent_dir2

        return agent_dir1 == agent_dir2

    def getOtherAgentInfo(self, handle, position, direction):
        agent_index_at_position = self.agent_can_choose_helper.get_agent_positions()[position]
        other_agent_found = 0
        same_agent_found = 0
        opp_agent_found = 0
        opp_agent_index_comp = 0
        if position is None or position[0] == -1 or position[1] == -1:
            return -1, -1, -1, -1
        if agent_index_at_position != -1 and agent_index_at_position != handle:
            other_agent_found = 1
            possible_transitions = \
                self.env.rail.get_transitions(*position, direction)
            opp_possible_transitions = \
                self.env.rail.get_transitions(*position, self.env.agents[agent_index_at_position].direction)
            same_agent_found = \
                self.check_intersecting_possible_transitions(possible_transitions,
                                                             direction,
                                                             opp_possible_transitions,
                                                             self.env.agents[agent_index_at_position].direction)

            opp_agent_found = 1.0 - same_agent_found
            opp_agent_index_comp = handle < agent_index_at_position

        return other_agent_found, same_agent_found, opp_agent_found, opp_agent_index_comp

    def getOtherAgentProbInfo(self, handle, agent_position, agent_direction):
        # calculate the 'opposite directed traveling' and 'same directed traveling' other agents occurence probability
        # the probability is the inverted 'intersection' of 'not having' any agent in the cell at next step
        possible_transitions = self.env.rail.get_transitions(*agent_position, agent_direction)
        opp_agent_next_steps = self.agent_positions_next_step.get(agent_position, {})
        aggregated_prob_opp = 1.0
        aggregated_prob_same = 1.0
        for opp_agent_idx in opp_agent_next_steps.keys():
            if opp_agent_idx != handle:
                opp_agent_info = opp_agent_next_steps.get(opp_agent_idx)
                opp_possible_transitions = \
                    self.env.rail.get_transitions(*agent_position, self.env.agents[opp_agent_idx].direction)
                if not self.check_intersecting_possible_transitions(possible_transitions,
                                                                    agent_direction,
                                                                    opp_possible_transitions,
                                                                    self.env.agents[opp_agent_idx].direction):
                    aggregated_prob_opp *= 1.0 - opp_agent_info['probability']
                else:
                    aggregated_prob_same *= 1.0 - opp_agent_info['probability']
        opp_agent_occurrence_probability = 1.0 - aggregated_prob_opp
        same_agent_occurrence_probability = 1.0 - aggregated_prob_same
        return same_agent_occurrence_probability, opp_agent_occurrence_probability

    def calculate_rail_agent_status(self):
        self.nb_agents_done = 0
        self.nb_agent_active = 0
        self.nb_agent_ready = 0
        self.smallest_index_active = self.env.get_num_agents()
        self.smallest_index_ready = self.env.get_num_agents()

        for i_agent, agent in enumerate(self.env.agents):
            # manage the boolean flag to check if all agents are indeed done (or done_removed)
            if (agent.state in [TrainState.DONE]):
                self.nb_agents_done += 1
            if (agent.state in [TrainState.MALFUNCTION, TrainState.MOVING, TrainState.STOPPED]):
                self.nb_agent_active += 1
                self.smallest_index_active = min(self.smallest_index_active, i_agent)
            if (agent.state == TrainState.READY_TO_DEPART):
                self.nb_agent_ready += 1
                self.smallest_index_ready = min(self.smallest_index_ready, i_agent)

    def project_possible_paths_as_probabilistic(self,
                                                agent_handle,
                                                agent_status_filter,
                                                break_max_step: int):

        prob_density_map_shortest: np.ndarray = np.zeros((4, self.env.height, self.env.width))
        prob_density_map_alternative: np.ndarray = np.zeros((4, self.env.height, self.env.width))

        agent = self.env.agents[agent_handle]
        if agent.state == agent_status_filter:
            agent_virtual_position, agent_virtual_direction, agent_status, agent_target = \
                self.agent_can_choose_helper.get_agent_position_and_direction(agent_handle)

            ppw = ProjectPathWalker(self.env, project_weighted_distance=True)
            ppw.walk_to_target(agent_handle,
                               position=agent_virtual_position,
                               direction=agent_virtual_direction,
                               break_max_step=break_max_step)
            pdmap = ppw.get_projected_density_map()
            prob_density_map_shortest = probabilistic_value_mixing(prob_density_map_shortest, pdmap)
            branch_entry_points = ppw.get_branch_entry_points()
            if len(branch_entry_points) > 0:
                for unroll_close_to_agent_branch in range(min(len(branch_entry_points), 4)):
                    branch_entry_point = branch_entry_points[unroll_close_to_agent_branch]
                    ppw = ProjectPathWalker(self.env, project_weighted_distance=True)
                    ppw.walk_to_target(agent_handle,
                                       position=branch_entry_point[0],
                                       direction=branch_entry_point[1],
                                       break_max_step=break_max_step)
                    pdmap = ppw.get_projected_density_map() * \
                            (branch_entry_point[2] / branch_entry_point[3])
                    prob_density_map_alternative = probabilistic_value_mixing(
                        prob_density_map_alternative, pdmap)

        return prob_density_map_shortest, prob_density_map_alternative

    def get_density_of_projected_path_to_target(self):
        break_max_step = 20

        for agent_handle in self.env.get_agent_handles():
            agent = self.env.agents[agent_handle]

            agent_virtual_position, agent_virtual_direction, agent_status, agent_target = \
                self.agent_can_choose_helper.get_agent_position_and_direction(agent_handle)
            if agent_status < TrainState.DONE:
                cache: AgentMapCache = self.agents_path_maps_cache.get(agent_handle, None)
                if cache is not None:
                    possible_transitions = self.env.rail.get_transitions(*cache.position, cache.direction)
                else:
                    possible_transitions = (0, 0, 0, 0)

                if fast_count_nonzero(possible_transitions) == 1 and \
                        cache is not None and \
                        cache.status == agent_status:
                    # use case
                    prob_shortest_path_map = cache.shortest_path_map
                    prob_branching_path_map = cache.branching_path_map
                    for dir_loop in range(4):
                        prob_shortest_path_map[dir_loop, agent_virtual_position[0], agent_virtual_position[0]] = 0
                        prob_branching_path_map[dir_loop, agent_virtual_position[0], agent_virtual_position[0]] = 0
                else:
                    prob_shortest_path_map, prob_branching_path_map = \
                        self.project_possible_paths_as_probabilistic(agent_handle, agent_status, break_max_step)

                cache = AgentMapCache(
                    direction=agent_virtual_direction,
                    position=agent_virtual_position,
                    status=agent_status,
                    shortest_path_map=prob_shortest_path_map,
                    branching_path_map=prob_branching_path_map)
                self.agents_path_maps_cache.update({agent_handle: cache})

        prob_shortest_path_map_active: np.ndarray = np.zeros((4, self.env.height, self.env.width))
        prob_branching_path_map_active: np.ndarray = np.zeros((4, self.env.height, self.env.width))
        prob_shortest_path_map_ready: np.ndarray = np.zeros((4, self.env.height, self.env.width))
        prob_branching_path_map_ready: np.ndarray = np.zeros((4, self.env.height, self.env.width))

        for agent_handle in self.env.get_agent_handles():
            cache: AgentMapCache = self.agents_path_maps_cache.get(agent_handle, None)
            agent_virtual_position, agent_virtual_direction, agent_status, agent_target = \
                self.agent_can_choose_helper.get_agent_position_and_direction(agent_handle)
            if agent_status in [TrainState.STOPPED, TrainState.MOVING, TrainState.MALFUNCTION]:
                prob_shortest_path_map_active = \
                    probabilistic_value_mixing(prob_shortest_path_map_active,
                                               cache.shortest_path_map)
                prob_branching_path_map_active = \
                    probabilistic_value_mixing(prob_branching_path_map_active,
                                               cache.branching_path_map)
            elif agent_status == TrainState.READY_TO_DEPART:
                prob_shortest_path_map_ready = \
                    probabilistic_value_mixing(prob_shortest_path_map_ready,
                                               cache.shortest_path_map)
                prob_branching_path_map_ready = \
                    probabilistic_value_mixing(prob_branching_path_map_ready,
                                               cache.branching_path_map)

        return prob_shortest_path_map_active, prob_branching_path_map_active, \
               prob_shortest_path_map_ready, prob_branching_path_map_ready

    def get_many(self, handles: Optional[List[int]] = None):
        self.agent_positions_next_step = get_agent_positions_next_step(self.env)
        self.agents_target = get_agent_targets(self.env)
        self.agent_can_choose_helper.reset_switch_cluster_occupied()

        self.prob_density_map_shortest_active, \
        self.prob_density_map_alternative_active, \
        self.prob_density_map_shortest_ready, \
        self.prob_density_map_alternative_ready = \
            self.get_density_of_projected_path_to_target()

        if False:
            plt.subplot(1, 1, 1)
            x = self.densitiy_of_prob_path_to_target_active[0]
            for i in range(3):
                x += self.densitiy_of_prob_path_to_target_active[i + 1]
            plt.imshow(x)

            plt.show(block=False)
            plt.pause(0.00001)

        self.calculate_rail_agent_status()
        self.shortest_path_walking_agent.start_step(False)
        if self.enabled_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.start_step(False)
        observations = super().get_many(handles)
        if self.enabled_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.end_step(False)
        self.shortest_path_walking_agent.end_step(False)
        return observations

    def get(self, handle: int = 0):
        observation = np.zeros(self.observation_dim) - 1
        visited = []

        agent = self.env.agents[handle]

        agent_done = False
        agent_virtual_position, agent_virtual_direction, agent_status, agent_target = \
            self.agent_can_choose_helper.get_agent_position_and_direction(handle)
        if agent_status == TrainState.READY_TO_DEPART:
            observation[0] = 1
        elif agent_status in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
            observation[1] = 1
        else:
            observation[2] = 1
            agent_done = True
        visited.append(agent_virtual_position)

        if not agent_done:
            distance_map = self.env.distance_map.get()
            current_cell_dist = distance_map[handle,
                                             agent_virtual_position[0], agent_virtual_position[1],
                                             agent_virtual_direction]

            possible_transitions = np.asarray(self.env.rail.get_transitions(*agent_virtual_position,
                                                                            agent_virtual_direction))
            orientation = agent_virtual_direction
            if fast_count_nonzero(possible_transitions) == 1:
                orientation = fast_argmax(possible_transitions)

            can_decide = True
            observation[3] = can_decide
            if can_decide:
                if self.enabled_dead_lock_avoidance_agent:
                    agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all = \
                        self.agent_can_choose_helper.check_agent_decision(agent_virtual_position,
                                                                          agent_virtual_direction)
                    other_agent_found, same_agent_found, opp_agent_found, opp_agent_index_comp = \
                        self.getOtherAgentInfo(handle, agent_virtual_position, agent_virtual_direction)
                    same_agent_occurrence_probability, opp_agent_occurrence_probability = \
                        self.getOtherAgentProbInfo(handle, agent_virtual_position, agent_virtual_direction)

                    dl_stop = int(self.dead_lock_avoidance_agent.act(handle, None, 0) == TrainState.STOPPED)
                    dl_val = self.dead_lock_avoidance_agent.get_agent_can_move_value(handle)
                    if np.isinf(dl_val):
                        dl_val = -1
                    observation[4] = dl_val
                    observation[5] = opp_agent_found
                    observation[6] = same_agent_occurrence_probability
                    observation[7] = dl_stop
                else:
                    agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all = \
                        self.agent_can_choose_helper.check_agent_decision(agent_virtual_position,
                                                                          agent_virtual_direction)
                    observation[4] = int(agents_on_switch)
                    observation[5] = int(agents_near_to_switch)
                    observation[6] = int(agents_near_to_switch_all)
                    observation[7] = int(agents_on_switch_all)

                max_next_cell_dist = 1
                for dir_loop, branch_direction in enumerate(
                        [(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):

                    if possible_transitions[branch_direction]:
                        new_position = get_new_position(agent_virtual_position, branch_direction)
                        if new_position is not None:
                            new_cell_dist = distance_map[handle,
                                                         new_position[0], new_position[1],
                                                         branch_direction]
                            if not np.isinf(new_cell_dist):
                                max_next_cell_dist = max(max_next_cell_dist, new_cell_dist)

                            if self.agent_can_choose_helper.get_agent_positions()[new_position] != -1:
                                possible_transitions[branch_direction] = 0
                                if fast_count_nonzero(possible_transitions) == 1:
                                    observation[3] = 1

                for dir_loop, branch_direction in enumerate(
                        [(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):

                    if possible_transitions[branch_direction]:
                        new_position = get_new_position(agent_virtual_position, branch_direction)
                        new_cell_dist = distance_map[handle,
                                                     new_position[0], new_position[1],
                                                     branch_direction]

                        if new_position is not None:
                            if not np.isinf(new_cell_dist):
                                observation[8 + branch_direction] = new_cell_dist / max_next_cell_dist
                                observation[12 + branch_direction] = int(current_cell_dist > new_cell_dist)

                            new_possible_transitions = self.env.rail.get_transitions(*new_position,
                                                                                     branch_direction)

                            observation[16 + dir_loop] = 0
                            observation[20 + dir_loop] = 0
                            observation[24 + dir_loop] = 0
                            observation[28 + dir_loop] = 0
                            observation[32 + dir_loop] = 0
                            observation[36 + dir_loop] = 0
                            observation[40 + dir_loop] = 0
                            observation[44 + dir_loop] = 0

                            agents_on_switch, agents_near_to_switch, \
                            agents_near_to_switch_all, agents_on_switch_all = \
                                self.agent_can_choose_helper.check_agent_decision(new_position, branch_direction)
                            observation[48 + dir_loop] = int(agents_on_switch)
                            observation[52 + dir_loop] = int(agents_near_to_switch)

                            for projected_dir in range(4):

                                map_shortest_active = self.prob_density_map_shortest_active[projected_dir,
                                                                                            new_position[0],
                                                                                            new_position[1]]

                                map_alternative_active = self.prob_density_map_alternative_active[projected_dir,
                                                                                                  new_position[0],
                                                                                                  new_position[1]]

                                map_shortest_ready = self.prob_density_map_shortest_ready[projected_dir,
                                                                                          new_position[0],
                                                                                          new_position[1]]

                                map_alternative_ready = self.prob_density_map_alternative_ready[projected_dir,
                                                                                                new_position[0],
                                                                                                new_position[1]]

                                if new_possible_transitions[projected_dir] == 0:
                                    # Opposite agent -> not reachable direction
                                    observation[16 + dir_loop] = \
                                        max(observation[16 + dir_loop], map_shortest_active)
                                    observation[20 + dir_loop] = \
                                        max(observation[20 + dir_loop], map_shortest_ready)
                                    observation[24 + dir_loop] = \
                                        max(observation[24 + dir_loop], map_alternative_ready)
                                    observation[28 + dir_loop] = \
                                        max(observation[28 + dir_loop], map_alternative_active)
                                else:
                                    # Same agent
                                    observation[32 + dir_loop] = \
                                        max(observation[32 + dir_loop], map_shortest_active)
                                    observation[36 + dir_loop] = \
                                        max(observation[36 + dir_loop], map_shortest_ready)
                                    observation[40 + dir_loop] = \
                                        max(observation[40 + dir_loop], map_alternative_ready)
                                    observation[44 + dir_loop] = \
                                        max(observation[44 + dir_loop], map_alternative_active)
        '''
        if handle == 0:
            self.env.dev_obs_dict.update({handle: visited})
            self.env.dev_obs_dict.update({handle + 1: visited_1})
            self.env.dev_obs_dict.update({handle + 2: visited_2})
        '''
        self.env.dev_obs_dict.update({handle: visited})

        observation[np.isinf(observation)] = -1
        observation[np.isnan(observation)] = -1

        return observation
