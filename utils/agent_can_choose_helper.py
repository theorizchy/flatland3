import matplotlib.pyplot as plt
import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import TrainState
from utils.fast_methods import fast_count_nonzero, fast_argmax
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


class AgentCanChooseHelper:
    def __init__(self):
        self.render_debug_information = False

    def reset(self, env):
        self.env = env
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        self.switch_cluster = {}
        self.switch_cluster_occupied = {}
        self.switch_cluster_lock = {}
        self.switch_cluster_grid = None
        self.agent_positions = None

        self.reset_swicht_cluster_lock()
        self.reset_switch_cluster_occupied()
        if self.env is not None:
            self.find_all_cell_where_agent_can_choose()

        self.calculate_agent_positions()

    def get_agent_positions(self):
        return self.agent_positions

    def calculate_agent_positions(self):
        self.agent_positions: np.ndarray = np.full((self.env.height, self.env.width), -1)
        for agent_handle in self.env.get_agent_handles():
            agent = self.env.agents[agent_handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                position = agent.position
                if position is None:
                    position = agent.initial_position
                self.agent_positions[position] = agent_handle

    def clear_switch_cluster_lock(self):
        '''
        clean up switch cluster lock
        '''
        self.switch_cluster_lock = {}

    def clear_switch_cluster_occupied(self):
        '''
        clean up switch cluster occupied
        '''
        self.switch_cluster_occupied = {}

    def lock_switch_cluster(self, handle, agent_pos, agent_dir):
        '''
        Lock the switch cluster if possible
        :param handle: Agent handle
        :param agent_pos: position to lock
        :param agent_dir: direction
        :return: True if lock is successfully done otherwise false (it might still have a lock)
        '''
        cluster_id, grid_cell_members = self.get_switch_cluster(agent_pos)
        if cluster_id < 1:
            return True
        lock_handle = self.switch_cluster_lock.get(cluster_id, None)
        if lock_handle is None:
            self.switch_cluster_lock.update({cluster_id: handle})
            return True
        if lock_handle == handle:
            return True
        return False

    def unlock_switch_cluster(self, handle, agent_pos, agent_dir):
        '''
        Lock the switch cluster if possible
        :param handle: Agent handle
        :param agent_pos: position to lock
        :param agent_dir: direction
        :return: True if unlock is successfully done otherwise false (it might still have a lock own by another agent)
        '''
        cluster_id, grid_cell_members = self.get_switch_cluster(agent_pos)
        if cluster_id < 1:
            return True
        lock_handle = self.switch_cluster_lock.get(cluster_id, None)
        if lock_handle == handle:
            self.switch_cluster_lock.update({cluster_id, None})
            return True
        return False

    def get_agent_position_and_direction(self, handle):
        '''
        Returns the agent position - if not yet started (active) it returns the initial position
        :param handle: agent reference (handle)
        :return: agent_pos, agent_dir, agent.state
        '''
        agent = self.env.agents[handle]
        agent_pos = agent.position
        agent_dir = agent.direction
        if agent_pos is None:
            agent_pos = agent.initial_position
            agent_dir = agent.initial_direction
        return agent_pos, agent_dir, agent.state, agent.target

    def has_agent_switch_cluster_lock(self, handle, agent_pos=None, agent_dir=None):
        '''
        Checks if the agent passed by the handle has the switch cluster lock
        :param handle: agent reference (handle)
        :param agent_pos: position to check
        :param agent_dir: direction property
        :return: True if handle owns the lock otherwise false
        '''
        if agent_pos is None or agent_dir is None:
            agent_pos, agent_dir, agent_status, agent_target = self.get_agent_position_and_direction(handle)
        cluster_id, grid_cell_members = self.get_switch_cluster(agent_pos)
        if cluster_id < 1:
            return False
        lock_handle = self.switch_cluster_lock.get(cluster_id, None)
        return lock_handle == handle

    def get_switch_cluster_occupiers_next_cell(self, handle, agent_pos, agent_dir):
        '''
        Returns all occupiers for the next cell
        :param handle: agent reference (handle)
        :param agent_pos: position to check
        :param agent_dir: direction property
        :return: a list of all agents (handles) which occupied the next cell switch cluster
        '''
        possible_transitions = self.env.rail.get_transitions(*agent_pos, agent_dir)
        occupiers = []
        for new_direction in range(4):
            if possible_transitions[new_direction] == 1:
                new_position = get_new_position(agent_pos, new_direction)
                occupiers += self.get_switch_cluster_occupiers(handle,
                                                               new_position,
                                                               new_direction)
        return occupiers

    def mark_switch_next_cluster_occupied(self, handle):
        agent_position, agent_direciton, agent_status, agent_target = \
            self.get_agent_position_and_direction(handle)

        possible_transitions = self.env.rail.get_transitions(*agent_position, agent_direciton)
        for new_direction in range(4):
            if possible_transitions[new_direction] == 1:
                new_position = get_new_position(agent_position, new_direction)
                self.mark_switch_cluster_occupied(handle, new_position, new_direction)

    def can_agent_enter_next_cluster(self, handle):
        agent_position, agent_direciton, agent_status, agent_target = \
            self.get_agent_position_and_direction(handle)
        occupiers = self.get_switch_cluster_occupiers_next_cell(handle,
                                                                agent_position,
                                                                agent_direciton)
        if len(occupiers) > 0 and handle not in occupiers:
            return False
        return True

    def get_switch_cluster_occupiers(self, handle, agent_pos, agent_dir):
        '''
        :param handle: agent reference (handle)
        :param agent_pos: position to check
        :param agent_dir: direction property
        :return: a list of all agents (handles) which occupied the switch cluster
        '''
        cluster_id, grid_cell_members = self.get_switch_cluster(agent_pos)
        if cluster_id < 1:
            return []
        return self.switch_cluster_occupied.get(cluster_id, [])

    def mark_switch_cluster_occupied(self, handle, agent_pos, agent_dir):
        '''
        Add the agent handle to the switch cluster occupied data. Set the agent (handle) as occupier
        :param handle: agent reference (handle)
        :param agent_pos: position to check
        :param agent_dir: direction property
        :return:
        '''
        cluster_id, grid_cell_members = self.get_switch_cluster(agent_pos)
        if cluster_id < 1:
            return
        agent_handles = self.switch_cluster_occupied.get(cluster_id, [])
        agent_handles.append(handle)
        self.switch_cluster_occupied.update({cluster_id: agent_handles})

    def reset_swicht_cluster_lock(self):
        '''
        Reset the explicit lock data  switch_cluster_lock
        '''
        self.clear_switch_cluster_lock()

    def reset_switch_cluster_occupied(self, handle_only_active_agents=False):
        '''
        Reset the occupied flag by recomputing the switch_cluster_occupied map
        :param handle_only_active_agents: if true only agent with status ACTIVE will be mapped
        '''
        self.clear_switch_cluster_occupied()
        for handle in range(self.env.get_num_agents()):
            agent_pos, agent_dir, agent_status, agent_target = self.get_agent_position_and_direction(handle)
            if handle_only_active_agents:
                if agent_status in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                    self.mark_switch_cluster_occupied(handle, agent_pos, agent_dir)
            else:
                if agent_status < TrainState.DONE:
                    self.mark_switch_cluster_occupied(handle, agent_pos, agent_dir)

    def get_switch_cluster(self, pos):
        '''
        Returns the switch cluster at position pos
        :param pos: the position for which the switch cluster must be returned
        :return: if the position is not None and the switch cluster are computed it returns the cluster_id and the
        grid cell members otherwise -1 and an empty list
        '''
        if pos is None:
            return -1, []
        if self.switch_cluster_grid is None:
            return -1, []
        cluster_id = self.switch_cluster_grid[pos]
        grid_cell_members = self.switch_cluster.get(cluster_id, [])
        return cluster_id, grid_cell_members

    def find_all_switches(self):
        '''
        Search the environment (rail grid) for all switch cells. A switch is a cell where more than one tranisation
        exists and collect all direction where the switch is a switch.
        '''
        self.switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    num_transitions = fast_count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        directions = self.switches.get(pos, [])
                        directions.append(dir)
                        self.switches.update({pos: directions})

    def find_all_switch_neighbours(self):
        '''
        Collect all cells where is a neighbour to a switch cell. All cells are neighbour where the agent can make
        just one step and he stands on a switch. A switch is a cell where the agents has more than one transition.
        '''
        self.switches_neighbours = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                # look one step forward
                for dir in range(4):
                    pos = (h, w)
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    for d in range(4):
                        if possible_transitions[d] == 1:
                            new_cell = get_new_position(pos, d)
                            if new_cell in self.switches.keys():
                                directions = self.switches_neighbours.get(pos, [])
                                directions.append(dir)
                                self.switches_neighbours.update({pos: directions})

    def find_cluster_label(self, in_label) -> int:
        label = int(in_label)
        while 0 != self.label_dict[label]:
            label = self.label_dict[label]
        return label

    def union_cluster_label(self, root, slave) -> None:
        root_label = self.find_cluster_label(root)
        slave_label = self.find_cluster_label(slave)
        if slave_label != root_label:
            self.label_dict[slave_label] = root_label

    def find_connected_clusters_and_label(self, binary_image):
        padded_binary_image = np.pad(binary_image, ((1, 0), (1, 0)), 'constant', constant_values=(0, 0))
        w = np.size(binary_image, 0)
        h = np.size(binary_image, 1)
        self.label_dict = [int(i) for i in np.zeros(w * h)]
        label = 1
        #  first pass
        for cow in range(1, h + 1):
            for col in range(1, w + 1):
                working_position = (cow, col)
                working_pixel = padded_binary_image[working_position]
                if working_pixel != 0:
                    left_pixel_pos = (cow, col - 1)
                    up_pixel_pos = (cow - 1, col)

                    left_pixel = padded_binary_image[left_pixel_pos]
                    up_pixel = padded_binary_image[up_pixel_pos]

                    # Use connections (rails) for clustering (only real connected pixels builds a real cluster)
                    if (cow < self.env.height) and (col < self.env.width):
                        left_ok = 0
                        up_ok = 0
                        # correct padded image position (railenv)
                        t_working_position = (working_position[0] - 1, working_position[1] - 1)
                        t_left_pixel_pos = (left_pixel_pos[0] - 1, left_pixel_pos[1] - 1)
                        t_up_pixel_pos = (up_pixel_pos[0] - 1, up_pixel_pos[1] - 1)
                        for direction_loop in range(4):
                            possible_transitions = self.env.rail.get_transitions(*t_working_position, direction_loop)
                            orientation = direction_loop
                            if fast_count_nonzero(possible_transitions) == 1:
                                orientation = fast_argmax(possible_transitions)
                            for dir_loop, new_direction in enumerate(
                                    [(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                                if possible_transitions[new_direction] == 1:
                                    new_pos = get_new_position(t_working_position, new_direction)
                                    if new_pos == t_left_pixel_pos:
                                        left_ok = 1
                                    if new_pos == t_up_pixel_pos:
                                        up_ok = 1
                        left_pixel *= left_ok
                        up_pixel *= up_ok

                    # build clusters
                    if left_pixel == 0 and up_pixel == 0:
                        padded_binary_image[working_position] = label
                        label += 1

                    if left_pixel != 0 and up_pixel != 0:
                        smaller = left_pixel if left_pixel < up_pixel else up_pixel
                        bigger = left_pixel if left_pixel > up_pixel else up_pixel
                        padded_binary_image[working_position] = smaller
                        self.union_cluster_label(smaller, bigger)

                    if up_pixel != 0 and left_pixel == 0:
                        padded_binary_image[working_position] = up_pixel

                    if up_pixel == 0 and left_pixel != 0:
                        padded_binary_image[working_position] = left_pixel

        for cow in range(1, h + 1):
            for col in range(1, w + 1):
                root = self.find_cluster_label(padded_binary_image[cow][col])
                padded_binary_image[cow][col] = root

        self.switch_cluster_grid = padded_binary_image[1:, 1:]
        for h in range(self.env.height):
            for w in range(self.env.width):
                working_position = (h, w)
                root = self.switch_cluster_grid[working_position]
                if root > 0:
                    pos_data = self.switch_cluster.get(root, [])
                    pos_data.append(working_position)
                    self.switch_cluster.update({root: pos_data})

    def cluster_all_switches(self):
        info_image = np.zeros((self.env.height, self.env.width))
        # for h in range(self.env.height):
        #     for w in range(self.env.width):
        #        # look one step forward
        #         if self.env.rail.grid[h][w] > 0:
        #             info_image[(h,w)] = -1

        for key in self.switches.keys():
            info_image[key] = 1

        # build clusters
        self.find_connected_clusters_and_label(info_image)

        if self.render_debug_information:
            # Setup renderer
            env_renderer = RenderTool(self.env, gl="PGL",
                                      agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX)
            env_renderer.set_new_rail()
            env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=True,
                show_predictions=False
            )

            plt.subplot(1, 2, 1)
            plt.imshow(info_image)
            plt.subplot(1, 2, 2)
            plt.imshow(self.switch_cluster_grid)
            plt.show()
            plt.pause(0.01)

    def find_all_cell_where_agent_can_choose(self):
        '''
        prepare the memory - collect all cells where the agent can choose more than FORWARD/STOP.
        '''
        self.find_all_switches()
        self.find_all_switch_neighbours()
        self.cluster_all_switches()

    def check_agent_decision(self, position, direction):
        '''
         Decide whether the agent is
         - on a switch
         - at a switch neighbour (near to switch). The switch must be a switch where the agent has more option than
           FORWARD/STOP
         - all switch : doesn't matter whether the agent has more options than FORWARD/STOP
         - all switch neightbors : doesn't matter the agent has more then one options (transistion) when he reach the
           switch
        :param position: (x,y) cell coordinate
        :param direction: Flatland direction
        :return: agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all
        '''
        agents_on_switch = False
        agents_on_switch_all = False
        agents_near_to_switch = False
        agents_near_to_switch_all = False
        if position in self.switches.keys():
            agents_on_switch = direction in self.switches[position]
            agents_on_switch_all = True

        if position in self.switches_neighbours.keys():
            new_cell = get_new_position(position, direction)
            if new_cell in self.switches.keys():
                if not direction in self.switches[new_cell]:
                    agents_near_to_switch = direction in self.switches_neighbours[position]
            else:
                agents_near_to_switch = direction in self.switches_neighbours[position]

            agents_near_to_switch_all = direction in self.switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all

    def requires_agent_decision(self):
        '''
        Returns for all agents its check_agent_decision values
        :return: dicts with check_agent_decision values stored (each agents)
        '''
        agents_can_choose = {}
        agents_on_switch = {}
        agents_on_switch_all = {}
        agents_near_to_switch = {}
        agents_near_to_switch_all = {}
        for a in range(self.env.get_num_agents()):
            ret_agents_on_switch, ret_agents_near_to_switch, ret_agents_near_to_switch_all, ret_agents_on_switch_all = \
                self.check_agent_decision(
                    self.env.agents[a].position,
                    self.env.agents[a].direction)
            agents_on_switch.update({a: ret_agents_on_switch})
            agents_on_switch_all.update({a: ret_agents_on_switch_all})
            ready_to_depart = self.env.agents[a].state == TrainState.READY_TO_DEPART
            agents_near_to_switch.update({a: (ret_agents_near_to_switch and not ready_to_depart)})

            agents_can_choose.update({a: agents_on_switch[a] or agents_near_to_switch[a]})

            agents_near_to_switch_all.update({a: (ret_agents_near_to_switch_all and not ready_to_depart)})

        return agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all
