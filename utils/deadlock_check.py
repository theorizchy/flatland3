import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnv

from utils.fast_methods import fast_count_nonzero, fast_argmax, fast_isclose


def get_agent_positions_next_step(env):
    agent_positions_next_step = {}
    for agent_handle in env.get_agent_handles():
        agent = env.agents[agent_handle]
        if agent.state < TrainState.DONE:
            position = agent.position
            direction = agent.direction
            if position is None:
                position = agent.initial_position
                direction = agent.initial_direction

            possible_transitions = env.rail.get_transitions(*position, direction)
            num_transitions = fast_count_nonzero(possible_transitions)
            orientation = agent.direction
            if num_transitions == 1:
                orientation = fast_argmax(possible_transitions)
            for dir_loop, new_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                if possible_transitions[new_direction]:
                    new_position = get_new_position(position, new_direction)
                    d = agent_positions_next_step.get(new_position, {})
                    d.update({agent_handle: {'direction': new_direction, 'probability': 1 / num_transitions}})
                    agent_positions_next_step.update({new_position: d})
    return agent_positions_next_step


def get_agent_targets(env):
    agent_targets = []
    for agent_handle in env.get_agent_handles():
        agent = env.agents[agent_handle]
        if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
            agent_targets.append(agent.target)
    return agent_targets


def check_if_all_blocked(env):
    """
    Checks whether all the agents are blocked (full deadlock situation).
    In that case it is pointless to keep running inference as no agent will be able to move.
    :param env: current environment
    :return:
    """

    # First build a map of agents in each position
    location_has_agent = {}
    for agent in env.agents:
        if agent.state in [TrainState.STOPPED, TrainState.MALFUNCTION, TrainState.MOVING, TrainState.DONE] and \
                agent.position:
            location_has_agent[tuple(agent.position)] = 1

    # Looks for any agent that can still move
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.state == TrainState.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
            agent_virtual_position = agent.position
        elif agent.state == TrainState.DONE:
            agent_virtual_position = agent.target
        else:
            continue

        possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
        orientation = agent.direction

        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_position = get_new_position(agent_virtual_position, branch_direction)

                if new_position not in location_has_agent:
                    return False

    # No agent can move at all: full deadlock!
    return True


def is_agent_chain_deadlocked(handle, next_cell_agent, dead_locked_agents, visited=[]):
    agents = next_cell_agent.get(handle, [])
    if handle in visited:
        for opp_handle in agents:
            if dead_locked_agents.get(opp_handle, False):
                return True
        return False

    for opp_handle in agents:
        visited.append(opp_handle)
        if dead_locked_agents.get(opp_handle, False):
            return True
        if is_agent_chain_deadlocked(opp_handle, next_cell_agent, dead_locked_agents, visited):
            return True
    return False


def find_and_punish_deadlock(env: RailEnv, all_rewards, penalty=-1000.0):
    agent_next_positions = {}
    positions_agents = {}
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
            position = agent.position
            direction = agent.direction
            x = positions_agents.get(position, [])
            x.append(handle)
            positions_agents.update({position: x})
            possible_transitions = env.rail.get_transitions(*position, direction)
            if fast_isclose(agent.speed_data['position_fraction'], 0.0, rtol=1e-03):
                for new_direction in range(4):
                    if possible_transitions[new_direction] == 1:
                        new_position = get_new_position(position, new_direction)
                        x = agent_next_positions.get(handle, [])
                        x.append(new_position)
                        agent_next_positions.update({handle: x})
            else:
                action = agent.speed_data['transition_action_on_cellexit']
                next_direction, transition_valid = env.check_action(agent, action)
                if possible_transitions[next_direction] == 1:
                    next_position = get_new_position(agent.position, next_direction)
                    x = agent_next_positions.get(handle, [])
                    x.append(next_position)
                    agent_next_positions.update({handle: x})

    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
            position = agent.position
            next_pos = agent_next_positions.get(handle, [])
            for n_pos in next_pos:
                opp_agents = positions_agents.get(n_pos, [])
                for opp_handle in opp_agents:
                    opp_positions = agent_next_positions.get(opp_handle, [])
                    if position in opp_positions:
                        next_pos_2 = agent_next_positions.get(handle, [])
                        next_pos_2.remove(n_pos)
                        agent_next_positions.update({handle: next_pos_2})
                        opp_positions_2 = agent_next_positions.get(opp_handle, [])
                        opp_positions_2.remove(position)
                        agent_next_positions.update({opp_handle: opp_positions_2})

    deadlocked_agents = dict.fromkeys(list(range(env.get_num_agents())) + ["__all__"], False)
    old_nbr_dl = -1
    nbr_dl = 0
    while old_nbr_dl != nbr_dl:
        for handle in env.get_agent_handles():
            agent = env.agents[handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                next_pos = agent_next_positions.get(handle, [])
                if len(next_pos) == 0:
                    all_rewards[handle] += penalty
                    deadlocked_agents[handle] = 1
                else:
                    s = 0
                    for p in next_pos:
                        h = positions_agents.get(p, [])
                        if len(h) > 0:
                            s += deadlocked_agents[h[0]]
                    if s == len(next_pos):
                        all_rewards[handle] += penalty
                        deadlocked_agents[handle] = True
        old_nbr_dl = nbr_dl
        nbr_dl = np.sum(deadlocked_agents)

    deadlocked_agents['__all__'] = True
    deadlocked_agents['__has__'] = False
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.state < TrainState.DONE:
            if deadlocked_agents[handle] == 0:
                deadlocked_agents['__all__'] = False
            else:
                deadlocked_agents['__has__'] = True

    return deadlocked_agents, all_rewards
