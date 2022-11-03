from flatland.envs.rail_env import RailEnvActions

# global action size
global _agent_action_config_action_size
_agent_action_config_action_size = 5


def get_flatland_full_action_size():
    '''
    Just get the default RailEnvActions space size back
    :return: The action space of flatland is 5 discrete actions
    '''
    return 5


def set_action_size_full():
    '''
    Set a global variable to define the action space size to full
    If set -> the build-in RailEnvActions will be used (default)
    '''
    global _agent_action_config_action_size
    # The agents (DDDQN, PPO, ... ) have this actions space
    _agent_action_config_action_size = 5


def set_action_size_reduced():
    '''
    Set a global variable to define the action space size to reduced
    If set -> the actions space has the DO_NOTHING action removed from RailEnvActions
    :return:
    '''
    global _agent_action_config_action_size
    # The agents (DDDQN, PPO, ... ) have this actions space
    _agent_action_config_action_size = 4


def get_action_size():
    '''
    :return: The current active action space size
    '''
    global _agent_action_config_action_size
    # The agents (DDDQN, PPO, ... ) have this actions space
    return _agent_action_config_action_size


def map_actions(actions):
    '''
    Converts the actions in the actions array. Iff the actions space is not set to full.
    :param actions: Array of actions to convert
    :return: converted actions as array
    '''
    # Map the
    if get_action_size() != get_flatland_full_action_size():
        ret_actions = {}
        for key in actions:
            value = actions.get(key, 0)
            ret_actions.update({key: map_action(value)})
        return ret_actions
    return actions


def map_action(action):
    '''
    if the action space is full -> no tranformation will be done (just using the RailEnvActions). Otherwise the action
    will be transformed/encoded -> Action [0,1,2,3] -> RailEnvActions -> DO_NOTHING does no longer exist.
    :param action: action to convert/transform : number in [0,1,2,3]
    :return: transformed action
    '''
    if get_action_size() == get_flatland_full_action_size():
        return action

    if action == 0:
        return RailEnvActions.MOVE_LEFT
    if action == 1:
        return RailEnvActions.MOVE_FORWARD
    if action == 2:
        return RailEnvActions.MOVE_RIGHT
    if action == 3:
        return RailEnvActions.STOP_MOVING
    return RailEnvActions.STOP_MOVING


def convert_default_rail_env_action(action):
    '''
    Converts a RailEnvActions (0,1,2,3,4)
    :param action: RailEnvAction or a number in [0,1,2,3,4]
    :return: Iff action space is set to full, no transformation will be done otherwise replace action DO_NOTHING -> STOP
    '''
    if get_action_size() == get_flatland_full_action_size():
        return action

    if action == RailEnvActions.MOVE_LEFT:
        return 0
    elif action == RailEnvActions.MOVE_FORWARD:
        return 1
    elif action == RailEnvActions.MOVE_RIGHT:
        return 2
    elif action == RailEnvActions.STOP_MOVING:
        return 3
    # action == RailEnvActions.DO_NOTHING:
    return 3
