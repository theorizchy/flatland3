from collections import deque

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.policy import DummyMemory, HybridPolicy
from reinforcement_learning.ppo_agent import PPOPolicy
from utils.agent_action_config import convert_default_rail_env_action, map_action
from utils.agent_can_choose_helper import AgentCanChooseHelper
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent


class MultiDecisionAgent(HybridPolicy):

    def __init__(self, env, state_size, action_size, in_parameters=None, evaluation_mode=False, use_ppo_policy=True):
        print(">> MultiDecisionAgent")
        super(MultiDecisionAgent, self).__init__()

        self.do_debug_print = True
        self.enable_cluster_lock = True
        self.enable_auto_move_forward = True
        self.memory = DummyMemory()
        self.loss = 0

        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.in_parameters = in_parameters
        self.evaluation_mode = evaluation_mode

        self.agent_can_choose_helper = AgentCanChooseHelper()
        self.deadlock_avoidance_agent = DeadLockAvoidanceAgent(env, action_size, enable_eps=False)
        self.nbr_extra_actions = 1
        if use_ppo_policy:
            self.learning_policy = PPOPolicy(state_size, action_size + self.nbr_extra_actions,
                                             use_replay_buffer=False, in_parameters=None)
        else:
            self.learning_policy = DDDQNPolicy(state_size, action_size + self.nbr_extra_actions,
                                               in_parameters, evaluation_mode)
        self.learning_agent_action = []
        self.done_agents = []
        self.policy_selector_actions = deque(maxlen=10000)

    def get_agent_handles(self, env):
        nbr_of_agents = env.get_num_agents()
        return np.random.choice(nbr_of_agents, nbr_of_agents, False)

    def shape_reward(self, handle, action, state, reward, done, deadlocked=None):
        return self.learning_policy.shape_reward(handle, action, state, reward, done)

    def step(self, handle, state, action, reward, next_state, done):
        if handle in self.learning_agent_action or (done and handle not in self.done_agents):
            self.learning_policy.step(handle, state, action, reward, next_state, done)
            self.deadlock_avoidance_agent.step(handle, state, action, reward, next_state, done)
        if done:
            self.done_agents.append(handle)
        self.loss = self.learning_policy.loss

    def check_is_occupied(self, handle, agent_pos, agent_dir):
        occupiers = self.agent_can_choose_helper.get_switch_cluster_occupiers_next_cell(handle, agent_pos, agent_dir)
        if handle in occupiers:
            return False
        return len(occupiers) > 0

    def mark_next_cell_occupied(self, handle, agent_pos, agent_dir):
        if not self.enable_cluster_lock:
            return
        possible_transitions = self.env.rail.get_transitions(*agent_pos, agent_dir)
        for new_direction in range(4):
            if possible_transitions[new_direction] == 1:
                new_position = get_new_position(agent_pos, new_direction)
                self.agent_can_choose_helper.mark_switch_cluster_occupied(handle, new_position, new_direction)

    def act(self, handle, state, eps=0.):
        agent_pos, agent_dir, agent_status, agent_target = \
            self.agent_can_choose_helper.get_agent_position_and_direction(handle)

        # special case : done/done_removed agents
        if agent_status in [TrainState.DONE]:
            return convert_default_rail_env_action(RailEnvActions.DO_NOTHING)

        # special case : active agents
        if agent_status in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
            # forward walk (moving) iff next switch cluster is not occupied by other agents
            if self.enable_cluster_lock and self.check_is_occupied(handle, agent_pos, agent_dir):
                self.policy_selector_actions.append(3)
                return convert_default_rail_env_action(RailEnvActions.STOP_MOVING)

            # if the agent is on a cell where it can only walk forward -> take the forward action
            agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all = \
                self.agent_can_choose_helper.check_agent_decision(agent_pos, agent_dir)
            if self.enable_auto_move_forward and (not (agents_on_switch or agents_near_to_switch)):
                self.policy_selector_actions.append(2)
                self.mark_next_cell_occupied(handle, agent_pos, agent_dir)
                return convert_default_rail_env_action(RailEnvActions.MOVE_FORWARD)

        # take an action (learning agent)
        action = self.learning_policy.act(handle, state, eps)
        if action == self.action_size:
            self.policy_selector_actions.append(1)
            dl_action = self.deadlock_avoidance_agent.act(handle, state, eps)
            if self.enable_cluster_lock and (map_action(dl_action) in [RailEnvActions.MOVE_LEFT,
                                                                       RailEnvActions.MOVE_FORWARD,
                                                                       RailEnvActions.MOVE_RIGHT]):
                self.mark_next_cell_occupied(handle, agent_pos, agent_dir)
            self.learning_agent_action.append(handle)
            return dl_action

        if self.enable_cluster_lock and (map_action(action) in [RailEnvActions.MOVE_LEFT,
                                                                RailEnvActions.MOVE_FORWARD,
                                                                RailEnvActions.MOVE_RIGHT]):
            if self.check_is_occupied(handle, agent_pos, agent_dir):
                self.policy_selector_actions.append(4)
                return convert_default_rail_env_action(RailEnvActions.STOP_MOVING)

        self.learning_agent_action.append(handle)
        self.mark_next_cell_occupied(handle, agent_pos, agent_dir)
        self.policy_selector_actions.append(0)
        return action

    def save(self, filename):
        self.deadlock_avoidance_agent.save(filename)
        self.learning_policy.save(filename)

    def load(self, filename):
        self.deadlock_avoidance_agent.load(filename)
        self.learning_policy.load(filename)

    def start_step(self, train):
        self.learning_agent_action = []
        self.agent_can_choose_helper.reset_switch_cluster_occupied(handle_only_active_agents=True)
        self.deadlock_avoidance_agent.start_step(train)
        self.learning_policy.start_step(train)

    def end_step(self, train):
        self.deadlock_avoidance_agent.end_step(train)
        self.learning_policy.end_step(train)

    def start_episode(self, train):
        self.deadlock_avoidance_agent.start_episode(train)
        self.learning_policy.start_episode(train)

    def end_episode(self, train):
        self.deadlock_avoidance_agent.end_episode(train)
        self.learning_policy.end_episode(train)

        if self.do_debug_print:
            actions = np.array(self.policy_selector_actions)
            print("\n>> [",
                  "RL: {:.2f}".format(np.sum(actions == 0) / len(actions)),  # RL = Reinforced Learned Agent
                  "DL: {:.2f}".format(np.sum(actions == 1) / len(actions)),  # DL = Dead Lock Avoidance Agent
                  "MF: {:.2f}".format(np.sum(actions == 2) / len(actions)),  # MV = MOVE FORWARD
                  "C1: {:.2f}".format(np.sum(actions == 3) / len(actions)),  # C1 = switch cluster stop (1)
                  "C2: {:.2f}".format(np.sum(actions == 4) / len(actions)),  # C2 = switch cluster stop (2)
                  "]")

    def load_replay_buffer(self, filename):
        self.deadlock_avoidance_agent.load_replay_buffer(filename)
        self.learning_policy.load_replay_buffer(filename)

    def test(self):
        self.deadlock_avoidance_agent.test()
        self.learning_policy.test()

    def reset(self, env: RailEnv):
        self.env = env
        self.agent_can_choose_helper.reset(self.env)
        self.deadlock_avoidance_agent.reset(self.env)
        self.learning_policy.reset(self.env)
        self.done_agents = []

    def clone(self):
        multi_descision_agent = MultiDecisionAgent(
            self.state_size,
            self.action_size,
            self.in_parameters
        )
        multi_descision_agent.deadlock_avoidance_agent = self.deadlock_avoidance_agent.clone()
        multi_descision_agent.learning_policy = self.learning_policy.clone()
        return multi_descision_agent
