from collections import deque

import numpy as np
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.a2c_agent import A2CPolicy
from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.policy import DummyMemory, HybridPolicy
from reinforcement_learning.ppo_agent import FLATLandPPOPolicy
from utils.agent_action_config import convert_default_rail_env_action
from utils.agent_can_choose_helper import AgentCanChooseHelper
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent


class DeadLockAvoidanceWithDecisionAgent(HybridPolicy):

    def __init__(self, env, state_size, action_size, in_parameters=None, evaluation_mode=False, use_policy=0):
        print(">> DeadLockAvoidanceWithDecisionAgent")
        super(DeadLockAvoidanceWithDecisionAgent, self).__init__()

        self.do_debug_print = True
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
        if use_policy == 0:
            self.learning_policy = FLATLandPPOPolicy(state_size, action_size + self.nbr_extra_actions,
                                                     use_replay_buffer=in_parameters.buffer_size > 0,
                                                     in_parameters=in_parameters)
            self.memory = self.learning_policy.memory
        elif use_policy == 1:
            self.learning_policy = A2CPolicy(state_size, action_size + self.nbr_extra_actions,
                                             use_replay_buffer=True, in_parameters=None)
            self.memory = self.learning_policy.memory
        else:
            self.learning_policy = DDDQNPolicy(state_size, action_size + self.nbr_extra_actions,
                                               in_parameters, evaluation_mode)
            self.memory = self.learning_policy.memory
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

    def act(self, handle, state, eps=0.):
        agent_pos, agent_dir, agent_status, agent_target = \
            self.agent_can_choose_helper.get_agent_position_and_direction(handle)

        # special case : done/done_removed agents
        if agent_status in [TrainState.DONE]:
            return convert_default_rail_env_action(RailEnvActions.DO_NOTHING)

        # take an action (learning agent)
        self.learning_agent_action.append(handle)
        action = self.learning_policy.act(handle, state, eps)
        if action == self.action_size:
            self.policy_selector_actions.append(1)
            dl_action = self.deadlock_avoidance_agent.act(handle, state, eps)
            return dl_action

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
        multi_descision_agent = DeadLockAvoidanceWithDecisionAgent(
            self.state_size,
            self.action_size,
            self.in_parameters
        )
        multi_descision_agent.deadlock_avoidance_agent = self.deadlock_avoidance_agent.clone()
        multi_descision_agent.learning_policy = self.learning_policy.clone()
        return multi_descision_agent
