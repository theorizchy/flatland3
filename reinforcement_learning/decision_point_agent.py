from collections import deque

import numpy as np
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.policy import HybridPolicy
from utils.agent_action_config import convert_default_rail_env_action, map_action
from utils.agent_can_choose_helper import AgentCanChooseHelper
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent
from utils.fast_methods import fast_isclose
from utils.shortest_path_walking_agent import ShortestPathWalkingAgent


class DecisionPointAgent(HybridPolicy):

    def __init__(self, env: RailEnv, state_size, action_size, learning_agent, plugin_dead_lock_avoidance_agent=True):
        print(">> DecisionPointAgent")
        super(DecisionPointAgent, self).__init__()
        self.env = env

        self.training = True

        self.state_size = state_size
        self.action_size = action_size
        self.learning_agent = learning_agent
        self.plugin_dead_lock_avoidance_agent = plugin_dead_lock_avoidance_agent

        self.memory = self.learning_agent.memory
        self.loss = self.learning_agent.loss

        self.learning_agent_action = []
        self.done_agents = []

        self.agent_can_choose_helper = AgentCanChooseHelper()
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(env, action_size,
                                                                    enable_eps=False,
                                                                    break_at_switch=0)
        self.shortest_path_walking_agent = ShortestPathWalkingAgent(env)

        self.policy_selector_actions = deque(maxlen=50000)
        self.do_debug_print = True

    def shape_reward(self, handle, action, state, reward, done, deadlocked=None):
        return self.learning_agent.shape_reward(handle, action, state, reward, done)

    def step(self, handle, state, action, reward, next_state, done):
        if handle in self.learning_agent_action or (done and handle not in self.done_agents):
            self.learning_agent.step(handle, state, action, reward, next_state, done)
        if done:
            self.done_agents.append(handle)
        self.loss = self.learning_agent.loss

    def act(self, handle, state, eps=0.):
        # this action is used in learning
        self.learning_agent_action.append(handle)

        # get agent and position, status
        agent = self.env.agents[handle]
        agent_pos, agent_dir, agent_status, agent_target = \
            self.agent_can_choose_helper.get_agent_position_and_direction(handle)

        # classify agent's cell
        agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all = \
            self.agent_can_choose_helper.check_agent_decision(agent_pos, agent_dir)

        # special case : done/done_removed agents
        if agent_status in [TrainState.DONE]:
            return convert_default_rail_env_action(RailEnvActions.DO_NOTHING)

        # special case : ready to depart
        if agent_status == TrainState.READY_TO_DEPART:
            if self.plugin_dead_lock_avoidance_agent:
                self.policy_selector_actions.append(0)
                return self.dead_lock_avoidance_agent.act(handle, state, eps)

        # special case : switch cluster still have an agent
        # if the switch cluster has still an agent in -> no new should enter
        if not self.agent_can_choose_helper.can_agent_enter_next_cluster(handle):
            self.policy_selector_actions.append(2)
            return convert_default_rail_env_action(RailEnvActions.STOP_MOVING)

        # ------------------------- RL ----------------------------------------------------------
        # take an action (learning agent)
        action = self.learning_agent.act(handle, state, eps)

        # Use dead lock avoidance to improve training?
        # is dead lock avoidance learning enable (for this episode) check for possible deadlock
        if self.plugin_dead_lock_avoidance_agent:
            cluster_id, grid_cell_members = self.agent_can_choose_helper.get_switch_cluster(agent_pos)
            # if map_action(action) in [RailEnvActions.DO_NOTHING, RailEnvActions.STOP_MOVING] or cluster_id < 1:
            if cluster_id < 1 and not agents_near_to_switch:
                self.policy_selector_actions.append(3)
                return self.dead_lock_avoidance_agent.act(handle, state, eps)

            if map_action(action) in [RailEnvActions.DO_NOTHING]:  # , RailEnvActions.STOP_MOVING]:
                self.policy_selector_actions.append(4)
                return self.dead_lock_avoidance_agent.act(handle, state, eps)

        self.policy_selector_actions.append(5)
        return action

    def save(self, filename):
        self.learning_agent.save(filename)
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.save(filename)

    def load(self, filename):
        self.learning_agent.load(filename)
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.load(filename)

    def start_episode(self, train):
        self.training = train
        self.learning_agent.start_episode(train)
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.start_episode(train)
        self.shortest_path_walking_agent.start_episode(train)

        if train and self.do_debug_print:
            actions = np.array(self.policy_selector_actions)
            print('[0: {:.2f} 1: {:.2f} 2: {:.2f} 3: {:.2f} 4: {:.2f} 5: {:.2f}]'.format(
                np.sum(actions == 0) / max(1, len(actions)),  # Ready to depart
                np.sum(actions == 1) / max(1, len(actions)),  # action still chosen
                np.sum(actions == 2) / max(1, len(actions)),  # ST = Stop cluster occupied
                np.sum(actions == 3) / max(1, len(actions)),  # Dead lock avoidance
                np.sum(actions == 4) / max(1, len(actions)),  # Semi RL = Reinforced Learned Agent
                np.sum(actions == 5) / max(1, len(actions))),  # RL = Reinforced Learned Agent
                end='', flush=True)

    def start_step(self, train):
        self.training = train
        self.learning_agent_action = []
        self.learning_agent.start_step(train)
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.start_step(train)
        self.shortest_path_walking_agent.start_step(train)
        self.agent_can_choose_helper.reset_switch_cluster_occupied()

    def end_step(self, train):
        self.training = train
        self.learning_agent.end_step(train)
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.end_step(train)
        self.shortest_path_walking_agent.end_step(train)

    def end_episode(self, train):
        self.training = train
        self.learning_agent.end_episode(train)
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.end_episode(train)
        self.shortest_path_walking_agent.end_episode(train)

    def load_replay_buffer(self, filename):
        self.learning_agent.load_replay_buffer(filename)
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.load_replay_buffer(filename)
        self.shortest_path_walking_agent.load_replay_buffer(filename)

    def test(self):
        self.learning_agent.test()
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.test()
        self.shortest_path_walking_agent.test()

    def reset(self, env: RailEnv):
        self.env = env
        self.agent_can_choose_helper.reset(self.env)
        self.learning_agent.reset(env)
        self.done_agents = []
        if self.plugin_dead_lock_avoidance_agent:
            self.dead_lock_avoidance_agent.reset(env)
        self.shortest_path_walking_agent.reset(env)

    def clone(self):
        return self
