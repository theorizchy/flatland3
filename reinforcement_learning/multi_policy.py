from collections import deque

from flatland.envs.rail_env import RailEnv

from reinforcement_learning.policy import HybridPolicy
from reinforcement_learning.ppo_agent import PPOPolicy, FLATLandPPOPolicy


class MultiPolicy(HybridPolicy):
    def __init__(self, state_size, action_size,
                 enable_random_network_distillation=True,
                 use_replay_buffer=False):
        self.state_size = state_size
        self.action_size = action_size
        self.loss = 0
        self.ppo_policy_1 = FLATLandPPOPolicy(state_size, action_size,
                                      use_replay_buffer=False,
                                      K_epoch=10,
                                      enable_random_network_distillation=False,
                                      enable_replay_curiosity_sampling=False)
        self.ppo_policy_2 = FLATLandPPOPolicy(state_size, action_size,
                                      use_replay_buffer=use_replay_buffer,
                                      K_epoch=10,
                                      buffer_size=32_000,
                                      enable_random_network_distillation=enable_random_network_distillation,
                                      enable_replay_curiosity_sampling=False)
        self.ppo_policy_3 = FLATLandPPOPolicy(state_size, action_size,
                                      use_replay_buffer=use_replay_buffer,
                                      K_epoch=10,
                                      buffer_size=32_000,
                                      enable_random_network_distillation=enable_random_network_distillation,
                                      enable_replay_curiosity_sampling=True)

        self.memory = self.ppo_policy_1.memory
        self.episode_counter = 0
        self.syn_policy_after_episodes = 20
        if self.syn_policy_after_episodes is not None:
            self.loss_window_1 = deque(maxlen=self.syn_policy_after_episodes)
            self.loss_window_2 = deque(maxlen=self.syn_policy_after_episodes)
            self.loss_window_3 = deque(maxlen=self.syn_policy_after_episodes)

    def shape_reward(self, handle, action, state, reward, done, deadlocked=None):
        return self.ppo_policy_1.shape_reward(handle, action, state, reward, done)

    def load(self, filename):
        self.ppo_policy_1.load(filename)
        self.ppo_policy_2.load(filename)
        self.ppo_policy_3.load(filename)

    def load_replay_buffer(self, filename):
        self.ppo_policy_1.load_replay_buffer(filename)
        self.ppo_policy_2.load_replay_buffer(filename)
        self.ppo_policy_3.load_replay_buffer(filename)

    def save(self, filename):
        self.ppo_policy_1.save(filename)
        self.ppo_policy_2.save(filename)
        self.ppo_policy_3.save(filename)

    def step(self, handle, state, action, reward, next_state, done):
        self.ppo_policy_1.step(handle, state, action, reward, next_state, done)
        self.ppo_policy_2.step(handle, state, action, reward, next_state, done)
        self.ppo_policy_3.step(handle, state, action, reward, next_state, done)

    def act(self, handle, state, eps=0.):
        return self.ppo_policy_1.act(handle, state, eps)

    def reset(self, env: RailEnv):
        self.ppo_policy_1.reset(env)
        self.ppo_policy_2.reset(env)
        self.ppo_policy_3.reset(env)

    def test(self):
        self.ppo_policy_1.test()
        self.ppo_policy_2.test()
        self.ppo_policy_3.test()

    def start_step(self, train):
        self.ppo_policy_1.start_step(train)
        self.ppo_policy_2.start_step(train)
        self.ppo_policy_3.start_step(train)

    def end_step(self, train):
        if self.syn_policy_after_episodes is not None:
            self.loss_window_1.append(self.ppo_policy_1.loss)
            self.loss_window_2.append(self.ppo_policy_2.loss)
            self.loss_window_3.append(self.ppo_policy_3.loss)
        self.ppo_policy_1.end_step(train)
        self.ppo_policy_2.end_step(train)
        self.ppo_policy_3.end_step(train)

    def start_episode(self, train):
        self.ppo_policy_1.start_episode(train)
        self.ppo_policy_2.start_episode(train)
        self.ppo_policy_3.start_episode(train)

    def end_episode(self, train):
        self._update_ppo()
        self.ppo_policy_1.end_episode(train)
        self.ppo_policy_2.end_episode(train)
        self.ppo_policy_3.end_episode(train)

    def _update_ppo(self):
        self.episode_counter += 1
        if self.syn_policy_after_episodes is None:
            return

        if self.episode_counter % self.syn_policy_after_episodes > 0:
            return

        do_update_feature_extractor_model = True
        if self.ppo_policy_1.feature_extractor_model is not None and \
                self.ppo_policy_2.feature_extractor_model is not None and \
                self.ppo_policy_1.feature_extractor_model is not None:
            tau = 0.1
            self._soft_update(self.ppo_policy_2.feature_extractor_model.model,
                              self.ppo_policy_1.feature_extractor_model.model, tau)
            self._soft_update(self.ppo_policy_3.feature_extractor_model.model,
                              self.ppo_policy_1.feature_extractor_model.model, tau)
            self._soft_update(self.ppo_policy_2.critic.model,
                              self.ppo_policy_1.critic.model, tau)
            self._soft_update(self.ppo_policy_3.critic.model,
                              self.ppo_policy_1.critic.model, tau)
            self._soft_update(self.ppo_policy_2.actor.model,
                              self.ppo_policy_1.actor.model, tau)
            self._soft_update(self.ppo_policy_3.actor.model,
                              self.ppo_policy_1.actor.model, tau)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
