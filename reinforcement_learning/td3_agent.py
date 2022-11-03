import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from reinforcement_learning.policy import LearningPolicy
from reinforcement_learning.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, hidsize=256):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.l1 = nn.Linear(state_size, hidsize)
        self.l2 = nn.Linear(hidsize, hidsize)
        self.l3 = nn.Linear(hidsize, action_size)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidsize=256):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Q1 architecture
        self.l1 = nn.Linear(state_size + action_size, hidsize)
        self.l2 = nn.Linear(hidsize, hidsize)
        self.l3 = nn.Linear(hidsize, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_size + action_size, hidsize)
        self.l5 = nn.Linear(hidsize, hidsize)
        self.l6 = nn.Linear(hidsize, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Twin Delayed Deep Deterministic Policy Gradients (TD3)
class TD3Policy(LearningPolicy):
    def __init__(
            self,
            state_size,
            action_size,
            max_action,
            buffer_size=2_000,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            policy_noise=0.1,
            noise_clip=0.2,
            policy_freq=2
    ):
        print(">> TD3Policy")
        super(TD3Policy, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action

        self.device = torch.device("cpu")
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_min_size = 0

        self.actor = Actor(state_size, action_size, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.update_every = 5
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = 0
        self.total_it = 0
        self.t_step = 0

    def act(self, handle, state, eps=0.):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action = (self.actor(state).cpu().data.numpy().flatten() +
                  np.random.normal(0, self.max_action * eps, size=self.action_size)).clip(-self.max_action,
                                                                                          self.max_action)
        act = np.argmax(action)
        return act

    def step(self, handle, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self.train_net()

    def train_net(self):
        if len(self.memory) <= self.buffer_min_size or len(self.memory) <= self.batch_size:
            return

        self.total_it += 1

        # Sample replay buffer
        states, actions, rewards, states_next, dones, _, _ = self.memory.sample()
        actions = torch.squeeze(actions)
        actions = F.one_hot(actions, num_classes=self.action_size)
        states_next = torch.squeeze(states_next)
        rewards = torch.squeeze(rewards)
        dones = torch.squeeze(dones)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            rnd = torch.randn((len(actions), self.action_size), dtype=torch.float32)
            noise = (rnd * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            next_action = (self.actor_target(states_next) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(states_next, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.reshape(rewards, (len(rewards), 1)) + torch.reshape(1.0 - dones, (len(dones), 1)) * \
                       self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.actor, self.actor_target, self.tau)

        self.loss = critic_loss.mean().detach().cpu().numpy()

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
