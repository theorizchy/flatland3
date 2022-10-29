import copy
import os

import numpy as np
import torch
import torch.nn as nn

from reinforcement_learning.policy import LearningPolicy
from reinforcement_learning.replay_buffer import ReplayBuffer


# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html


class EpisodeBuffers:
    def __init__(self):
        self.reset()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def reset(self):
        self.memory = {}

    def get_transitions(self, handle):
        return self.memory.get(handle, [])

    def push_transition(self, handle, transition):
        transitions = self.get_transitions(handle)
        transitions.append(transition)
        self.memory.update({handle: transitions})


# Actor module
class FeatureExtractorNetwork(nn.Module):
    def __init__(self, state_size, device, hidsize1=512, hidsize2=256):
        super(FeatureExtractorNetwork, self).__init__()
        self.device = device
        self.nn_layer_outputsize = hidsize2
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh()
        ).to(self.device)

    def forward(self, X):
        return self.model(X)

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.model.state_dict(), filename + ".ppo_feature_extractor")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".ppo_feature_extractor")


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, device, feature_extractor_model: FeatureExtractorNetwork = None,
                 hidsize=256,
                 learning_rate=0.5e-3):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.feature_extractor_model = feature_extractor_model
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize) if (self.feature_extractor_model is None)
            else nn.Linear(feature_extractor_model.nn_layer_outputsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        if self.feature_extractor_model is None:
            return self.model(input)
        return self.model(self.feature_extractor_model(input))

    def get_actor_dist(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs=probs)
        return dist, probs

    def evaluate(self, states, actions):
        dist, action_probs = self.get_actor_dist(states)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def save(self, filename):
        torch.save(self.model.state_dict(), filename + ".ppo_actor")
        torch.save(self.optimizer.state_dict(), filename + ".ppo_optimizer_actor")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".ppo_actor")
        self.optimizer = self._load(self.optimizer, filename + ".ppo_optimizer_actor")


# Critic module
class CriticNetwork(nn.Module):
    def __init__(self, state_size, device, feature_extractor_model: FeatureExtractorNetwork = None, hidsize=256,
                 learning_rate=0.5e-3):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.feature_extractor_model = feature_extractor_model
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize) if (self.feature_extractor_model is None)
            else nn.Linear(feature_extractor_model.nn_layer_outputsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, 1)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        if self.feature_extractor_model is None:
            return self.model(input)
        return self.model(self.feature_extractor_model(input))

    def evaluate(self, states):
        state_value = self.forward(states)
        return torch.squeeze(state_value)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename + ".ppo_critic")
        torch.save(self.optimizer.state_dict(), filename + ".ppo_optimizer_critic")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".ppo_critic")
        self.optimizer = self._load(self.optimizer, filename + ".ppo_optimizer_critic")


class PPOPolicy(LearningPolicy):
    def __init__(self, state_size, action_size, use_replay_buffer=False, in_parameters=None,
                 buffer_size=10_000, batch_size=1024, K_epoch=10,
                 use_shared_feature_extractor=False, clip_grad_norm=0.5,
                 enable_replay_curiosity_sampling=True,
                 skip_unfinished_agent=0.0):
        print(">> PPOPolicy")
        super(PPOPolicy, self).__init__()
        # parameters
        self.state_size = state_size
        self.action_size = action_size
        self.ppo_parameters = in_parameters
        if self.ppo_parameters is not None:
            self.hidsize = self.ppo_parameters.hidden_size
            self.buffer_size = self.ppo_parameters.buffer_size
            self.batch_size = self.ppo_parameters.batch_size
            self.learning_rate = self.ppo_parameters.learning_rate
            self.gamma = self.ppo_parameters.gamma
            # Device
            if self.ppo_parameters.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("🐇 Using GPU")
            else:
                self.device = torch.device("cpu")
                print("🐢 Using CPU")
        else:
            self.hidsize = 128
            self.learning_rate = 0.5e-4
            self.gamma = 0.99
            self.buffer_size = buffer_size
            self.batch_size = batch_size
            self.device = torch.device("cpu")

        self.K_epoch = K_epoch
        self.surrogate_eps_clip = 0.2
        self.weight_loss = 0.5
        self.weight_entropy = 0.001
        self.lmbda = 0.9

        self.skip_unfinished_agent = skip_unfinished_agent

        self.buffer_min_size = 0
        self.use_replay_buffer = use_replay_buffer
        self.enable_replay_curiosity_sampling = enable_replay_curiosity_sampling
        self.enable_replay_curiosity_fix_size_batch_size = True

        self.current_episode_memory = EpisodeBuffers()
        self.agent_done = {}
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = 0

        self.feature_extractor_model = None
        if use_shared_feature_extractor:
            self.feature_extractor_model = FeatureExtractorNetwork(state_size,
                                                                   self.device,
                                                                   hidsize1=self.hidsize,
                                                                   hidsize2=self.hidsize)
        self.actor = ActorNetwork(state_size,
                                  action_size,
                                  self.device,
                                  feature_extractor_model=self.feature_extractor_model,
                                  hidsize=self.hidsize,
                                  learning_rate=self.learning_rate)
        self.critic = CriticNetwork(state_size,
                                    self.device,
                                    feature_extractor_model=self.feature_extractor_model,
                                    hidsize=self.hidsize,
                                    learning_rate=2.0 * self.learning_rate)

        self.loss_function = nn.MSELoss()
        self.clip_grad_norm = clip_grad_norm

    def set_loss_function(self, nn_loss_function):
        # nn.BCEWithLogitsLoss()
        # nn.MSELoss() * default
        # nn.SmoothL1Loss()
        self.loss_function = nn_loss_function

    def rollout_extra_reward(self, transitions_array, all_done):
        return 0

    def reset(self, env):
        pass

    def act(self, handle, state, eps=0.0):
        action, _, _ = self.act_intern(handle, state, eps)
        return action

    def act_intern(self, handle, state, eps=0.0):
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist, action_probs = self.actor.get_actor_dist(torch_state)
        action = dist.sample()
        return action.item(), dist, action_probs

    def step(self, handle, state, action, reward, next_state, done):
        if self.agent_done.get(handle, False):
            return  # remove? if not Flatland?
        # record transitions ([state] -> [action] -> [reward, next_state, done])
        torch_action = torch.tensor(action, dtype=torch.float).to(self.device)
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        torch_next_state = torch.tensor(state, dtype=torch.float).to(self.device)
        # evaluate actor
        dist, _ = self.actor.get_actor_dist(torch_state)
        value = self.critic.evaluate(torch_state).detach().cpu().numpy()
        next_value = self.critic.evaluate(torch_next_state).detach().cpu().numpy()

        action_logprobs = dist.log_prob(torch_action)
        transition = (state, action, reward, next_state, action_logprobs.item(), done, value, next_value)
        self.current_episode_memory.push_transition(handle, transition)
        if done:
            self.agent_done.update({handle: done})

    def _push_transitions_to_replay_buffer(self,
                                           state_list,
                                           action_list,
                                           reward_list,
                                           state_next_list,
                                           done_list,
                                           prob_a_list,
                                           advantages_list):
        for idx in range(len(reward_list)):
            state_i = state_list[idx]
            action_i = action_list[idx]
            reward_i = reward_list[idx]
            state_next_i = state_next_list[idx]
            done_i = done_list[idx]
            prob_action_i = prob_a_list[idx]
            advantage_i = advantages_list[idx]
            self.memory.add(state_i, action_i, reward_i, state_next_i, done_i, prob_action_i, advantage_i)

    def _rollout_episode_buffer(self, transitions_array, all_done):
        # build empty lists(arrays)
        state_list, action_list, return_list, state_next_list, prob_a_list, done_list, advantages_list = \
            [], [], [], [], [], [], []

        # set discounted_reward to zero
        discounted_reward = 0
        extra_reward = self.rollout_extra_reward(transitions_array, all_done)
        for transition in transitions_array[::-1]:
            state_i, action_i, reward_i, state_next_i, prob_action_i, done_i, value_i, next_value_i = transition

            reward_i += extra_reward
            extra_reward = 0

            state_list.insert(0, state_i)
            action_list.insert(0, action_i)
            done_list.insert(0, int(done_i))
            mask_i = 1.0 - int(done_i)

            discounted_reward = reward_i + self.gamma * mask_i * discounted_reward
            return_list.insert(0, discounted_reward)

            advantages_list.insert(0, discounted_reward - value_i)

            state_next_list.insert(0, state_next_i)
            prob_a_list.insert(0, prob_action_i)

        if self.use_replay_buffer:
            self._push_transitions_to_replay_buffer(state_list, action_list,
                                                    return_list, state_next_list,
                                                    done_list, prob_a_list, advantages_list)

        # convert data to torch tensors
        states, actions, returns, states_next, dones, prob_actions, advantages = \
            torch.tensor(state_list, dtype=torch.float).to(self.device), \
            torch.tensor(action_list).to(self.device), \
            torch.tensor(return_list, dtype=torch.float).to(self.device), \
            torch.tensor(state_next_list, dtype=torch.float).to(self.device), \
            torch.tensor(done_list, dtype=torch.float).to(self.device), \
            torch.tensor(prob_a_list).to(self.device), \
            torch.tensor(advantages_list).to(self.device),

        # Normalize the rewards and advantages
        returns = (returns - returns.mean()) / (returns.std() + 1.0e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        return states, actions, returns, states_next, dones, prob_actions, advantages

    def _sample_replay_buffer(self, states, actions, returns, states_next, dones, probs_action, advantages):
        # https://arxiv.org/pdf/1611.01224v2.pdf
        if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
            states, actions, returns, states_next, dones, probs_action, advantages = self.memory.sample()

            states = torch.squeeze(states)
            actions = torch.squeeze(actions)
            returns = torch.squeeze(returns)
            states_next = torch.squeeze(states_next)
            dones = torch.squeeze(dones)
            probs_action = torch.squeeze(probs_action)
            advantages = torch.squeeze(advantages)

            # curiosity filtering - just use the 50 percent highest positive difference
            if self.enable_replay_curiosity_sampling:
                # Focus on observed rewards which are higher than the cirtic expect (estimate) - surprise
                deltas = (returns - self.critic.evaluate(states).detach()).pow(2.0)
                # Find indices for filtering
                indices = torch.nonzero(deltas.ge(deltas.median()), as_tuple=False).squeeze(1)
                # Apply filter
                states = torch.index_select(states, 0, indices)
                actions = torch.index_select(actions, 0, indices)
                returns = torch.index_select(returns, 0, indices)
                states_next = torch.index_select(states_next, 0, indices)
                dones = torch.index_select(dones, 0, indices)
                probs_action = torch.index_select(probs_action, 0, indices)
                advantages = torch.index_select(advantages, 0, indices)

                if self.enable_replay_curiosity_fix_size_batch_size:
                    # Enforce fix-size batch_size -> fit the batch_size by appending missing data with randomized picked
                    states2, actions2, returns2, states_next2, dones2, probs_action2, advantages2 = \
                        self.memory.sample(k_samples=max(10, self.memory.batch_size - len(indices)))
                    # concatenate the data
                    states = torch.cat((states, torch.squeeze(states2)))
                    actions = torch.cat((actions, torch.squeeze(actions2)))
                    returns = torch.cat((returns, torch.squeeze(returns2)))
                    states_next = torch.cat((states_next, torch.squeeze(states_next2)))
                    dones = torch.cat((dones, torch.squeeze(dones2)))
                    probs_action = torch.cat((probs_action, torch.squeeze(probs_action2)))
                    advantages = torch.cat((advantages, torch.squeeze(advantages2)))

        return states, actions, returns, states_next, dones, probs_action, advantages

    def train_net(self):
        # All agents have to propagate their experiences made during past episode
        all_done = True
        for handle in range(len(self.current_episode_memory)):
            all_done = all_done and self.agent_done.get(handle, False)

        for handle in range(len(self.current_episode_memory)):
            if (not self.agent_done.get(handle, False)) and (np.random.random() < self.skip_unfinished_agent):
                continue

            # Extract agent's episode history (list of all transitions)
            agent_episode_history = self.current_episode_memory.get_transitions(handle)
            if len(agent_episode_history) > 0:
                # Convert the replay buffer to torch tensors (arrays)
                states, actions, returns, states_next, dones, probs_action, advantages = \
                    self._rollout_episode_buffer(agent_episode_history, all_done)

                # Optimize policy for K epochs:
                do_k_epoch = int(np.ceil(max(1.0, self.K_epoch / max(1, len(self.agent_done)))))
                for k_loop in range(do_k_epoch):
                    # update by random sampling
                    if self.use_replay_buffer:
                        states, actions, returns, states_next, dones, probs_action, advantages = \
                            self._sample_replay_buffer(
                                states, actions, returns, states_next, dones, probs_action, advantages
                            )

                    # Evaluating actions (actor) and values (critic)
                    logprobs, dist_entropy = self.actor.evaluate(states, actions)

                    # Finding the ratios (pi_thetas / pi_thetas_replayed):
                    delta_logprobs = logprobs - probs_action.detach()
                    ratios = torch.exp(delta_logprobs)

                    # Calculate the current values
                    state_values = self.critic.evaluate(states)

                    # Finding Surrogate Loos
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios,
                                        1. - self.surrogate_eps_clip,
                                        1. + self.surrogate_eps_clip) * advantages

                    # The loss function is used to estimate the gardient and use the entropy function based
                    # heuristic to penalize the gradient function when the policy becomes deterministic this would let
                    # the gradient becomes very flat and so the gradient is no longer useful.
                    loss_actor = \
                        -torch.min(surr1, surr2).mean() \
                        - self.weight_entropy * dist_entropy.mean()
                    loss_critic = \
                        self.weight_loss * self.loss_function(state_values, returns)

                    loss = \
                        loss_actor + \
                        loss_critic

                    # Make a gradient step -> update actor and critic
                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    loss_actor.backward()
                    loss_critic.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.model.parameters(), self.clip_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.model.parameters(), self.clip_grad_norm)
                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

                    # Transfer the current loss to the agents loss (information) for debug purpose only
                    self.loss = loss.mean().detach().cpu().numpy()

    def end_episode(self, train):
        if train:
            self.train_net()
        # Reset all collect transition data
        self.current_episode_memory.reset()
        self.agent_done = {}

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        if self.feature_extractor_model is not None:
            self.feature_extractor_model.save(filename)
        self.actor.save(filename)
        self.critic.save(filename)

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        else:
            print(" >> file not found!")
        return obj

    def load(self, filename):
        print("load policy and optimizer from file", filename)
        if self.feature_extractor_model is not None:
            self.feature_extractor_model.load(filename)
        self.actor.load(filename)
        self.critic.load(filename)

    def clone(self):
        policy = PPOPolicy(self.state_size, self.action_size)
        if self.feature_extractor_model is not None:
            policy.feature_extractor_model = copy.deepcopy(self.feature_extractor_model)
        policy.actor = copy.deepcopy(self.actor)
        policy.critic = copy.deepcopy(self.critic)
        return self


class FLATLandPPOPolicy(PPOPolicy):
    def __init__(self, state_size, action_size, use_replay_buffer=False, in_parameters=None,
                 buffer_size=10_000, batch_size=1024, K_epoch=10,
                 use_shared_feature_extractor=False, clip_grad_norm=0.5, enable_replay_curiosity_sampling=True,
                 skip_unfinished_agent=0.0):
        print(">> FLATLandPPOPolicy")
        super(FLATLandPPOPolicy, self).__init__(state_size, action_size, use_replay_buffer, in_parameters,
                                                buffer_size, batch_size, K_epoch, use_shared_feature_extractor,
                                                clip_grad_norm, enable_replay_curiosity_sampling,
                                                skip_unfinished_agent)
        self.deadlocked_agent = {}

    def act(self, handle, state, eps=0.0):
        return super(FLATLandPPOPolicy, self).act(handle, state, eps)

    def rollout_extra_reward(self, transitions_array, all_done):
        return 0
        if all_done and len(self.deadlocked_agent.keys()) == 0:
            return pow(len(self.agent_done), 2.0)
        return 0

    def end_episode(self, train):
        super(FLATLandPPOPolicy, self).end_episode(train)
        self.deadlocked_agent = {}

    def shape_reward(self, handle, action, state, reward, done, deadlocked=None):
        if self.deadlocked_agent.get(handle, False):
            return 0.0
        if self.agent_done.get(handle, False):
            return 0.0

        is_deadlocked = False
        if deadlocked is not None:
            is_deadlocked = deadlocked
        if is_deadlocked:
            self.deadlocked_agent.update({handle: True})
            return -0.01
        if done:
            return 1.0 + 1000.0 / (1.0 + len(self.current_episode_memory.get_transitions(handle)))

        return -0.00001
