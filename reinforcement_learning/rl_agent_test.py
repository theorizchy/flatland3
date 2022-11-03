from collections import deque
from collections import namedtuple

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning.a2c_agent import A2CPolicy
from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.multi_policy import MultiPolicy
from reinforcement_learning.ppo_agent import PPOPolicy

dddqn_param_nt = namedtuple('DDDQN_Param', ['hidden_size', 'buffer_size', 'batch_size', 'update_every', 'learning_rate',
                                            'tau', 'gamma', 'buffer_min_size', 'use_gpu'])
dddqn_param = dddqn_param_nt(hidden_size=128,
                             buffer_size=32_000,
                             batch_size=1024,
                             update_every=5,
                             learning_rate=0.5e-3,
                             tau=0.5e-2,
                             gamma=0.95,
                             buffer_min_size=0,
                             use_gpu=False)


def do_training(env_key, use_policy, max_episode, shape_reward_function, make_decision=None, do_render=False):
    eps = 1.0
    eps_decay = 0.9975
    min_eps = 0.01
    training_mode = True

    env = gym.make(env_key)

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    print(env_key, 'state:', observation_space, 'actions', action_space)

    comment = ''
    if use_policy == 1:
        comment = '_A2CPolicy'
        policy = A2CPolicy(observation_space, action_space)
    elif use_policy == 2:
        comment = '_PPOPolicy_ReplayBufferOn'
        policy = PPOPolicy(observation_space, action_space,
                           use_replay_buffer=True)
    elif use_policy == 3:
        comment = '_PPOPolicy_ReplayBufferOn_RCS'
        policy = PPOPolicy(observation_space, action_space,
                           use_replay_buffer=True,
                           enable_replay_curiosity_sampling=True)
    elif use_policy == 4:
        comment = '_PPOPolicy_ReplayBufferOff'
        policy = PPOPolicy(observation_space, action_space,
                           use_replay_buffer=False)
    elif use_policy == 5:
        comment = '_MultiPolicy'
        policy = MultiPolicy(observation_space, action_space)
    else:  # use_policy == 0:
        comment = '_DDDQNPolicy'
        policy = DDDQNPolicy(observation_space, action_space, dddqn_param)

    if make_decision is not None:
        comment = '_HARDCODED'

    episode = 0
    checkpoint_interval = 20
    scores_window = deque(maxlen=100)
    shaped_scores_window = deque(maxlen=100)

    writer = SummaryWriter(comment='_' + env_key + comment)

    is_rendering_window_open = False
    while episode < max_episode:
        episode += 1
        state = env.reset()
        policy.reset(env)
        handle = 0
        tot_reward = 0
        tot_shaped_reward = 0

        policy.start_episode(train=training_mode)
        while True:
            policy.start_step(train=training_mode)
            if make_decision is not None:
                action = make_decision(handle, state, eps)
            else:
                action = policy.act(handle, state, eps)
            state_next, reward, terminal, info = env.step(action)
            tot_reward += reward
            s_reward = shape_reward_function(handle, action, state, reward, terminal)
            tot_shaped_reward += s_reward
            policy.step(handle, state, action, s_reward, state_next, terminal)
            policy.end_step(train=training_mode)
            state = np.copy(state_next)

            if terminal:
                break

            if episode % 50 == 0 and do_render:
                env.render()
                is_rendering_window_open = True

        policy.end_episode(train=training_mode)
        eps = max(min_eps, eps * eps_decay)
        scores_window.append(tot_reward)
        shaped_scores_window.append(tot_shaped_reward)
        if episode % checkpoint_interval == 0:
            print('\rEpisode: {:5}\treward: {:10.3f}\t avg: {:10.3f}\tshaped reward: {:10.3f}\t avg: {:10.3f}\t eps: {'
                  ':5.3f}\t replay buffer: {} '.format(
                episode,
                tot_reward,
                np.mean(
                    scores_window),
                tot_shaped_reward,
                np.mean(
                    shaped_scores_window),
                eps,
                len(policy.memory)))
        else:
            print('\rEpisode: {:5}\treward: {:10.3f}\t avg: {:10.3f}\tshaped reward: {:10.3f}\t avg: {:10.3f}\t eps: {'
                  ':5.3f}\t replay buffer: {}'.format(episode,
                                                      tot_reward,
                                                      np.mean(
                                                          scores_window),
                                                      tot_shaped_reward,
                                                      np.mean(
                                                          shaped_scores_window),
                                                      eps,
                                                      len(policy.memory)),
                  end=" ")

        writer.add_scalar(env_key + "/org_value", tot_reward, episode)
        writer.add_scalar(env_key + "/shaped_value", tot_shaped_reward, episode)
        writer.add_scalar(env_key + "/loss", policy.loss, episode)
        writer.add_scalar(env_key + "/org_smoothed_value", np.mean(scores_window), episode)
        writer.add_scalar(env_key + "/shaped_smoothed_value", np.mean(shaped_scores_window), episode)
        writer.add_scalar(env_key + "/eps", eps, episode)
        writer.add_scalar(env_key + "/replay_buffer_size", len(policy.memory), episode)
        writer.flush()

    if is_rendering_window_open:
        env.close()


def shape_reward_default(handle, action, state, reward, done):
    return reward


def make_decision_cart_pole(handle, obs, eps):
    L = 1
    dt = 0.027
    x, v, theta, omega = obs
    predicted_theta = theta + omega * dt
    predicted_x_cart = x + v * dt
    predicted_x_tip_pole = predicted_x_cart + np.sin(predicted_theta) * L
    return 0 if predicted_x_tip_pole < predicted_x_cart else 1


def make_decision_mountain_car(handle, obs, eps):
    position, velocity = obs
    actions = {'left': 0, 'stop': 1, 'right': 2}
    if velocity > 0:
        action = actions['right']
    elif velocity < 0:
        action = actions['left']
    else:
        action = actions['left']
    return action


def shape_reward_mountain_car(handle, action, state, reward, done):
    extra = 0.0
    if state[0] > -0.2:
        extra = -1.0
    return extra


if __name__ == "__main__":
    max_episode = 2000
    runs = 1

    # if true -> rendering each 50 episode
    do_render = False

    # if true -> use hardcoded policy (act)
    enable_hardcoded = False

    env_keys = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    for env_key in env_keys:
        for run_loop in range(runs):
            for select_algo in [2, 3, 4]:
                make_decision = None
                shape_reward_function = shape_reward_default
                if env_key == "MountainCar-v0":
                    if enable_hardcoded:
                        make_decision = make_decision_mountain_car
                    # shape_reward_function = shape_reward_mountain_car
                elif env_key == "CartPole-v1":
                    if enable_hardcoded:
                        make_decision = make_decision_cart_pole
                do_training(env_key, select_algo, max_episode, shape_reward_function, make_decision, do_render)
