import itertools
import random
from collections import deque

import gym
import numpy as np
import torch
from gym.envs.classic_control.cartpole import CartPoleEnv
from torch.nn import functional

from dqn import DQN
from replay_memory import ReplayMemory

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1
EPSILON_END = 0.02
EPSILON_DECAY = 10000
OPTIMIZER_LR = 5e-4
TARGET_UPDATE_FREQ = 1000
TRANSITION_SIZE = [4, 1, 1, 1, 4]  # state, action, reward, done, new_state

env: CartPoleEnv = gym.make('CartPole-v0').unwrapped

replay_memory = ReplayMemory(transition_size=TRANSITION_SIZE, maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

online_net = DQN(env=env, optimizer_lr=OPTIMIZER_LR)
target_net = DQN(env=env, optimizer_lr=OPTIMIZER_LR)

target_net.load_state_dict(online_net.state_dict())

# Initialize Replay Buffer
state = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_state, reward, done, _ = env.step(action=action)

    replay_memory.push(state=state, action=action, reward=reward, done=done, next_state=new_state)
    state = new_state

    if done:
        state = env.reset()

# Main Training Loop
episodes = 1
episode_reward = 0.0

state = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    if random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.get_action(state=state)

    new_state, reward, done, _ = env.step(action=action)
    episode_reward += reward

    replay_memory.push(state=state, action=action, reward=reward, done=done, next_state=new_state)
    state = new_state

    if done:
        state = env.reset()
        episodes += 1

        reward_buffer.append(episode_reward)
        episode_reward = 0

    # After solved, watch it play
    if len(reward_buffer) >= 100 and np.mean(reward_buffer) >= 200:
        print('Solved', step)
        total_r = 0
        while True:
            action = online_net.get_action(state=state)

            state, re, done, info = env.step(action=action)
            total_r += re
            env.render()
            if done:
                print(total_r, info)
                total_r = 0
                env.reset()

    # Start Gradient Step
    states_t, actions_t, rewards_t, dones_t, new_states_t = replay_memory.get_batch(batch_size=BATCH_SIZE)

    # Compute Targets
    target_q_values = target_net(new_states_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute Loss
    q_values = online_net(states_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    # https://pytorch.org/docs/1.9.0/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    # Similar to HuberLoss.
    loss = functional.smooth_l1_loss(input=action_q_values, target=targets)

    # Gradient Descent Step
    online_net.optimizer.zero_grad()
    loss.backward()
    online_net.optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step, 'Episodes', episodes)
        print('Avg reward', np.mean(reward_buffer))
