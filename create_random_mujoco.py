import argparse
import warnings
from copy import deepcopy
import os
import gym

import numpy as np

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from tqdm import tqdm


"""
To create a new random dataset you just need to inform env_name (need to be the gym name environment)
and where you want to save the random dataset (e.g., ./dataset/ant/random_ant)

Ex: python create_random_mujoco.py --env_name Walker2d-v3 --data_path ./dataset/walker/random_walker
"""


def get_args():
    parser = argparse.ArgumentParser(description='Args for general configs for CILO')

    parser.add_argument('--env_name', default=None, type=str)
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    return args


if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name)

    state = env.reset()
    states = np.ndarray((0, *state.shape))
    next_states = np.ndarray((0, *state.shape))
    actions = np.ndarray((0, *env.action_space.shape))
    rewards = []

    total_reward = 0
    for i in tqdm(range(int(50000))):
        states = np.append(states, state[None], axis=0)

        action = env.action_space.sample()
        actions = np.append(actions, np.array([action]), axis=0)

        next_state, reward, done, _ = env.step(action)
        next_states = np.append(next_states, next_state[None], axis=0)
        total_reward += reward

        if done:
            done = True
            state = env.reset()
            rewards.append(total_reward)
            total_reward = 0
        else:
            state = deepcopy(next_state)

    path = '/'.join(args.data_path.split('/')[:-1])
    if not os.path.exists(path):
        os.makedirs(path)

    np.savez(
        args.data_path,
        **{
            'states': states,
            'next_states': next_states,
            'actions': actions,
            'random': np.mean(rewards),
            'std': np.std(rewards)
        }
    )
