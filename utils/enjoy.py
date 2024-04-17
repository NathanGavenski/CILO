from collections import defaultdict
from copy import deepcopy
import os
import shutil

import numpy as np
from PIL import Image
import torch


class EnjoyController():
    def __init__(self, max_count):
        self.prev_state = defaultdict(int)
        self.max_count = max_count

    def append(self, state):
        state = tuple(state)
        self.prev_state[state] += 1
        return True if self.prev_state[state] > self.max_count else False


def play(
    env,
    model,
    dataset,
    transforms,
    device,
    tensorboard=None,
):
    done, state = False, env.reset()

    total_reward = 0
    run = {
        'states': np.ndarray((0, *state.shape)),
        'next_states': np.ndarray((0, *state.shape)),
        'actions': np.ndarray((0, *env.action_space.shape)),
    }

    while not done:
        s = transforms(state)
        s = s.view(1, *s.shape)
        s = s.to(device)

        action = model.act(s)
        action = np.clip(action, -1, 1)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if dataset is True:
            run['states'] = np.append(run['states'], state[None], axis=0)
            run['next_states'] = np.append(run['next_states'], next_state[None], axis=0)
            run['actions'] = np.append(run['actions'], action[None], axis=0)

        state = deepcopy(next_state)

    if tensorboard is not None:
        actions = run['actions']
        actions = torch.swapaxes(torch.Tensor(actions), 1, 0)
        for i, a in enumerate(actions):
            tensorboard.add_histogram(f'Test/Action Distribution {i}', a)

    return total_reward, True, run


def get_environment(domain, size=None, seed=None, random=None):
    import gym
    import mujoco_py

    if domain['name'] == 'swimmer':
        return gym.make('Swimmer-v2')
    elif domain['name'] == 'ant':
        return gym.make('Ant-v2')
    elif domain['name'] == 'walker':
        return gym.make('Walker2d-v2')
    elif domain['name'] == 'hopper':
        return gym.make('Hopper-v2')
    elif domain['name'] == 'humanoid':
        return gym.make('Humanoid-v2')
    elif domain['name'] == 'cheetah':
        return gym.make('HalfCheetah-v2')
    elif domain['name'] == 'pendulum':
        return gym.make('InvertedPendulum-v2')
