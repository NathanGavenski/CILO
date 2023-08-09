import os
from typing import List

from imitation_datasets import Policy, Context, get_args, Controller, Experts
import numpy as np
from tqdm import tqdm
import gym


def run(expert: Policy, path: str, context: Context) -> bool:
    env = gym.make(expert.get_environment())
    state_shape = env.reset().shape
    expert.load()

    ep_returns = 0
    states = np.ndarray((0, *state_shape))
    next_states = np.ndarray((0, *state_shape))
    actions = []

    obs, done = env.reset(), False
    tmp_states = np.ndarray((0, *state_shape))
    tmp_next_states = np.ndarray((0, *state_shape))
    tmp_actions = []

    while not done:
        tmp_states = np.append(tmp_states, obs[None], axis=0)
        action, _ = expert.predict(obs)
        tmp_actions.append(action)

        obs, reward, done, _ = env.step(action)

        tmp_next_states = np.append(tmp_next_states, obs[None], axis=0)

        ep_returns += reward

    if ep_returns >= expert.threshold:
        print(f'Agent reached {ep_returns}')
        states = np.append(states, tmp_states, axis=0)
        next_states = np.append(next_states, tmp_next_states, axis=0)
        actions += tmp_actions

        dataset = {
            'state': states,
            'actions': actions,
            'next_states': next_states,
        }
        np.savez(f'{path}/tmp_{expert.get_environment()}_{context.index}_{ep_returns}', **dataset)
        return True
    else:
        print(f'Agent did not reach threshold ({expert.threshold}) - reached {ep_returns}')
        return False


def create_data_file(path: str, data: List[str]):
    try:
        data = [_f for _f in data if "npz" in _f]
        f_path = f"{path}{data[0]}"
        shape = np.load(f_path, allow_pickle=True)['state'].shape[1:]
        action_shape = np.load(f_path, allow_pickle=True)['actions'].shape[1:]
        
        states = np.ndarray((0, *shape))
        next_states = np.ndarray((0, *shape))
        actions = np.ndarray((0, *action_shape))
        starts = []
        rewards = []

        for f in tqdm(data):
            _data = np.load(f"{path}/{f}", allow_pickle=True)
            states = np.append(states, _data['state'], axis=0)
            next_states = np.append(next_states, _data['next_states'], axis=0)
            actions = np.append(actions, _data['actions'], axis=0)
            reward = float(f.split('_')[-1].replace('.npz', ''))
            rewards.append(reward)
            start = np.zeros(_data['state'].shape[0])
            start[0] = 1
            starts += start.astype(bool).tolist()

        np.savez(
            f"{path}teacher",
            **{
                'states': states,
                'next_states': next_states,
                'actions': actions,
                'starts': starts,
                'expert': np.mean(rewards),
                'std': np.std(rewards)
            }
        )

        print(f"Expert reward: {np.mean(rewards)} ± {np.std(rewards)}")

        for f in tqdm(data):
            os.remove(f"{path}/{f}")
    
        return True
    finally:
        return False


if __name__ == "__main__":
    os.system("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")
    os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia")

    args = get_args()
    controller = Controller(run, create_data_file, args.episodes, args.threads)
    controller.start(args)
