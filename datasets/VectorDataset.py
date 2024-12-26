from collections import defaultdict
import torch
import numpy as np
import signatory

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.enjoy import get_environment

np.set_printoptions(suppress=True)


def detect_path(file):
    if '/' in file:
        return True
    else:
        return False


def read_vector(dataset_path, env_name, idm=True):
    state_size = get_environment({'name': env_name}).reset().shape[0]
    count = defaultdict(list)
    actions = []
    states = np.ndarray((0, 2, state_size), dtype=str)
    with open(f'{dataset_path}{env_name}.txt') as f:
        for idx, line in enumerate(f):
            word = line.replace('\n', '').split(';')
            state = np.fromstring(word[0].replace('[', '').replace(']', '').replace(',', ' '), sep=' ', dtype=float)
            nState = np.fromstring(word[1].replace('[', '').replace(']', '').replace(',', ' '), sep=' ', dtype=float)

            s = np.append(state[None], nState[None], axis=0)

            action = int(word[-1])
            actions.append(action)
            states = np.append(states, s[None], axis=0)
            count[action].append(idx)

    return count, states, np.array(actions)


def balance_dataset(dataset_path, env_name, downsample_size=5000, replace=True, sampling=True, vector=False):
    data = read_vector(dataset_path, env_name)
    count, states, actions = data

    sizes = []
    dict_sizes = {}
    for key in count:
        sizes.append(len(count[key]))
        dict_sizes[key] = len(count[key])
    print('Size each action:', dict_sizes)

    if sampling is True:
        max_size = np.min(sizes) if downsample_size is not None else None
        downsample_size = downsample_size if downsample_size is not None else np.inf
        downsample_size = min(downsample_size, max_size)

    classes = list(range(3))
    all_idxs = np.ndarray((0), dtype=np.int32)
    if downsample_size is not None:
        for i in classes:
            size = len(count[i])

            try:
                random_idxs = np.random.choice(size, downsample_size, replace=replace)
            except ValueError:
                random_idxs = np.random.choice(size, size, replace=replace)

            idxs = np.array(count[i])[random_idxs]
            all_idxs = np.append(all_idxs, idxs, axis=0).astype(int)

        states = states[all_idxs]
        a = actions[all_idxs]
    else:
        a = actions

    print('Final size action:', np.bincount(a))
    return states, a


def split_dataset(states, actions, stratify=True):
    if stratify:
        return train_test_split(states, actions, test_size=0.3, stratify=actions)
    else:
        return train_test_split(states, actions, test_size=0.3)


class IDM_Vector_Dataset(Dataset):

    transforms = transforms.Compose([
        torch.from_numpy,
    ])

    def __init__(self, states, next_states, actions):
        super().__init__()
        self._states = states
        self._next_states = next_states
        self._actions = actions

        self.states = states
        self.next_states = next_states
        self.actions = actions

    def __len__(self):
        return self.actions.shape[0]

    def reset(self):
        self.states = self._states
        self.next_states = self._next_states
        self.actions = self._actions

    def sample(self, amount):
        indexes = np.arange(0, self._states.size(0))
        indexes = np.random.choice(indexes, amount)
        return indexes

    def __getitem__(self, idx):        
        s = torch.from_numpy(self.states[idx])
        nS = torch.from_numpy(self.next_states[idx])
        a = torch.tensor(self.actions[idx])

        return (s, nS, a)


class Policy_Vector_Dataset(Dataset):

    transforms = transforms.Compose([
        torch.from_numpy
    ])

    expert = 0
    random = 0

    def __init__(self, states, next_states, actions, starts=None, depth=2):
        super().__init__()
        states = states
        next_states = next_states

        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.starts = starts
        begins = np.where(self.starts == True)[0]
        ends = np.append(begins[1:], [self.states.shape[0]], axis=0)
        self.begins, self.ends = begins, ends

        self.trajectories = np.array([self.states[b:e] for (b, e) in zip(begins, ends)])

        stream, channels = self.trajectories[0].shape
        size = signatory.signature(torch.rand(1, stream, channels), depth).size(-1)
        signatures = torch.Tensor(size=(0, size))
        for trajectory in self.trajectories:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            signature = signatory.signature(self.transforms(trajectory[None]).to(device), depth).detach().cpu()
            signatures = torch.cat((signatures, signature), dim=0)

        self.signatures = signatures

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.states[idx])
        nS = torch.from_numpy(self.next_states[idx])
        a = torch.tensor(self.actions[idx])

        return (s, nS, a)

    def get_signatures(self):
        return self.signatures

    def get_signatures_by_amount(self, amount):
        indexes = np.arange(0, self.signatures.size(0))
        indexes = np.random.choice(indexes, amount)
        return self.signatures[indexes]

    def get_trajectory_by_index(self, idx) -> dict:
        begin, end = self.begins[idx], self.ends[idx]
        idxs = [idx for idx in range(begin, end)]
        states = self.states[idxs]
        next_states = self.next_states[idxs]
        actions = self.actions[idxs]
        starts = self.starts[idxs]
        return {
            'states': states,
            'next_states': next_states,
            'actions': actions,
            'starts': starts
        }

    def get_performance_rewards(self):
        return self.expert, self.random

    def get_performance(self, reward):
        return (reward - self.random) / (self.expert - self.random)


def get_idm_vector_dataset(
    path,
    batch_size,
    shuffle=True,
    **kwargs
):
    trajectories = np.load(path, allow_pickle=True)
    states = trajectories['states']
    next_states = trajectories['next_states']
    actions = trajectories['actions']

    train_idx = int(states.shape[0] * 0.7)
    states_train, states_eval = states[:train_idx], states[train_idx:]
    next_states_train, next_states_eval = next_states[:train_idx], next_states[train_idx:]
    actions_train, actions_eval = actions[:train_idx], actions[train_idx:]

    train_dataset = IDM_Vector_Dataset(states_train, next_states_train, actions_train)
    validation_dataset = IDM_Vector_Dataset(states_eval, next_states_eval, actions_eval)
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation


def get_policy_vector_dataset(
    path,
    random_path,
    batch_size,
    shuffle=True,
    repeat=10,
    valid=False,
    **kwargs
):
    trajectories = np.load(path, allow_pickle=True)
    
    if valid:
        starts = np.where(trajectories['starts'] == True)[0]
        train_idx = int(len(starts) * 0.7)
        valid_idx = train_idx + 1
        train_idx = starts[train_idx]

        states = trajectories['states'][:train_idx]
        next_states = trajectories['next_states'][:train_idx]
        actions = trajectories['actions'][:train_idx]
        starts = trajectories['starts'][:train_idx]

        valid_states = trajectories['states'][train_idx:]
        valid_next_states = trajectories['next_states'][train_idx:]
        valid_actions = trajectories['actions'][train_idx:]
        valid_starts = trajectories['starts'][train_idx:]
    else:
        repeat = 1 if repeat < 1 else repeat
        states = np.repeat(trajectories['states'], repeats=repeat, axis=0)
        next_states = np.repeat(trajectories['next_states'], repeats=repeat, axis=0)
        actions = np.repeat(trajectories['actions'], repeats=repeat, axis=0)
        starts = np.repeat(trajectories['starts'][None], repeats=repeat, axis=0)
        starts = starts.flatten()

    train_dataset = Policy_Vector_Dataset(states, next_states, actions, starts)
    train_dataset.expert = float(trajectories['expert'])
    train_dataset.random = float(np.load(random_path, allow_pickle=True)['random'])
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    if valid:    
        valid_dataset = Policy_Vector_Dataset(valid_states, valid_next_states, valid_actions, valid_starts)
        valid_dataset.expert = float(trajectories['expert'])
        valid_dataset.random = float(np.load(random_path, allow_pickle=True)['random'])
        valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
        return train, valid
    else:
        return train, []
        
