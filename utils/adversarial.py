from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets.VectorDataset import Policy_Vector_Dataset


class AdversarialDataset(Dataset):
    def __init__(self, expert: Policy_Vector_Dataset, policy: Policy_Vector_Dataset) -> None:
        super().__init__()
        self._policy = policy
        self.policy = policy.get_signatures()
        self.expert = expert.get_signatures_by_amount(self.policy.size(0))

        self.trajectories = torch.cat((self.policy, self.expert), dim=0)
        is_expert = torch.ones(self.expert.size(0))
        is_policy = torch.zeros(self.policy.size(0))
        self.is_expert = torch.cat((is_policy, is_expert)).long()

    def __len__(self) -> int:
        return self.trajectories.size(0)

    def __getitem__(self, index: int) -> Union[int, torch.Tensor, torch.Tensor]:
        return index, self.trajectories[index], self.is_expert[index]

    def get_shape(self) -> Union[list, list]:
        return self._policy.states.shape[1:], self._policy.actions.shape[1:]

    def get_distance(self):
        distance = []
        for trajectory in self.policy:
            distance.append((trajectory - self.expert).abs().sum(1).min())
        return np.mean(distance)


def create_dataloader(expert, policy, batch_size=1) -> DataLoader:
    '''This function should create a PyTorch dataloader with both expert and policy trajectories'''
    dataset = AdversarialDataset(expert, policy)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def transform_dict_into_dataset(trajectories) -> Dataset:
    '''
        Transform a dict into a dataset.
        The following keys should be present:
        states: sequence of states from the environment
        next_states: sequence of states from the environment
        actions: sequence of actions from the environment
        starts: True for states that start a trajectory, and False for the others
    '''
    states = trajectories['states']
    next_states = trajectories['next_states']
    actions = trajectories['actions']
    starts = np.array(trajectories['starts']).astype(bool)

    return Policy_Vector_Dataset(states, next_states, actions, starts)


def transform_dataset_into_dict(dataset: AdversarialDataset, idxs: List[int]) -> dict:
    '''Get a dataset and transform into a dict again'''
    state, action = dataset.get_shape()
    run = {
        'states': np.ndarray((0, *state)),
        'next_states': np.ndarray((0, *state)),
        'actions': np.ndarray((0, *action)),
        'starts': np.ndarray((0))
    }

    for idx in idxs:
        trajectory = dataset._policy.get_trajectory_by_index(idx)
        run['states'] = np.append(run['states'], trajectory['states'], axis=0)
        run['next_states'] = np.append(run['next_states'], trajectory['next_states'], axis=0)
        run['actions'] = np.append(run['actions'], trajectory['actions'], axis=0)
        run['starts'] = np.append(run['starts'], trajectory['starts'], axis=0)

    return run


def adversarial(
    expert: Dataset,
    policy: dict,
    optimizer,
    criterion,
    model,
    batch_size,
    device,
    board=None,
) -> dict:
    '''
        This function creates a dataset with all expert and policy samples,
        feeds a discriminator and returns the data that fooled the model as expert.
    '''

    policy = transform_dict_into_dataset(policy)
    dataloader = create_dataloader(expert, policy, batch_size)

    acc, fool_idx = [], []
    for mini_batch in dataloader:
        idx, x, y = mini_batch

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        argmax = torch.argmax(pred.detach().cpu(), dim=1)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        acc += [(((argmax == y.detach().cpu()).sum().item() / y.size(0)) * 100)]

        # All that the model predicted as expert
        pred_expert = np.where(argmax.detach().cpu() == 1)[0]

        # All that are fake
        true_fake = np.where(y.detach().cpu() == 0)[0]

        # Intersection between all that the model predicted as expert and are fake
        intersec = np.intersect1d(pred_expert, true_fake)
        fool_idx += idx.numpy()[intersec].tolist()

    percentage = len(fool_idx) / len(dataloader.dataset)
    acc = np.mean(acc)
    distance = dataloader.dataset.get_distance()

    if board is not None:
        board.add_scalars(
            train=False,
            Discriminator_Acc=acc,
            Fooled_Percentage=percentage,
            Distance=distance,
        )
    
    run = transform_dataset_into_dict(
        dataloader.dataset,
        fool_idx
    )

    return run, percentage, distance
