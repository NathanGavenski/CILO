import torch
import torch.nn as nn

from .General.MLP import *


class Policy(nn.Module):

    def __init__(self, action_size, input=4):
        super(Policy, self).__init__()
        self.action_size = action_size
        self.model = MlpWithAttention(input, action_size)

    def forward(self, state):
        return self.model(state)

    def act(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state)

            if len(state.size()) < 2:
                state = state[None]

            return self.forward(state)[0].detach().cpu().numpy()


def train(model, idm_model, data, criterion, optimizer, device, args, actions=None, tensorboard=None):
    if not model.training:
        model.train()

    if idm_model.training:
        idm_model.eval()

    s, nS, a_gt = data

    s = s.to(device)
    nS = nS.to(device)
    a_gt = a_gt.to(device)

    with torch.no_grad():
        prediction = idm_model(s, nS)
    if args.choice == 'explore':
        if actions is None:
            actions = prediction.std(dim=0)

        actions = actions[None].repeat_interleave(prediction.size(0), dim=0)
        action = torch.normal(prediction, actions.to(device))
        action = torch.clip(action, -1, 1)
    else:
        action = prediction
        action = torch.clip(action, -1, 1)

    if tensorboard is not None:
        _action = torch.swapaxes(action, 1, 0)
        _a_gt = torch.swapaxes(a_gt, 1, 0)

        for i, (a, gt) in enumerate(zip(_action, _a_gt)):
            tensorboard.add_histogram(f'Train/Action Distribution {i}', a)
            tensorboard.add_histogram(f'Train/GT Action Distribution {i}', gt)

        if args.choice == 'explore':
            _actions = torch.swapaxes(actions, 1, 0)
            for i, std in enumerate(_actions):
                tensorboard.add_histogram(f'Train/Action STD {i}', std)

    optimizer.zero_grad()
    pred = model(s)

    loss = 100 * criterion(pred, action)
    loss.backward()
    for params in model.parameters():
        params.grad.data.clamp_(-1, 1)

    optimizer.step()

    s.detach().cpu()
    nS.detach().cpu()
    a_gt.detach().cpu()

    acc = torch.pow((action - pred), 2).mean(dim=0).detach().cpu()
    idm_acc = torch.pow((a_gt - prediction), 2).mean(dim=0).detach().cpu()
    return loss.item(), acc, idm_acc


def validation(model, idm_model, data, device, args, actions=None, tensorboard=None):
    if model.training:
        model.eval()

    if idm_model.training:
        idm_model.eval()

    s, nS, a_gt = data

    s = s.to(device)
    nS = nS.to(device)

    with torch.no_grad():
        prediction = idm_model(s, nS)
    if args.choice == 'explore':
        if actions is None:
            actions = prediction.std(dim=0)

        actions = actions[None].repeat_interleave(prediction.size(0), dim=0)
        action = torch.normal(prediction, actions.to(device))
    else:
        action = prediction

    if tensorboard is not None:
        _action = torch.swapaxes(action, 1, 0)
        _a_gt = torch.swapaxes(a_gt, 1, 0)

        for i, (a, gt) in enumerate(zip(_action, _a_gt)):
            tensorboard.add_histogram(f'Train/Action Distribution {i}', a)
            tensorboard.add_histogram(f'Train/GT Action Distribution {i}', gt)

        if args.choice == 'explore':
            _actions = torch.swapaxes(actions, 1, 0)
            for i, std in enumerate(_actions):
                tensorboard.add_histogram(f'Train/Action STD {i}', std)

    with torch.no_grad():
        pred = model(s)

    s.detach().cpu()
    nS.detach().cpu()

    acc = torch.pow((action - pred), 2).mean(dim=0).detach().cpu()
    return acc
