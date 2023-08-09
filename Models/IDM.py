import torch
import torch.nn as nn

from .General.MLP import *


class IDM(nn.Module):

    def __init__(self, action_size, input=8):
        super(IDM, self).__init__()
        self.model = MlpWithAttention(input, action_size)

    def forward(self, state, nState):
        input = torch.cat((state, nState), 1)
        return self.model(input)


def train(model, data, criterion, optimizer, device):
    if model.training is False:
        model.train()

    s, nS, a = data

    s = s.to(device)
    nS = nS.to(device)
    a = a.to(device)

    optimizer.zero_grad()
    pred = model(s, nS)

    loss = 100 * criterion(pred, a)
    loss.backward()
    for params in model.parameters():
        params.grad.data.clamp_(-1, 1)

    optimizer.step()

    s.detach().cpu()
    nS.detach().cpu()
    a.detach().cpu()

    acc = torch.pow((a - pred), 2).mean(dim=0).detach().cpu()
    return loss.item(), acc


def validation(model, data, device, tensorboard=None):
    if model.training is True:
        model.eval()

    s, nS, a = data

    s = s.to(device)
    nS = nS.to(device)
    a = a.to(device)

    with torch.no_grad():
        pred = model(s, nS)

    s.detach().cpu()
    nS.detach().cpu()
    a.detach().cpu()

    loss = torch.abs((a - pred)).mean().detach().cpu()
    acc = torch.pow((a - pred), 2).mean(dim=0).detach().cpu()
    return loss.item(), acc
