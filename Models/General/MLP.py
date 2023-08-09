import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .Attention import Self_Attn1D


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.5):
        super(MLP, self).__init__()

        out = max(8, 32)
        self.input = nn.Linear(in_dim, (out))
        self.fc = nn.Linear((out), (out))
        self.fc2 = nn.Linear((out), (out))
        self.output = nn.Linear((out), out_dim)
        self.dropout = nn.Dropout(p=0.9, inplace=False)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.input(x))
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class MlpWithAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        out = max(512, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        
        self.fc = nn.Linear(out, out)
        self.attention = Self_Attn1D(out, nn.LeakyReLU)
        
        self.fc2 = nn.Linear(out, out)
        self.attention2 = Self_Attn1D(out, nn.LeakyReLU)
        
        self.fc3 = nn.Linear(out, out)        
        self.fc4 = nn.Linear(out, out)        
        self.output = nn.Linear(out, out_dim)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = x.float()
 
        x = self.input(x)
        x = self.relu(x)

        x = self.fc(x)
        x = self.relu(x)
        x, _ = self.attention(x)

        x = self.fc2(x)
        x = self.relu(x)
        x, _ = self.attention2(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.relu(x)

        return self.output(x)


class MlpAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MlpAttention, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)
        self.attention = Self_Attn1D(out, nn.LeakyReLU)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.input(x)
        x, att = self.attention(x)
        x = self.relu(self.output(x))
        return x
