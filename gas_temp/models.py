import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def linear(in_channels, out_channels, batch_norm=False):
    layers = []
    layers.append(nn.Linear(in_channels, out_channels, bias = False))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)

class GasTempNet(nn.Module):
    def __init__(self, hidden_dim, in_dims, out_dims):
        super (Net, self).__init__()
        self.input = linear(in_dims, hidden_dim)
        self.output = linear(hidden_dim, out_dims)

    def forward(self, x):
        out = F.relu(self.input(x))
        out = self.output(x)
        return out
