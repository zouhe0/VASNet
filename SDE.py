import torch
from torch import nn


class Net_ms2pan(nn.Module):
    def __init__(self):
        super(Net_ms2pan, self).__init__()
        self.net = nn.Sequential(nn.Linear(8, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1),)

    def forward(self, ms):
        out = self.net(ms.permute(0, 2, 3, 1))
        return out.permute(0, 3, 1, 2)