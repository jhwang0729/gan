import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, c1: int, c2: int, normalize: bool = True, device: str = 'cpu'):
        super().__init__()
        self.normalize = normalize
        self.x = nn.Linear(c1, c2, device=device)
        self.bn = nn.BatchNorm1d(c2, 0.8)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.x(x))) if self.normalize else self.act(self.x(x))
