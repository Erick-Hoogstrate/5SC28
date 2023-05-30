import torch
from torch import nn


class Narx(nn.Module):
    def __init__(self, n_in, n_hidden) -> None:
        super().__init__()
        self.lay1 = nn.Linear(n_in, n_hidden).double()
        self.lay2 = nn.Linear(n_hidden, 1).double()

    def forward(self, x):
        x1 = torch.sigmoid(self.lay1(x))
        y = self.lay2(x1)[:, 0]
        return y
