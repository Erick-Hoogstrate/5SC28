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


if __name__ == '__main__':
    my_narx=Narx(5,10)
    my_data=torch.rand((10,5)).double()
    assert my_narx.forward(my_data).shape == torch.Size([10]), 'something went wrong'
    print('test passed')