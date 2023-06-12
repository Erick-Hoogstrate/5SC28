import torch
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Narx(nn.Module):
    def __init__(self, n_in, n_hidden_nodes, n_hidden_layers=2) -> None:
        super().__init__()

        # add input, hidden, and output layers
        layers = [nn.Linear(n_in, n_hidden_nodes), nn.Sigmoid()]
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
            layers.append(nn.Sigmoid())
        layers += [nn.Linear(n_hidden_nodes, 1)]
        layers = [layer.double() for layer in layers]

        # unroll layers into sequential net
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)[:, 0]



if __name__ == "__main__":
    my_narx = Narx(5, 10)
    my_data = torch.rand((10, 5)).double()
    assert my_narx.forward(my_data).shape == torch.Size([10]), "something went wrong"
    print("test passed")
