from matplotlib import pyplot as plt
import numpy as np
import torch
from data import load_narx_data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, Matern, Product

n_a, n_b = 2, 2

Xtrain,Ytrain = load_narx_data(n_a, n_b, section="train", split=[0.6, 0.2, 0.2], as_tensor=True)
Xval,Yval = load_narx_data(n_a, n_b, section="validation", split=[0.6, 0.2, 0.2], as_tensor=True)
Xtest,Ytest = load_narx_data(n_a, n_b, section="test", split=[0.6, 0.2, 0.2], as_tensor=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Narx(nn.Module):
    def __init__(self, n_in, n_hidden_nodes, n_hidden_layers=2) -> None:
        super().__init__()

        # add input, hidden, and output layers
        layers = [nn.Linear(n_in, n_hidden_nodes), nn.Sigmoid()]
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
            layers.append(nn.ReLU())
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
