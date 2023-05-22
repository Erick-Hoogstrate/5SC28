import numpy as np
import torch
import pandas as pd

TRAIN_DATA = r"disc-benchmark-files\training-data.csv"
TEST_DATA = r"disc-benchmark-files\test-simulation-submission-file.csv"


def load_data(train_test="train", as_tensor=True):
    """
    Loads training or test data from given files.

    params:
    train_test: whether to load train or test data
    as_tensor: whether to output data as a torch tensor or not

    returns:
    x_data, y_data
    """

    if train_test == "train":
        data = pd.read_csv(TRAIN_DATA).to_numpy()
    else:
        data = pd.read_csv(TEST_DATA).to_numpy()

    if as_tensor:
        data = torch.tensor(data)

    return data[:, 0], data[:, 1]


def load_narx_data(n_a, n_b, train_test="train", as_tensor=True):
    """
    Loads training or test data from given files, and formats it as a NARX input([X_{k-n_b}, ..., X_{k-1}, Y_{k-n_a}, ... , Y_{k-1}]).

    params:
    n_a: amount of X-samples considered
    n_b: amount of Y-samples considered
    train_test: whether to load train or test data
    as_tensor: whether to output data as a torch tensor or not

    returns:
    x_data, y_data
    """
    x, y = load_data(train_test, as_tensor)

    X = []
    Y = []
    for k in range(max(n_a, n_b), len(x)):
        X.append(np.concatenate([x[k - n_b : k], y[k - n_a : k]]))
        Y.append(y[k])

    X, Y = np.array(X), np.array(Y)
    if as_tensor:
        X, Y = torch.as_tensor(X), torch.as_tensor(Y)

    return X, Y
