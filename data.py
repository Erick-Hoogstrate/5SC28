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
