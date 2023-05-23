import numpy as np
import torch
import pandas as pd

TRAIN_DATA = r"disc-benchmark-files\training-data.csv"


def load_data(as_tensor=True):
    """
    Loads training data from given files.

    params:
    as_tensor: whether to output data as a torch tensor or not

    returns:
    x_data, y_data
    """

    data = pd.read_csv(TRAIN_DATA).to_numpy()

    if as_tensor:
        data = torch.tensor(data)

    return data[:, 0], data[:, 1]
