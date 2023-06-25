import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from typing import Union, Tuple
from dataclasses import dataclass
import model

TRAIN_DATA = r"..\..\__00_disc-benchmark-files\training-data.npz"


def load_data(
    as_tensor: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Loads training or test data from given files.

    params:
    as_tensor: whether to output data as a torch tensor or not

    returns:
    x_data, y_data
    """

    data = np.load(TRAIN_DATA)

    x, y = data["u"], data["th"]

    if as_tensor:
        x, y = torch.tensor(x), torch.tensor(y)

    return x, y


def load_narx_data(
    n_a: int, n_b: int, as_tensor: bool = True
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
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
    x, y = load_data(as_tensor)
    x, y = convert_to_narx(x, y, n_a, n_b, as_tensor)

    return x, y


def convert_to_narx(
    x, y, n_a, n_b, as_tensor: bool = True
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    X = []
    Y = []
    for k in range(max(n_a, n_b), len(x)):
        X.append(np.concatenate([x[k - n_b : k], y[k - n_a : k]]))
        Y.append(y[k])

    X, Y = np.array(X), np.array(Y)
    if as_tensor:
        X, Y = torch.as_tensor(X), torch.as_tensor(Y)

    return X, Y


@dataclass
class GS_Dataset:
    x_data_train: np.ndarray = None
    y_data_train: np.ndarray = None
    x_data_val: np.ndarray = None
    y_data_val: np.ndarray = None
    x_data_test: np.ndarray = None
    y_data_test: np.ndarray = None
    x_train: torch.Tensor = None
    y_train: torch.Tensor = None
    x_val: torch.Tensor = None
    y_val: torch.Tensor = None
    x_test: torch.Tensor = None
    y_test: torch.Tensor = None


@dataclass
class GS_Results:
    best_model: model.Narx = None
    best_sim_model: model.Narx = None
    best_nrms: float = None
    best_sim_nrms: float = None
    loss_list: list = None
    nrms_list: list = None
    sim_nrms_list: list = None


def make_gs_dataset(
    x_data: np.ndarray, y_data: np.ndarray, n_a: int, n_b: int, device: torch.device
) -> GS_Dataset:
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data, y_data, shuffle=False
    )
    x_data_train, x_data_val, y_data_train, y_data_val=train_test_split(
        x_data_train, y_data_train, shuffle=False
    )

    x_train, y_train = convert_to_narx(x_data_train, y_data_train, n_a, n_b)
    x_test, y_test = convert_to_narx(x_data_test, y_data_test, n_a, n_b)
    x_val, y_val = convert_to_narx(x_data_val, y_data_val, n_a, n_b)
    x_train, x_test, x_val, y_train, y_test, y_val = [
        x.to(device) for x in [x_train, x_test, x_val, y_train, y_test, y_val]
    ]

    return GS_Dataset(
        x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test,x_train, y_train, x_val, y_val, x_test, y_test
    )
