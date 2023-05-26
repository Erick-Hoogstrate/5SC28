import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

TRAIN_DATA = r"disc-benchmark-files\training-data.csv"


def load_data(section="train", split=[0.6, 0.2, 0.2], total_number_of_points=2000, as_tensor=True):
    """
    Loads training, validation or test data from given files.

    params:
    section: whether to load train, validation or test data
    split: training, validation, test split. "[train, val, test]"
    total_number_of_points: specifies how many data points are considered from the total data set
    as_tensor: whether to output data as a torch tensor or not

    returns:
    x_data, y_data
    """
    data = pd.read_csv(TRAIN_DATA).to_numpy()[:total_number_of_points]
    print(f"Considering {len(data)} datapoints")
    X_train_val, X_test, y_train_val, y_test = train_test_split(data[:, 0], data[:, 1], test_size=split[2], shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=split[1] / (split[0] + split[1]), shuffle=False)

    if section == "train":
        x = X_train
        y = y_train
        print(f"Using {len(X_train)} datapoints for training set")
    elif section == "validation":
        x = X_val
        y = y_val
        print(f"Using {len(X_val)} datapoints for validation set")
    elif section == "test":
        x = X_test
        y = y_test
        print(f"Using {len(X_test)} datapoints for test set")
    else:
        raise ValueError("Invalid section value. Must be 'train', 'validation', or 'test'.")

    if as_tensor:
        x = torch.tensor(x)
        y = torch.tensor(y)

    return x, y


def load_narx_data(n_a, n_b, total_number_of_points, section="train", split=[0.6, 0.2, 0.2], as_tensor=True):
    """
    Loads training, validation or test data from given files, and formats it as a NARX input([X_{k-n_b}, ..., X_{k-1}, Y_{k-n_a}, ... , Y_{k-1}]).

    params:
    n_a: amount of X-samples considered
    n_b: amount of Y-samples considered
    section: whether to load train, validation or test data
    split: training, validation, test split. "[train, val, test]"
    total_number_of_points: specifies how many data points are considered from the total data set
    as_tensor: whether to output data as a torch tensor or not

    returns:
    x_data, y_data
    """
    x, y = load_data(section, split, total_number_of_points,as_tensor)

    X = []
    Y = []
    for k in range(max(n_a, n_b), len(x)):
        X.append(np.concatenate([x[k - n_b : k], y[k - n_a : k]]))
        Y.append(y[k])

    X, Y = np.array(X), np.array(Y)
    if as_tensor:
        X, Y = torch.as_tensor(X), torch.as_tensor(Y)

    return X, Y
