import torch
import numpy as np


def calculate_error_nrms(y_predicted: torch.Tensor or np.ndarray, y_true: torch.Tensor or np.ndarray) -> float:
    y_predicted, y_true = [y.detach().cpu().numpy() if type(y) == torch.Tensor else y for y in [y_predicted, y_true]]
    nrms = np.mean((y_predicted - y_true) ** 2) ** 0.5 / np.std(y_true)

    return nrms