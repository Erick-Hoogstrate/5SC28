import torch
import numpy as np


def calculate_error_nrms(y_predicted:torch.Tensor|np.ndarray, y_true:torch.Tensor|np.ndarray) -> float:
    if type(y_predicted)==torch.Tensor:
        nrms = torch.mean((y_predicted - y_true) ** 2).item() ** 0.5 /torch.std(y_true)
        nrms=nrms.item()
    else:
        nrms = np.mean((y_predicted - y_true) ** 2).item() ** 0.5 /np.std(y_true)

    return nrms
