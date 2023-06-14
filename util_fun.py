<<<<<<< HEAD
import torch
import numpy as np


def calculate_error_nrms(y_predicted: torch.Tensor or np.ndarray, y_true: torch.Tensor or np.ndarray) -> float:
    y_predicted, y_true = [y.detach().cpu().numpy() if type(y) == torch.Tensor else y for y in [y_predicted, y_true]]
    nrms = np.mean((y_predicted - y_true) ** 2) ** 0.5 / np.std(y_true)

    return nrms
=======
from copy import deepcopy
import torch
import numpy as np
from data import GS_Dataset, GS_Results
from model import Narx
from typing import Tuple


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_error_nrms(
    y_predicted: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray
) -> float:
    y_predicted, y_true = [
        y.detach().cpu().numpy() if type(y) == torch.Tensor else y
        for y in [y_predicted, y_true]
    ]
    nrms = (np.mean((y_predicted - y_true) ** 2) ** 0.5) / np.std(y_true)

    return nrms


def narx_sim_nrms(
    model,
    n_a: int,
    n_b: int,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    init_values: bool = False,
    device: torch.device = DEVICE,
    n_val_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # convert tensors to ndarrays if applicable
    x_data, y_data = [
        x.detach().cpu().numpy() if type(x) == torch.Tensor else x
        for x in [x_data, y_data]
    ]
    # init upast and ypast as lists.
    if init_values:
        skip = max(n_a, n_b)
        upast = list(x_data[skip - n_b : skip])
        ypast = list(y_data[skip - n_a : skip])
        data_end = min(len(x_data), n_val_samples + skip)
    else:
        upast = [0] * n_b
        ypast = [0] * n_a
        data_end = n_val_samples
        skip=0

    x_data, y_data = [x[skip:data_end] for x in [x_data, y_data]]
    ylist = []
    for unow in x_data:
        # compute the current y given by f
        narx_input = torch.as_tensor(np.concatenate([upast, ypast])[None, :]).double()
        narx_input = narx_input.to(device)
        ynow = model.forward(narx_input).cpu().detach().item()

        # update past arrays
        upast.append(unow)
        upast.pop(0)
        ypast.append(ynow)
        ypast.pop(0)

        # save result
        ylist.append(ynow)

    nrms = calculate_error_nrms(np.array(ylist), y_data)
    return x_data, y_data, np.array(ylist), nrms


def train_narx_simval(
    model: Narx,
    n_a: int,
    n_b: int,
    data: GS_Dataset,
    log_file: str = None,
    param_msg: str = "",
    n_epochs: int = 100,
    device: torch.device = DEVICE,
):
    # initialise comparison values and results lists
    best_nrms = float("inf")
    best_model = None
    best_sim_nrms = float("inf")
    best_sim_model = None
    loss_list = []
    nrms_list = []
    sim_nrms_list = []

    # initialise checkpoints for validation
    checkpoints = [*range(0, n_epochs + 1, max(n_epochs // 25, 1))]
    if checkpoints[-1] != n_epochs:
        checkpoints += [n_epochs]
    optimizer = torch.optim.Adam(model.parameters())
    # start training loop
    for epoch in range(n_epochs):
        loss = torch.mean(((model(data.x_train) - data.y_train) ** 2)/torch.std(data.y_train))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch in checkpoints:
            print_log(f"Checkpoint at epoch {epoch+1}: " + param_msg + " \n", log_file)
            # append loss to list, check prediction and simulation nrms
            loss_list.append(loss.item())

            nrms = calculate_error_nrms(model.forward(data.x_val), data.y_val)
            nrms_list.append(nrms)
            if nrms < best_nrms:
                print_log(
                    f"current best pred NRMS: {nrms}, previous best pred NRMS: {best_nrms} \n",
                    log_file,
                )
                best_nrms = nrms
                best_model = deepcopy(model)
            else:
                print_log(
                    f"current pred NRMS: {nrms}, current best pred NRMS: {best_nrms} \n",
                    log_file,
                )
            _, _, _, sim_nrms = narx_sim_nrms(
                model, n_a, n_b, data.x_data, data.y_data, True, device
            )
            sim_nrms_list.append(sim_nrms)
            if sim_nrms < best_sim_nrms:
                print_log(
                    f"current best sim NRMS: {sim_nrms}, previous best sim NRMS: {best_sim_nrms} \n",
                    log_file,
                )
                best_sim_nrms = sim_nrms
                best_sim_model = deepcopy(model)
            else:
                print_log(
                    f"current sim NRMS: {sim_nrms}, current best sim NRMS: {best_sim_nrms} \n",
                    log_file,
                )

    results = GS_Results(
        best_model,
        best_sim_model,
        best_nrms,
        best_sim_nrms,
        loss_list,
        nrms_list,
        sim_nrms_list,
    )
    return results


def print_log(msg: str, log_file: str = None) -> None:
    print(msg)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg)
>>>>>>> 2cf30b2c88046bc805d0ac3f72f38a7045ab0476
