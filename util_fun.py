import torch
import numpy as np


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_error_nrms(
    y_predicted: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray
) -> float:
    y_predicted, y_true = [y.detach().cpu().numpy() if type(y) == torch.Tensor else y for y in [y_predicted, y_true]]
    nrms = np.mean((y_predicted - y_true) ** 2) ** 0.5 / np.std(y_true)

    return nrms

def narx_sim_nrms(model, n_a, n_b, x_data, y_data, device=DEVICE, n_val_samples=3000):
    # init upast and ypast as lists.
    upast = [0] * n_b
    ypast = [0] * n_a

    x_data, y_data = [x[:n_val_samples] for x in [x_data, y_data]]
    ylist = []
    for unow in x_data.cpu().detach().numpy():
        # compute the current y given by f
        narx_input = torch.as_tensor(
            np.concatenate([upast, ypast])[None, :]).double()
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

def print_log(msg:str, log_file:str)->None:
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg) 