from matplotlib import pyplot as plt
import torch
import numpy as np


def calculate_error_nrms(y_predicted: torch.Tensor or np.ndarray, y_true: torch.Tensor or np.ndarray, as_percentage: bool=True) -> float:
    y_predicted, y_true = [y.detach().cpu().numpy() if type(y) == torch.Tensor else y for y in [y_predicted, y_true]]
    nrms = np.mean((y_predicted - y_true) ** 2) ** 0.5 / np.std(y_true)

    return nrms*100 if as_percentage else nrms


def use_NARX_model_in_simulation(ulist, f, na, nb):
    #init upast and ypast as lists.
    upast = [0]*nb 
    ypast = [0]*na 
    
    ylist = []
    for unow in ulist:
        #compute the current y given by f
        ynow = f(upast,ypast) 
        
        #update past arrays
        upast.append(unow)
        upast.pop(0)
        ypast.append(ynow)
        ypast.pop(0)
        
        #save result
        ylist.append(ynow)
    return np.array(ylist) #return result


def plot_NRMS_Pred_vs_Sim(NRMS_pred, NRMS_sim, na_list, nb_list): # Plot results for validation data
    fontsize=15
    fig = plt.figure(figsize=(15, 5),layout='constrained')
    ax1=fig.add_subplot(121)
    ax1.imshow(NRMS_pred, interpolation='none', norm='log')
    ax2=fig.add_subplot(122)
    ax2.imshow(NRMS_sim, interpolation='none', norm='log')
    ax1.set_title('Prediction NRMS',fontsize=fontsize)
    ax2.set_title('Simulation NRMS',fontsize=fontsize)
    fig.suptitle('Prediction and simulation validation NRMS for different combinations of na and nb', fontsize=fontsize+10)
    ax1.set_ylabel('na',fontsize=fontsize)
    ax1.set_xlabel('nb',fontsize=fontsize)
    ax2.set_xlabel('nb',fontsize=fontsize)
    ax1.set_yticklabels([0]+na_list)
    ax1.set_xticks([*range(len(nb_list))],nb_list)
    ax2.set_yticklabels([0]+na_list)
    ax2.set_xticks([*range(len(nb_list))],nb_list)

    params = np.zeros((len(na_list), len(nb_list), 2))
    for i, n_a in enumerate(na_list):
        for j, n_b in enumerate(nb_list):
            params[i, j, :] = [n_a, n_b]

    min_arg = np.unravel_index(NRMS_pred.argmin(keepdims=True), NRMS_pred.shape)
    best_na, best_nb = params[min_arg].ravel()
    print(f"Best parameters by prediction NRMS: na= {best_na}, nb= {best_nb}")

    min_arg = np.unravel_index(NRMS_sim.argmin(keepdims=True), NRMS_sim.shape)
    best_na, best_nb = params[min_arg].ravel()
    print(f"Best parameters by simulation NRMS: na= {best_na}, nb= {best_nb}")