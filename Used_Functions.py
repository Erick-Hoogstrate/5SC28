import pandas as pd
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from data import load_data
from sklearn.model_selection import train_test_split

def make_OE_init_state_data(udata, ydata, nf=100, n_encode=20):
    U = [] 
    Y = [] 
    hist = [] 
    for k in range(nf+n_encode,len(udata)+1):
        hist.append(np.concatenate((udata[k-nf-n_encode:k-nf], ydata[k-nf-n_encode:k-nf])))
        U.append(udata[k-nf:k])
        Y.append(ydata[k-nf:k])
    return np.array(hist), np.array(U), np.array(Y)

class simple_encoder_RNN(nn.Module):
    def __init__(self, hidden_size, n_encoder=20):
        super(simple_encoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 1
        self.output_size = 1
        net = lambda n_in,n_out: nn.Sequential(nn.Linear(n_in,40),nn.Sigmoid(),nn.Linear(40,n_out)).double() #short hand for a 1 hidden layer NN
        self.h2h = net(self.input_size + hidden_size, self.hidden_size)
        self.h2o = net(self.input_size + hidden_size, self.output_size)
        self.encoder = net(n_encoder*2,hidden_size).double()

    def forward(self,inputs,hist):
        hidden = self.encoder(hist)
        outputs = []
        for i in range(inputs.shape[1]):
            input = inputs[:,i]
            combined = torch.cat((hidden, input[:,None]), 1)
            outputs.append(self.h2o(combined)[:,0])
            hidden = self.h2h(combined)
        return torch.stack(outputs,dim=1)
    
def get_SSNN_data(ulist, model):
    upast = []
    ypast =[]
    ylist = []

    for unow in ulist:
        ynow = model(upast,ypast)
        upast.append(unow)
        upast.pop(0)
        ypast.append(ynow)
        ypast.pop(0)

        ylist.append(ynow)
    return np.array(ylist)