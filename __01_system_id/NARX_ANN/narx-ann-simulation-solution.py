import numpy as np
from model import Narx, DEVICE
import torch

out = np.load(r'..\..\__00_disc-benchmark-files\training-data.npz')
th_train = out['th'] #th[0],th[1],th[2],th[3],...
u_train = out['u'] #u[0],u[1],u[2],u[3],...

data = np.load(r'..\..\__00_disc-benchmark-files\test-simulation-submission-file.npz')
u_test = data['u']
th_test = data['th'] #only the first 50 values are filled the rest are zeros

def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

na = 15
nb = 25
Xtrain, Ytrain = create_IO_data(u_train, th_train, na, nb)

narx_model=Narx(na+nb,50,5).to(DEVICE)
narx_model.load_state_dict(torch.load('narx15K_na15_nb25_nlay5_nnode50_sim'))

Ytrain_pred = narx_model(torch.tensor(Xtrain, device=DEVICE)).detach().cpu().numpy()
print('train prediction errors:')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5,'radians')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/Ytrain.std()*100,'%')


def simulation_IO_model(f, ulist, ylist, skip=50):

    upast = ulist[skip-na:skip].tolist() #good initialization
    ypast = ylist[skip-nb:skip].tolist()
    Y = ylist[:skip].tolist()
    for u in ulist[skip:]:
        x = torch.tensor(np.concatenate([upast,ypast],axis=0), device=DEVICE)
        ypred = f(x).item()
        Y.append(ypred)
        upast.append(u)
        upast.pop(0)
        ypast.append(ypred)
        ypast.pop(0)
    return np.array(Y)

skip = max(na,nb)
th_train_sim = simulation_IO_model(lambda x: narx_model(x[None,:]).detach().cpu().numpy()[0], u_train, th_train, skip=skip)
print('train simulation errors:')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5,'radians')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/th_train.std()*100,'%')


skip = 50
th_test_sim = simulation_IO_model(lambda x: narx_model(x[None,:]).detach().cpu().numpy()[0], u_test, th_test, skip=skip)

assert len(th_test_sim)==len(th_test)
np.savez('narx-ann-simulation-example-submission-file.npz', th=th_test_sim, u=u_test)