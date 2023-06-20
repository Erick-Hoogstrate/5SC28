import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable
from swing_up_enviroment import SwingUpCartPoleEnv, SwingUpCartPoleEnv_sincos
np.random.seed(0)

def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]

def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
    X = []; Y = [];
    x = env.reset()
    ep_return_full = 0
    ep_return_sampled = 0
    for timestep in range(timesteps):
        if render: env.render()
        u = policy(env, pilco, x, random)
        for i in range(SUBS):
            x_new, r, done, _ = env.step(u)
            ep_return_full += r
            if done: break
            if render: env.render()
        if verbose:
            print("Action: ", u)
            print("State : ", x_new)
            print("Return so far: ", ep_return_full)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        ep_return_sampled += r
        x = x_new
        if done: break
    return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full

def nrollout(n, env, horizon, SUBS, render=True, verbose=True):
    X, Y = [], []
    for i in range(n):
        Xroll,Yroll, _, _ = rollout(env=env, pilco=None, random=True, timesteps=horizon, SUBS=SUBS, render=render, verbose=verbose) 
        X.append(Xroll)
        Y.append(Yroll)
    return np.concatenate(X,axis=0), np.concatenate(Y,axis=0)

#define environment
dt = 0.1
umax = 10
SUBS = 1
sincos_angle = True
env = SwingUpCartPoleEnv_sincos(umax=umax,dt=dt) if sincos_angle else SwingUpCartPoleEnv(umax=umax,dt=dt)
state_dim = env.observation_space.shape[0]
control_dim = env.action_space.shape[0]
horizon = 30
norm = True
n_start_rollout = 10
nc = 10 # number of RFS of the controller


#initial state
m_init =  np.mean([env.reset() for i in range(100)],axis=0)[None] #(1,Nobs) state mean
S_init =  0.1 * np.eye(state_dim) #state standard deviation

#target state and reward function
target         = np.array([0., 0., 0., 0., -1.]) if sincos_angle else np.array([0., 0., 0., np.pi])
# target_weights = np.diag([2, 2, 2, 2, 2.])       if sincos_angle else np.diag([0.05, 0.05, 0.05, 0.25])
target_weights = np.diag([1, 1, 1, 1, 1.])       if sincos_angle else np.diag([0.05, 0.05, 0.05, 0.25])


controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=nc, max_action=umax)

if norm:
    X,Y = nrollout(n_start_rollout, env, horizon, SUBS, render=False,verbose=False)
    std = np.std(X,axis=0)[:-1] #approx (2,2,5,0.5,0.5)
    if sincos_angle:
        class SwingUpCartPoleEnv_sincos_norm(SwingUpCartPoleEnv_sincos):
            def get_obs(self):
                x, dx, dth, th = self.state
                return np.array([x, dx, dth, np.sin(th), np.cos(th)])/std
        env = SwingUpCartPoleEnv_sincos_norm()
    else:
        class SwingUpCartPoleEnv_norm(SwingUpCartPoleEnv_sincos):
            def get_obs(self):
                x, dx, dth, th = self.state
                return np.array([x, dx, dth, th])/std
        env = SwingUpCartPoleEnv_norm()
    target = target/std #rescale target #is this still correct?
    m_init = m_init/std
    



X, Y = nrollout(n_start_rollout, env, horizon, SUBS, render=False,verbose=False)
print('mean std:',np.mean(X,axis=0)[:-1],np.std(X,axis=0)[:-1]) #(2,2,5,0.5,0.5)
print('target = ',target)

R = ExponentialReward(state_dim=state_dim,
                      t=target,
                      W=target_weights
                     ) #define target with t and the weights with w

pilco = PILCO((X, Y), controller=controller, horizon=horizon, reward=R, m_init=m_init, S_init=S_init)

#fix the variance of the model
for model in pilco.mgpr.models: #for each output?
    model.likelihood.variance.assign(0.03) #fix the variance
    set_trainable(model.likelihood.variance, False)

for rollouts in range(1,10):
    

    pilco.optimize_models() #update transition model
    pilco.optimize_policy(maxiter=100, restarts=1) #use model to optimize policy
    X_new, Y_new, ep_return_sampled, ep_return_full = rollout(env=env, pilco=pilco, timesteps=horizon, SUBS=SUBS, render=True,verbose=False) #rollout 

    #compute reward over the new episode
    total_r = sum([R.compute_reward(X_new[i,None,:-1], 0.001 * np.eye(state_dim))[0] for i in range(len(X_new))])
    _, _, r = pilco.predict(m_init, S_init, horizon)
    print('##########################')
    print(f"rollout = {rollouts}")
    print('inherent reward', ep_return_full)
    print("Gained    Total exp reward", float(total_r[0,0]))
    print("Predicted Total exp reward", float(r[0,0])) #these two quantities should be approximately the same
    print('##########################')


    #update data gathered
    X = np.vstack((X, X_new))
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))
