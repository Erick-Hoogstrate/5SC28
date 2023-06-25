%% dynamics_pendulum.m
% *Summary:* Implements ths ODE for simulating the pendulum dynamics, where 
% an input torque f can be applied  
%
%    function dz = dynamics_pendulum(t,z,u)
%
%
% *Input arguments:*
%
%		t     current time step (called from ODE solver)
%   z     state                                                    [2 x 1]
%   u     (optional): torque f(t) applied to pendulum
%
% *Output arguments:*
%   
%   dz    if 3 input arguments:      state derivative wrt time
%
%   Note: It is assumed that the state variables are of the following order:
%         dtheta:  [rad/s] angular velocity of pendulum
%         theta:   [rad]   angle of pendulum
%
% A detailed derivation of the dynamics can be found in:
%
% M.P. Deisenroth: 
% Efficient Reinforcement Learning Using Gaussian Processes, Appendix C, 
% KIT Scientific Publishing, 2010.
%
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-18

function dz = dynamics_pendulum(t,z,u)
%% Code

omega0 = 11.339846957335382;
%%delta_th = 0.;
gamma = 1.3328339309394384;
Ku = 28.136158407237073;
Fc = 6.062729509386865;
coulomb_omega = 0.001;

dz = zeros(2,1);
dz(1) = (-(omega0)^2)*sin(z(2))-gamma*z(1)-Fc*tanh(z(1)/coulomb_omega)+Ku*u(t);
dz(2) = z(1);