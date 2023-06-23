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
dt = 0.025;                                     % Sample Time
umax = 3;                                       % Max voltage


% Keep this part the same to ensure that the input signal does not exceed umax
u(u>umax) = umax;                               % Clip input signal
u(-umax>u) = -umax;
% plot(t,u)


%% Real-time loop
tic;                                            % Reset Matlab's tic-toc timer
                                                    
fugiboard('Write',Board,0,1,u,0);        % Send control input to process
MOPS_sensors = fugiboard('Read',Board);     % Read sensor data

td(k) = toc;                                % Record time spent on the calculations
while toc < dt; end                         % Wait for tick
tic;                                        % Reset Matlab's tic-toc timer

% MOPS_sensors(3,:) %theta
% MOPS_sensors(4,:) %omega

dz = zeros(2,1);
dz(1) = (MOPS_sensors(4,:) - z.omega) / dt;
% dz(1) = MOPS_sensors(4,:);
dz(2) = MOPS_sensors(3,:);