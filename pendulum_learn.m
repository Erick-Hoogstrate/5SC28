%% pendulum_learn.m
% *Summary:* Script to learn a controller for the pendulum swingup
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-27
%
%% High-Level Steps
% # Load parameters
% # Create J initial trajectories by applying random controls
% # Controlled learning (train dynamics model, policy learning, policy
% application)

%% Code

% 1. Initialization
clear all; close all;
% settings_pendulum;            % load scenario-specific settings
basename = 'pendulumSetup_';       % filename used for saving data


load_api;

%% Constants and definitions
Ts = 0.025;                                     % Sample Time
Tf = 10;                                        % Experiment duration [s]
N = Tf/Ts;                                      % Experiment length (number of samples)
umax = 3;                                       % Max voltage
t = (0:Ts:Ts*(N-1))';
%% History variables
    data = zeros(9,length(t));

%% Initialize real-time API
try
    fugiboard('CloseAll');                      % Close port
    H = fugiboard('Open', 'mops1');             % Open port
    H.WatchdogTimeout = 5;                      % Watchdog timeout
    fugiboard('SetParams', H);                  % Set the parameter
    fugiboard('Write', H, 1, 1, 0, 0);          % Dummy write to sync interface board
    fugiboard('Write', H, 1, 1, 0, 0);          % Reset position
    fugiboard('Write', H, 0, 1, 0, 0);          % End reset
    for i = 1 : 10
        tmp = fugiboard('Read',H);              % Dummy read sensor data                
    end
    MOPSconnected = 1;
catch
    error('Warning: the setup is not connected to the computer.');
    MOPSconnected = 0;
end

% 2. Initial J random rollouts
for jj = 1:J
  [xx, yy, realCost{jj}, latent{jj}] = ...
    rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost);
  x = [x; xx]; y = [y; yy];       % augment training sets for dynamics model
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_pendulum;
  end
end

mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

% 3. Controlled learning (N iterations)
for j = 1:N
  trainDynModel;   % train (GP) dynamics model
  learnPolicy;     % learn policy
  applyController; % apply controller to system
  disp(['controlled trial # ' num2str(j)]);
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_pendulum;
  end
end

%% Close real-time API
if MOPSconnected
    fugiboard('Write', H, 0, 1, 0, 0.0);        % Reset actuator
    fugiboard('CloseAll');                      % Close the port
end