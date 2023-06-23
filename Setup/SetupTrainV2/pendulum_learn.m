clear; close all; clc;
load_api;


%% Initialize real-time API
try
    fugiboard('CloseAll');                      % Close port
    Board = fugiboard('Open', 'mops1');             % Open port
    Board.WatchdogTimeout = 5;                      % Watchdog timeout
    fugiboard('SetParams', Board);                  % Set the parameter
    fugiboard('Write', Board, 1, 1, 0, 0);          % Dummy write to sync interface board
    fugiboard('Write', Board, 1, 1, 0, 0);          % Reset position
    fugiboard('Write', Board, 0, 1, 0, 0);          % End reset
    for i = 1 : 10
        tmp = fugiboard('Read',Board);              % Dummy read sensor data                
    end
    MOPSconnected = 1;
catch
    error('Warning: the setup is not connected to the computer.');
    MOPSconnected = 0;
end



% 1. Initialization
settings_pendulum;            % load scenario-specific settings
basename = 'pendulumSetup_';  % filename used for saving data

% 2. Initial J random rollouts
for jj = 1:J
  [xx, yy, realCost{jj}, latent{jj}] = ...
    rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost);
  x = [x; xx]; y = [y; yy];       % augment training sets for dynamics model
end

mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

% 3. Controlled learning (N iterations)
for j = 1:N
  trainDynModel;   % train (GP) dynamics model
  learnPolicy;     % learn policy
  applyController; % apply controller to system
  disp(['controlled trial # ' num2str(j)]);
end




%% Close real-time API
if MOPSconnected
    fugiboard('Write', Board, 0, 1, 0, 0.0);        % Reset actuator
    fugiboard('CloseAll');                      % Close the port
end
