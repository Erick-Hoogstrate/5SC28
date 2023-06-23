clear; close all; clc;
load_api;

%% Constants and definitions
Ts = 0.025;                                     % Sample Time
Tf = 10;                                        % Experiment duration [s]
Nlength = Tf/Ts;                                % Experiment length (number of samples)
umax = 3;                                       % Max voltage
t = (0:Ts:Ts*(Nlength-1))';


% Keep this part the same to ensure that the input signal does not exceed umax
u(u>umax) = umax;                               % Clip input signal
u(-umax>u) = -umax;
% plot(t,u)

%% History variables
data = zeros(9,length(t));

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

%% Real-time loop
tic;                                            % Reset Matlab's tic-toc timer
for k = 1 : Nlength                                   % RT loop                                                        
    if MOPSconnected
        fugiboard('Write',Board,0,1,u(k),0);        % Send control input to process
        MOPS_sensors = fugiboard('Read',Board);     % Read sensor data
        data(:,k) = MOPS_sensors;
    else
        y(k) = 0;
    end

    td(k) = toc;                                % Record time spent on the calculations
    while toc < Ts; end                         % Wait for tick
    tic;                                        % Reset Matlab's tic-toc timer
end

%% Close real-time API
if MOPSconnected
    fugiboard('Write', Board, 0, 1, 0, 0.0);        % Reset actuator
    fugiboard('CloseAll');                      % Close the port
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
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    pause(dt);
  end
end



%% Plot Data
figure()
subplot(3,1,1)
plot(t,u);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Control Input $(u)$ [V]','Interpreter','latex');grid on;
subplot(3,1,2);
plot(t,data(3,:));xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angle $(\theta)$ [rad]','Interpreter','latex');grid on;
subplot(3,1,3);
plot(t,data(4,:));xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angular Velocity $(\omega)$ [rad/s]','Interpreter','latex');grid on;
