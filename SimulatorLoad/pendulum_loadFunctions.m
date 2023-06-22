% 1. Initialization
clear all; close all; clc;
load("pendulum_23_H40.mat")
% load("pendulum_20_H16.mat")
load('parameters.mat')


dt = 0.025;                      % [s] sampling time
Runtime = 10;                     % [s] prediction time
time = ceil(T/dt);                % prediction steps (optimization horizon)
umax = 3;
omega = 0;
theta = 0;
u = 0;



state = init_state_disk();

for i = 1:time
    thetas(i) = state.theta;
    us(i) = u;
    omegas(i) = state.omega;
    
    s = [omega sin(theta) cos(theta)]';
    u = policy.fcn(policy,s,zeros(length(poli)));

    u(u>umax) = umax;                               % Clip input signal
    u(-umax>u) = -umax;
    state = unbalanced_disk(state, u, dt);
end
plot(thetas)
% title(num2str(max(thetas)))



%%
%% Plot Data
figure()
subplot(3,1,1)
plot(us);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Control Input $(u)$ [V]','Interpreter','latex');grid on;
subplot(3,1,2);
plot(thetas);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angle $(\theta)$ [rad]','Interpreter','latex');grid on;
subplot(3,1,3);
plot(omegas);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angular Velocity $(\omega)$ [rad/s]','Interpreter','latex');grid on;