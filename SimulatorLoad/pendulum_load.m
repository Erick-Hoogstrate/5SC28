% 1. Initialization
clear all; close all; clc;
load("pendulum_23_H40.mat")
load('parameters.mat')


dt = 0.025;                      % [s] sampling time
% T = 4;                         % [s] prediction time
% H = ceil(T/dt);                % prediction steps (optimization horizon)
% mu0 = [0 0]';                  % initial state mean
% S0 = 0.01*eye(2);              % initial state variance

omega = 0;
theta = 0;

umax = 3;

while true

    s = [omega sin(theta) cos(theta)]';
    u = policy.fcn(policy,s,zeros(length(poli)));

    u(u>umax) = umax;                               % Clip input signal
    u(-umax>u) = -umax;
end


%%
% %% Plot Data
% figure()
% subplot(3,1,1)
% plot(t,u);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Control Input $(u)$ [V]','Interpreter','latex');grid on;
% subplot(3,1,2);
% plot(t,data(3,:));xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angle $(\theta)$ [rad]','Interpreter','latex');grid on;
% subplot(3,1,3);
% plot(t,data(4,:));xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angular Velocity $(\omega)$ [rad/s]','Interpreter','latex');grid on;