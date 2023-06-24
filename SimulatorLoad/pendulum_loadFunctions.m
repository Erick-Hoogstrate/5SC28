% 1. Initialization
clear all; close all; clc;
% load("pendulum_8_H160.mat")
load("multi_pendulum_p210_H160.mat")

dt = 0.025;                      % [s] sampling time
Runtime = 4;                     % [s] prediction time
time = ceil(Runtime/dt);                % prediction steps (optimization horizon)
umax = 3;
omega = 0;
theta = 0;
u = 0;



state = init_state_disk();

for i = 1:time
    thetas(i) = state.theta;
    us(i) = u;
    omegas(i) = state.omega;
    
    s = [state.omega sin(state.theta) cos(state.theta)]';
    u = policy.fcn(policy,s,zeros(length(poli)));

    state = unbalanced_disk(state, u, dt);
end
% plot(thetas)
% title(num2str(max(thetas)))



%%
%% Plot Data
figure()
subplot(3,1,1)
plot(us);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Control Input $(u)$ [V]','Interpreter','latex');grid on;
ylim([-3.5, 3.5]);  
title(['Running policy: ', filename, '.mat'], 'Interpreter', 'none');
subplot(3,1,2);
plot(thetas);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angle $(\theta)$ [rad]','Interpreter','latex');grid on;
hold on;  % Enable hold on to plot multiple lines on the same axes
line([0, numel(thetas)], [pi, pi], 'Color', 'r', 'LineStyle', '--');
line([0, numel(thetas)], [17/18*pi, 17/18*pi], 'Color', 'r', 'LineStyle', '--');
line([0, numel(thetas)], [19/18*pi, 19/18*pi], 'Color', 'r', 'LineStyle', '--');
line([0, numel(thetas)], [-pi, -pi], 'Color', 'r', 'LineStyle', '--');
ylim([-4, 4]);  
hold off;  % Disable hold on to return to normal plotting behavior
subplot(3,1,3);
plot(omegas);xlabel('Time $(t)$ [s]','Interpreter','latex');ylabel('Angular Velocity $(\omega)$ [rad/s]','Interpreter','latex');grid on;