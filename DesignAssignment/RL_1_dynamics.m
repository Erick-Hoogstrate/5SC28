%% Motion dynamics of the Cart-Pole systems
%
% Based on a script by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%

%% Explanation of the Cart-Pole dynamics:
%
% Definition of the ODE for simulating the cart-pole dynamics.
%
%    function dz = Sol_3_dynamics(t, z, f)
%
%
% *Input arguments:*
%
%   t     current time step (called from the ODE solver)
%   z     state                                                    [4 x 1]
%   f     input (optional) = force f(t)
%
% *Output arguments:*
%
%   dz    if 3 input arguments:      state derivative w.r.t time
%         if only 2 input arguments: total mechanical energy [only used for verifcation not for RL]
%
%
% Note: It is assumed that the state variables are of the following order:
%       x:        [m]     position of cart
%       dx:       [m/s]   velocity of cart
%       dtheta:   [rad/s] angular velocity
%       theta:    [rad]   angle
%
%
% A detailed derivation of the dynamics can be found in:
%
% M.P. Deisenroth:
% Efficient Reinforcement Learning Using Gaussian Processes, Appendix C,
% KIT Scientific Publishing, 2010.

function dz = Sol_3_dynamics(t,z,f)
%% Code

l = 0.5;  % [m]      length of pendulum
m = 0.5;  % [kg]     mass of pendulum
M = 0.5;  % [kg]     mass of cart
b = 0.1;  % [N/m/s]  coefficient of friction between cart and ground
g = 9.82; % [m/s^2]  acceleration of gravity

if nargin==3 
  % State derivative 
  dz = zeros(4,1);
  dz(1) = z(2);
  dz(2) = ( 2*m*l*z(3)^2*sin(z(4)) + 3*m*g*sin(z(4))*cos(z(4)) ...
          + 4*f(t) - 4*b*z(2) )/( 4*(M+m)-3*m*cos(z(4))^2 );
  dz(3) = (-3*m*l*z(3)^2*sin(z(4))*cos(z(4)) - 6*(M+m)*g*sin(z(4)) ...
          - 6*(f(t)-b*z(2))*cos(z(4)) )/( 4*l*(m+M)-3*m*l*cos(z(4))^2 );
  dz(4) = z(3);
else
  dz = (M+m)*z(2)^2/2 + 1/6*m*l^2*z(3)^2 + m*l*(z(2)*z(3)-g)*cos(z(4))/2; % Total mech energy [not used directly by the RL]
end