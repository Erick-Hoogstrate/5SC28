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


function dz = dynamics_pendulum(t,z,u)

dz = zeros(2,1);
dz(1) = (-(omega0)^2)*sin(z(2))-gamma*z(1)-Fc*tanh(z(1)/coulomb_omega)+Ku*u(t);
dz(2) = z(1);
