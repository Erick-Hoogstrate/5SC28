import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym import spaces
from scipy.integrate import solve_ivp


class SwingUpCartPoleEnv(gym.Env):
    def __init__(self, umax=10, dt=0.1):
        self.l = 0.5
        self.m = 0.5
        self.M = 0.5
        self.b = 0.1
        self.g = 9.82
        self.umax = umax
        self.xmax = 3
        self.action_space = spaces.Box(low=np.array([-self.umax],dtype=np.float32),\
                                       high=np.array([self.umax],dtype=np.float32))
        low = np.array([-float('inf'), -float('inf'), -float('inf'), -float('inf')],dtype=np.float32)
        high = np.array([float('inf'), float('inf'), float('inf'), float('inf')],dtype=np.float32)
        self.observation_space = spaces.Box(low=high,\
                                            high=high)
        self.dt = dt
        self.viewer = None
        self.reset()

    def reset(self):
        self.x, self.dx, self.dtheta, self.theta = 0., 0., 0., 0.
        return self.get_obs()

    @property
    def state(self):
        return self.x, self.dx, self.dtheta, self.theta

    def get_obs(self):
        return np.array(self.state)

    def step(self, action):
        z = self.state
        deriv = lambda t, y: self.deriv(y,action[0])

        sol = solve_ivp(deriv, [0,self.dt], z)
                # sol = solve_ivp(f,[0,self.dt],[self.th,self.omega]) #integration
        self.x, self.dx, self.dtheta, self.theta = sol.y[:,-1]

        done = False
        obs = self.get_obs()
        info = {}
        reward = np.exp(-((self.theta%(2*np.pi))-np.pi)**2/(2*0.5**2))
        return obs, reward, done, info

    def deriv(self, z, u):
        #matlab code
        # l = 0.5;  % [m]      length of pendulum
        # m = 0.5;  % [kg]     mass of pendulum
        # M = 0.5;  % [kg]     mass of cart
        # b = 0.1;  % [N/m/s]  coefficient of friction between cart and ground
        # g = 9.82; % [m/s^2]  acceleration of gravity

        # x = z(1);
        # dx = z(2);
        # dtheta = z(3);
        # theta = z(4);

        # if nargin==3 
        #   % State derivative 

        #   dz = zeros(4,1);
        #   dz(1) = dx;
        #   dz(2) = (2*m*l*dtheta^2*sin(dtheta) + 3*m*g*sin(dtheta)*cos(dtheta) ...
        #           + 4*f(t) - 4*b*dx )/( 4*(M+m)-3*m*cos(dtheta)^2 );
        #   dz(3) = (-3*m*l*dtheta^2*sin(dtheta)*cos(dtheta) - 6*(M+m)*g*sin(dtheta) ...
        #           - 6*(f(t)-b*dx)*cos(dtheta) )/( 4*l*(m+M)-3*m*l*cos(dtheta)^2 );
        #   dz(4) = dtheta;
        l, m, M, b, g = self.l, self.m, self.M, self.b, self.g
        x, dx, dtheta, theta = z
        dxdt     = dx;
        ddxdt    = (2*m*l*dtheta**2*np.sin(theta) + 3*m*g*np.sin(theta)*np.cos(theta) \
                  + 4*u - 4*b*dx )/( 4*(M+m)-3*m*np.cos(theta)**2 );
        ddthetadt = (-3*m*l*dtheta**2*np.sin(theta)*np.cos(theta) - 6*(M+m)*g*np.sin(theta) \
                - 6*(u-b*dx)*np.cos(theta) )/( 4*l*(m+M)-3*m*l*np.cos(theta)**2 );
        dthetadt = dtheta;
        return dxdt, ddxdt, ddthetadt, dthetadt

    def render(self, mode='human'):

        #yreal is from -1.4 to 1.4 = 2.8
        #xreal is from -3 to 3 = 6

        world_height = 1.4 * 2
        world_width = self.xmax * 2 #(-2,2)

        screen_width = 600
        scale = screen_width/world_width #600/4 = 150
        screen_height = int(scale*world_height)

        carty = screen_height/2 # TOP OF CART
        polewidth = 10.0
        polelen = 2 * self.l * scale #two times the length to get the true length
        cartwidth = 0.3* scale
        cartheight = 0.1* scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            target = rendering.make_circle(radius=polewidth)
            target.add_attr(rendering.Transform(translation=(screen_width/2,screen_height/2 + polelen)))
            target.set_color(.2, .8, .2)
            self.viewer.add_geom(target)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x, dx, dtheta, theta = self.state #x, dx, dtheta, theta
        cartx = x * scale + screen_width / 2.0 
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta+np.pi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class SwingUpCartPoleEnv_sincos(SwingUpCartPoleEnv):
    def __init__(self, umax=10, dt=0.1):
        super(SwingUpCartPoleEnv_sincos, self).__init__(umax=umax,dt=dt)
        low = np.array([-float('inf'), -float('inf'), -float('inf'), -1, -1],dtype=np.float32)
        high = np.array([float('inf'), float('inf'), float('inf'), 1, 1,],dtype=np.float32)
        self.observation_space = spaces.Box(low=high,\
                                            high=high)
    def get_obs(self):
        x, dx, dth, th = self.state
        return np.array([x, dx, dth, np.sin(th), np.cos(th)])

if __name__ == '__main__':
    env = SwingUpCartPoleEnv(dt=0.1)
    import time
    # print(env.step([1]))
    try:
        obs = env.reset()
        print(-1,obs)
        for i in range(100):
            action = [4*np.sign(np.cos(env.dt*i*3))] #np.random.uniform(low=-3,high=3,size=(1,))
            obs, reward, done, info = env.step(action)
            print(i,obs)
            env.render()
            time.sleep(1/60)
            print(obs)
    finally:
        env.close()
