import numpy as np
from autograd import jacobian


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def earth_to_body_frame(ii, jj, kk):
    # C^b_n
    R = [[C(kk) * C(jj), C(kk) * S(jj) * S(ii) - S(kk) * C(ii), C(kk) * S(jj) * C(ii) + S(kk) * S(ii)],
         [S(kk) * C(jj), S(kk) * S(jj) * S(ii) + C(kk) * C(ii), S(kk) * S(jj) * C(ii) - C(kk) * S(ii)],
         [-S(jj), C(jj) * S(ii), C(jj) * C(ii)]]
    return np.array(R)

# Pick wind speed function.
def wind(x, y, z):
    v = [
        0.02*(np.cos(y)), # + 0.001* np.random.rand(),
        0.02*(np.cos(z)),
        0.02*(np.cos(x)),
    ]
    return np.array(v)

def body_to_earth_frame(ii, jj, kk):
    # C^n_b
    return np.transpose(earth_to_body_frame(ii, jj, kk))

class Quad(object):

    # num_states = 22
    # num_actions = 4

    def __init__(self, init_state):
        self.init_pos = init_state[:3]
        self.init_vel = init_state[3:]

        self.gravity = -9.81  # m/s
        self.rho = 1.2
        self.mass = 0.044  # 300 g
        self.time_step = 1 / 50.0  # Timestep
        self.C_d = 0.3
        self.k1 = np.array([0.02, 0.02, 0.02])
        width, length, height = .15, .15, .03
        self.areas = np.array([length * height, width * height, width * length])

        env_bounds = 300.0  # 300 m / 300 m / 300 m
        self.lower_bounds = np.array([-env_bounds / 2., -env_bounds / 2, 0])
        self.upper_bounds = np.array([env_bounds / 2, env_bounds / 2, env_bounds])
        self.reset()

    def reset(self):
        # self.time = 0.0
        self.pos = self.init_pos
        self.att = np.array([0.0, 0.0, 0.0])
        self.vel = self.init_vel
        self.acc = np.array([0.0, 0.0, 0.0])
        self.done = False


    def get_linear_forces(self, thrust):
        # # Gravity
        # gravity_force = self.mass * self.gravity * np.array([0, 0, 1])
        # # Thrust
        # thrust_body_force = np.array([0, 0, thrust])
        # # Drag
        # body_velocity = np.matmul(earth_to_body_frame(*list(self.att)), self.vel)
        # drag_body_force = - 0.5 * self.rho * body_velocity ** 2 * self.areas * self.C_d
        # # drag_body_force = self.k1 * np.square(body_velocity)
        # body_forces = thrust_body_force + drag_body_force
        #
        # linear_forces = gravity_force + np.matmul(body_to_earth_frame(*list(self.att)), body_forces)


        # another description
        gravity_force = self.mass * self.gravity * np.array([0, 0, 1])
        thrust_body_force = np.array([0, 0, thrust])
        wind_force = wind(self.pos[0], self.pos[1], self.pos[2])
        linear_forces = gravity_force + np.matmul(body_to_earth_frame(*list(self.att)), thrust_body_force) #- self.k1 * np.square(self.vel) # + wind_force

        return linear_forces

    def f(self, state, action):
        # state
        self.pos = state[:3]
        self.vel = state[3:]
        # control
        thrust = action[3]
        self.att = (action[:3] + 2 * np.pi) % (2 * np.pi)

        self.acc = self.get_linear_forces(thrust) / self.mass

        return np.concatenate((self.vel, self.acc))

    def step(self, state, action):
        k1 = self.f(state, action) * self.time_step
        k2 = self.f(state + k1/2.0, action) * self.time_step
        k3 = self.f(state + k2/2.0, action) * self.time_step
        k4 = self.f(state + k3, action) * self.time_step
        next_state = state + (k1 + 2.0 * (k2 + k3) + k4)/6.0
        
        # # state
        # self.pos = state[:3]
        # self.vel = state[3:]
        # # control
        # thrust = action[3]
        # self.att = (action[:3] + 2 * np.pi) % (2 * np.pi)

        # self.acc = self.get_linear_forces(thrust) / self.mass
        # position = self.pos + self.vel * self.time_step + 0.5 * self.acc * self.time_step ** 2
        # self.vel = self.vel + self.acc * self.time_step

        # next_state = np.array(list(position) + list(self.vel))
        return next_state

