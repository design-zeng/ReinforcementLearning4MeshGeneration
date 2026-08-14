import numpy as np
import numpy.typing as npt
import gymnasium as gym

from sac_fast.boundary import Boundary
from sac_fast.mesh import Mesh


class Gym_Env(gym.Env):
    def __init__(self, initial_boundary: npt.NDArray[np.floating]):
        super().__init__()

        self.initial_boundary = initial_boundary.copy()
        self.boundary_env = Boundary(self.initial_boundary.copy())
        self.mesh = Mesh(self.boundary_env)

        self.action_space = gym.spaces.Box(
            low=np.array([-1, 0, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32)

        self.n_fails = 0

        self.n_steps = 0
        self.max_steps = 1000

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.boundary_env.reset_with_boundary(self.initial_boundary.copy())
        self.mesh = Mesh(self.boundary_env)

        self.n_fails = 0
        self.n_steps = 0

        observation = self.boundary_env.get_state()

        return observation, {}

    def step(self, action):
        terminated = False
        truncated = False

        if action[0] <= -0.5:
            action_type = -1
        elif action[0] >= 0.5:
            action_type = 1
        else:
            action_type = 0

        polar_pair = (action[1], action[2])

        reward, is_valid = self.boundary_env.get_reward(action_type, polar_pair)

        if is_valid:
            self.mesh.update_mesh(action_type, polar_pair)
            self.boundary_env.update_boundary(action_type, polar_pair)
            self.n_fails = 0
        else:
            self.n_fails += 1

        if self.n_fails > 100:
            terminated = True

        self.n_steps += 1

        if self.n_steps >= self.max_steps:
            truncated = True

        if self.boundary_env.is_quad_left():
            reward = 10.0
            terminated = True

        observation = self.boundary_env.get_state()

        return observation, reward, terminated, truncated, {}
