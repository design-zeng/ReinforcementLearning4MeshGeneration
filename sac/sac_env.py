import numpy as np
import gym
from gym import spaces

from general.config import NEIGHBOR_NUM, FAN_NUM, MAX_FAILURES
from general.geometry import Polygon, Mesh
from general.boundary import Boundary
from general.recognition import recognize
from general.action import act, act_last
from general.quality import reward
from general.plotting import render_boundary, close_render


class Sac_Env(gym.Env):
    def __init__(self, boundary):
        self.problem = boundary.deep_copy()
        self.history_info = {
            -1: [],
            1: [],
            0: []
        }

        self.action_space = spaces.Box(np.array([-1, -1.5, 0]), np.array([1, 1.5, 1.5]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-999, high=999,
            shape=(2 * (NEIGHBOR_NUM + FAN_NUM), ),
            dtype=np.float32)

        self.reset()

    def reset(self, static=False):
        fresh = self.problem.deep_copy()
        self.mesh = Mesh(fresh)
        self.boundary = fresh.copy()
        self.candidates = []
        Boundary.score_candidates(self.boundary, self.candidates)
        self.area_bounds = self.boundary.estimate_area_bounds()
        self.failed_num = 0
        self.view = {}

        recognize(self.boundary, self.candidates, self.view, self.mesh.area_ratio(), static)
        if not self.view['reference_point']:
            return None
        return np.array(self.view['state']).astype(np.float32)

    def step(self, action):
        done = False
        absorbed = False
        r_t = 0

        if len(self.boundary.vertices) <= 5:
            r_t = 10
            done = True
        else:
            outcome = {}
            act(self.view, action, self.boundary, self.mesh, outcome)
            r_t = reward(outcome, self.boundary, self.mesh, self.area_bounds)

            if outcome['absorbed']:
                self.mesh.current_area -= outcome['quad'].area()
                self.history_info[outcome['rule']].append(outcome['measurement'])
                absorbed = True

                if outcome['last']:
                    done = True
                    if len(self.boundary.vertices) == 4:
                        act_last(self.boundary, self.mesh)

        self.failed_num = 0 if absorbed else self.failed_num + 1
        if self.failed_num >= MAX_FAILURES:
            done = True

        recognize(self.boundary, self.candidates, self.view, self.mesh.area_ratio(), False)
        if self.view['reference_point']:
            observation = np.array(self.view['state']).astype(np.float32)
        else:
            observation = None
        return observation, np.float64(r_t), done, {'is_complete': self.failed_num < MAX_FAILURES}

    def seed(self, seed=None):
        return [seed]

    def render(self, mode='human'):
        render_boundary(Polygon(self.mesh.vertices()))

    def close(self):
        close_render()
