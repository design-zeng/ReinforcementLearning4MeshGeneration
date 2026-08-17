import numpy as np
import gym
from gym import spaces

from general.config import NEIGHBOR_NUM, RADIUS_NUM, MAX_FAILURES
from general.geometry import Quad, Polygon, Mesh
from general.recognition import reference_state, reference_candidates, next_reference_point, updated_candidates
from general.action import action_point, rule_quad
from general.quality import combined_quality, can_add_quad, speed_penalty
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
            shape=(2 * (NEIGHBOR_NUM + RADIUS_NUM), ),
            dtype=np.float32)

        self.reset()

    def reset(self, static=False):
        fresh = self.problem.deep_copy()
        self.mesh = Mesh(fresh)
        self.boundary = fresh.copy()
        self.candidates = reference_candidates(self.boundary)
        self.current_area = self.mesh.original_area
        self.estimated_area_range = self.boundary.estimate_area_range()
        self.failed_num = 0

        reference_point = next_reference_point(self.candidates)
        if not reference_point:
            return None
        self.current_ref_state = reference_state(
            self.boundary, reference_point, self.current_area / self.mesh.original_area, static)
        return np.array(self.current_ref_state['state']).astype(np.float32)

    def step(self, action):
        done = False
        absorbed = False
        reward = 0

        if len(self.boundary.vertices) <= 5:
            reward = 10
            done = True
        else:
            reference_point = self.current_ref_state['reference_point']
            index = self.boundary.vertices.index(reference_point)
            rule_type = action[0]
            new_point = action_point(self.current_ref_state,
                                     [round(action[1], 4), round(action[2], 4)])

            new_vertex = None
            if rule_type <= -0.5:
                rule = -1
                quad = rule_quad(self.boundary, -1, index)
            elif rule_type >= 0.5:
                rule = 1
                quad = rule_quad(self.boundary, 1, index)
            else:
                rule = 0
                if not self.boundary.contains_point(new_point):
                    quad = None
                elif self.boundary.find_same_point(new_point):
                    quad = rule_quad(self.boundary, -1, index)
                else:
                    quad = rule_quad(self.boundary, 0, index, new_point)
                    new_vertex = new_point

            if quad is not None and can_add_quad(self.boundary, quad, reference_point):
                self.mesh.add_quad(quad)
                retired, rescored = self.boundary.update_boundary(quad)
                self.candidates = updated_candidates(self.candidates, self.boundary, retired, rescored)
                self.current_area -= quad.area()

                reward = combined_quality(self.boundary, quad, new_vertex)
                reward += speed_penalty(quad.area(), self.estimated_area_range)
                self.history_info[rule].append(reward)
                absorbed = True

                if len(self.boundary.vertices) <= 5:
                    reward += 10
                    done = True
                    if len(self.boundary.vertices) == 4:
                        self.mesh.add_quad(Quad(self.boundary.vertices))
            else:
                n = len(self.mesh.generated_quads)
                reward = -1 / n if n else -1

        self.failed_num = 0 if absorbed else self.failed_num + 1
        if self.failed_num >= MAX_FAILURES:
            done = True

        reference_point = next_reference_point(self.candidates)
        if reference_point:
            self.current_ref_state = reference_state(
                self.boundary, reference_point, self.current_area / self.mesh.original_area, False)
            observation = np.array(self.current_ref_state['state']).astype(np.float32)
        else:
            observation = None
        return observation, np.float64(reward), done, {'is_complete': self.failed_num < MAX_FAILURES}

    def seed(self, seed=None):
        return [seed]

    def render(self, mode='human'):
        render_boundary(Polygon(self.mesh.vertices()))

    def close(self):
        close_render()
