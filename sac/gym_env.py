from typing import Any

import numpy as np
import gym
from gym import spaces

from general.mesh import Mesh
from general.geometry import Quad, Vertex, Lin_Alg
from general.plotting import render_boundary, close_render


class Sac_Env(gym.Env):
    def __init__(self, boundary):
        self.mesh = Mesh(boundary)
        self.boundary = boundary.copy()
        self.current_area = self.mesh.original_area

        self.neighbor_num = 6
        self.radius_num = 3
        self.radius = 4

        self.current_ref_state: dict = None
        self.not_valid_points = []
        self.last_not_valid_points = []
        self.target_angle = 0

        self.action_space = spaces.Box(np.array([-1, -1.5, 0]), np.array([1, 1.5, 1.5]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-999, high=999,
            shape=(2 * (self.neighbor_num + self.radius_num), ),
            dtype=np.float32)

        self.estimated_area_range: Any = None
        self.history_info = {
            -1: [],
            1: [],
            0: []
        }

    def seed(self, seed=None):
        return [seed]

    def reset(self, static=False):
        self.mesh.reset()
        self.boundary = self.mesh.boundary.copy()
        self.not_valid_points = []
        self.current_area = self.mesh.original_area
        self.current_ref_state = None
        self.failed_num = 0
        self.estimated_area_range = self.mesh.boundary.estimate_area_range()
        return self.find_next_state(static=static)

    def find_next_state(self, not_valid_points=None, static=False):
        r_p = self.boundary.find_reference_point(not_valid_points, target_angle=self.target_angle)

        if r_p:
            self.current_ref_state = self.boundary.reference_state(
                r_p, self.neighbor_num, self.radius_num,
                self.current_area / self.mesh.original_area, self.radius, static)
            return np.array(self.current_ref_state['state']).astype(np.float32)
        else:
            return None

    def get_middle_points(self, state):
        v1 = np.asarray([state[self.neighbor_num], state[self.neighbor_num + 1]], dtype=float)
        v2 = np.asarray([state[self.neighbor_num + 2], state[self.neighbor_num + 3]], dtype=float)
        return v1, v2

    def detransformation(self, point, is_move=False):
        v1, v2 = self.get_middle_points(Vertex.flatten(self.current_ref_state['neighbors']))

        de_point = Lin_Alg.detransformation(point, self.current_ref_state['base_length'] if not is_move else 1, v1, v2)
        return Vertex(round(de_point[0], 4), round(de_point[1], 4))

    def close(self):
        close_render()

    def render(self, mode='human'):
        render_boundary(self.mesh.boundary)

    def step(self, action):
        done = False
        failed = True
        reward = 0

        rule_type = action[0]
        rule = None
        new_point = self.detransformation([round(action[1], 4), round(action[2], 4)])
        reference_point = self.current_ref_state['reference_point']
        index = self.boundary.vertices.index(reference_point)

        quad = None
        if len(self.boundary.vertices) <= 5:
            reward = 10
            done = True
        elif rule_type <= -0.5:
            quad = self.boundary.rule_element(-1, index)
            rule = -1
        elif rule_type >= 0.5:
            quad = self.boundary.rule_element(1, index)
            rule = 1
        else:
            if self.boundary.contains_point(new_point):
                quad = self.boundary.rule_element(-1, index) if self.boundary.find_same_point(new_point) \
                    else self.boundary.rule_element(0, index, new_point)
            else:
                n = len(self.mesh.generated_quads)
                reward += -1 / n if n else -1
                quad = None
            rule = 0

        valid = quad is not None and self.mesh.can_commit_quad(self.boundary, quad, reference_point)

        if valid:
            self.mesh.commit_quad(self.boundary, quad, reference_point)
            quad_area = quad.area()
            self.current_area -= quad_area

            quality = self.mesh.get_quality(self.boundary, quad, 2)

            min_area = self.estimated_area_range[0] ** 2
            critical_area = self.estimated_area_range[1] ** 2
            if min_area <= quad_area < critical_area:
                speed_penalty = (quad_area - critical_area) / (critical_area - min_area)
            elif quad_area < min_area:
                speed_penalty = -1
            else:
                speed_penalty = 0
            reward += quality + speed_penalty

            self.history_info[rule].append(reward)

            failed = False
            if len(self.boundary.vertices) <= 5:
                reward += 10
                done = True
                if len(self.boundary.vertices) == 4:
                    quad = Quad(self.boundary.vertices)
                    quad.connect_vertices()
                    self.mesh.generated_quads.append(quad)
            else:
                done = False
        elif quad is not None:
            n = len(self.mesh.generated_quads)
            reward += -1 / n if n else -1

        is_complete = True
        next_state = self.find_next_state(self.not_valid_points)

        if not failed:
            self.failed_num = 0
        else:
            self.failed_num += 1
            if self.failed_num >= 100:
                done = True
                is_complete = False
        return next_state, np.float64(reward), done, {'is_complete': is_complete}
