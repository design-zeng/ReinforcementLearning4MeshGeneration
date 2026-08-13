from typing import Any

import numpy as np
import gym
from gym import spaces

from general.geometry import Quad
from general.mesh_env import MeshEnv


class Gym_Env(MeshEnv, gym.Env):
    def __init__(self, boundary):
        super().__init__(boundary)
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
        state = super().reset(static=static)
        self.failed_num = 0
        self.estimated_area_range = self.mesh.boundary.estimate_area_range()
        return state

    def step(self, action):
        done = False
        failed = True
        reward = 0

        rule_type = action[0]
        rule = None
        new_point = self.detransformation([round(action[1], 4), round(action[2], 4)])
        reference_point = self.current_ref_state.reference_point
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
