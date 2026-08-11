import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import gym
from gym import spaces

from general.mesh import Mesher
from general.components import Vertex, Quad
from general.point_environment import PointEnvironment
from general.math_utils import detransformation
from general.plotting import render_boundary, close_render


base_path = Path(__file__).parent.parent
output_path = base_path / "general" / "output"


class MeshEnv(gym.Env):
    TYPE_THRESHOLD = 0.3

    def __init__(self, boundary):
        self.mesher = Mesher(boundary)
        self.original_boundary = self.mesher.boundary.deep_copy()
        self.original_area = self.mesher.boundary.poly_area()
        self.current_area = self.original_area
        self.max_radius = 2
        self.action_space = spaces.Box(np.array([-1, -1.5, 0]), np.array([1, 1.5, 1.5]), dtype=np.float32)

        self.neighbor_num = 6
        self.radius_num = 3
        self.radius = 4
        self.observation_space = spaces.Box(
            low=-999, high=999,
            shape=(2 * (self.neighbor_num + self.radius_num), ),
            dtype=np.float32)
        self.current_point_environment: Any = None
        self.not_valid_points = []
        self.last_not_valid_points = []

        self.target_angle = 0
        self.estimated_area_range: Any = None

        self.history_info = {
            -1: [],
            1: [],
            0: []
        }

    def __getattr__(self, name):
        # Thin gym adapter: anything not defined on the env is delegated to the
        # composed Mesher (boundary state, generated_quads, smoothing, quality, ...).
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        mesher = self.__dict__.get('mesher')
        if mesher is not None:
            return getattr(mesher, name)
        raise AttributeError(name)

    def reset(self, static=False):  # pyright: ignore[reportIncompatibleMethodOverride]
        self.mesher.boundary = self.original_boundary.deep_copy()
        self.mesher.updated_boundary = self.mesher.boundary.copy()
        self.mesher.original_vertices = list(self.mesher.boundary.vertices)
        self.mesher.generated_quads = []
        self.not_valid_points = []
        self.current_area = self.original_area
        self.current_point_environment = None
        self.failed_num = 0
        state = self.find_next_state(static=static)
        self.estimated_area_range = self.mesher.boundary.estimate_area_range()
        return state

    def get_middle_points(self, state):
        v1 = np.asarray([state[self.neighbor_num], state[self.neighbor_num + 1]], dtype=float)
        v2 = np.asarray([state[self.neighbor_num + 2], state[self.neighbor_num + 3]], dtype=float)
        return v1, v2

    def detransformation(self, point, is_move=False):
        v1, v2 = self.get_middle_points(Vertex.points_as_array(self.current_point_environment.neighbors))

        de_point = detransformation(point, self.current_point_environment.base_length if not is_move else 1, v1, v2)
        return Vertex(round(de_point[0], 4), round(de_point[1], 4))

    def step(self, action):  # pyright: ignore[reportIncompatibleMethodOverride]
        done = False
        failed = True
        reward = 0

        rule_type = action[0]
        rule = None
        new_point = self.action_2_point(action[1:])
        reference_point = self.current_point_environment.reference_point
        index = self.updated_boundary.vertices.index(reference_point)

        if len(self.updated_boundary.vertices) <= 5:
            reward = 10
            done = True
        else:
            if rule_type <= -0.5: #self.TYPE_THRESHOLD:
                quad = self.updated_boundary.rule_element(-1, index)
                rule = -1
            elif rule_type >= 0.5: #1 - self.TYPE_THRESHOLD:
                quad = self.updated_boundary.rule_element(1, index)
                rule = 1
            else:
                # reward -= 0.1 * math.fabs(rule_type)
                if self.updated_boundary.contains_point(new_point):
                    quad = self.updated_boundary.rule_element(-1, index) if self.updated_boundary.find_same_point(new_point) \
                        else self.updated_boundary.rule_element(0, index, new_point)
                else:
                    reward += self.failure_penalty()
                    quad = None
                rule = 0

            if quad is not None:
                if quad.is_valid(0) and \
                        not self.updated_boundary.check_intersection_with_boundary(quad, reference_point): # intersection check remove for type 1 2
                    quad.connect_vertices()
                    self.generated_quads.append(quad)

                    self.updated_boundary.update_boundary(reference_point, quad, self.boundary)
                    quad_area = quad.area()
                    self.current_area -= quad_area

                    quality = self.get_quality(quad, 2)

                    speed_penalty = self.get_speed_penalty(quad_area)
                    reward += quality + speed_penalty

                    self.history_info[rule].append(reward)

                    failed = False
                    if len(self.updated_boundary.vertices) <= 5:
                        reward += 10
                        done = True
                        if len(self.updated_boundary.vertices) == 4:
                            quad = Quad(self.updated_boundary.vertices)
                            quad.connect_vertices()
                            self.generated_quads.append(quad)
                    else:
                        done = False
                else:
                    reward += self.failure_penalty()
        is_complete = True
        next_state = self.find_next_state(self.not_valid_points)
        # if next_state is None:
        #     self.not_valid_points = []
        #     next_state = self.find_next_state(self.not_valid_points)
        #     done = True

        if not failed:
            self.failed_num = 0
        else:
            self.failed_num += 1
            if self.failed_num >= 100: # self.action_space.shape[0] * 40
                done = True
                is_complete = False
        return next_state, np.float64(reward), done, {'is_complete': is_complete}

    def move(self, new_point, rule_type, lr_1=None, lr_2=None):
        x = self.current_point_environment.base_length * self.radius * new_point[0] * math.cos(new_point[1])
        y = self.current_point_environment.base_length * self.radius * new_point[0] * math.sin(new_point[1])

        new_point = self.detransformation([round(x, 6), round(y, 6)], is_move=True)

        done = False
        next_state = None
        not_valid_element = True
        is_complete = True

        reference_point = self.current_point_environment.reference_point

        if len(self.updated_boundary.vertices) <= 5:
            reward = 10
            done = True

        else:
            index = self.updated_boundary.vertices.index(reference_point)
            quad = None

            if rule_type <= self.TYPE_THRESHOLD:
                quad = self.updated_boundary.rule_element(-1, index)
                # reward -= 0.1 * math.fabs(rule_type + 1)
            elif rule_type >= 1 - self.TYPE_THRESHOLD:
                quad = self.updated_boundary.rule_element(1, index)
            else:
                if self.updated_boundary.contains_point(new_point):
                    quad = self.updated_boundary.rule_element(0, index, new_point)

            if quad is None:
                pass
            elif quad.is_valid(0) and \
                not self.updated_boundary.check_intersection_with_boundary(quad, reference_point):

                quad.connect_vertices()
                not_valid_element = False
                self.generated_quads.append(quad)

                self.updated_boundary.update_boundary(reference_point, quad, self.boundary)

                next_state = self.find_next_state(self.not_valid_points, static=True)

                if len(self.updated_boundary.vertices) <= 5:
                    done = True
                    if len(self.updated_boundary.vertices) == 4:
                        quad = Quad(self.updated_boundary.vertices)
                        self.generated_quads.append(quad)

            ## old handling
            if not_valid_element:
                if reference_point not in self.not_valid_points:
                    self.not_valid_points.append(reference_point)
                next_state = self.find_next_state(self.not_valid_points, static=True)
                # if next_state is None:
                # done = True
            else:
                self.not_valid_points = []

            # if next_state is None:
            #     done = True
            # if len(self.updated_boundary.vertices) > 4:
            #     is_complete = False
            # else:
            #     is_complete = True
            #     done = True

            if len(self.updated_boundary.vertices) > 4:
                is_complete = False
                if next_state is None:
                    if lr_1 and lr_2:
                        self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices, lr_1, lr_2, iteration=400)
                        # self.smooth(self.boundary.vertices, lr_1, lr_2, iteration=500)
                    else:
                        self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices, iteration=400)

                    if len(self.last_not_valid_points) > 0 and len(self.not_valid_points) > 0:
                        if self.last_not_valid_points[0] == self.not_valid_points[0] and \
                            self.last_not_valid_points[-1] == self.not_valid_points[-1] and \
                                len(self.not_valid_points) == len(self.last_not_valid_points):
                            done = True

                    self.last_not_valid_points = self.not_valid_points
                    self.not_valid_points = []

                    next_state = self.find_next_state(not_valid_points=self.not_valid_points, static=True)
                    if next_state is None:
                        done = True

            else:
                is_complete = True

        return next_state, 0, done, {'is_complete': is_complete}

    def failure_penalty(self):
        n = len(self.generated_quads)
        return -1 / n if n else -1

    def get_speed_penalty(self, quad_area):
        min_area = self.estimated_area_range[0] ** 2
        critical_area = self.estimated_area_range[1] ** 2

        if min_area <= quad_area < critical_area:
            speed_penalty = ((quad_area - critical_area) / (critical_area - min_area))
        elif quad_area < min_area:
            speed_penalty = -1
        else:
            speed_penalty = 0
        return speed_penalty

    def find_next_state(self, not_valid_points=None, static=False):
        r_p = self.updated_boundary.find_reference_point(not_valid_points, target_angle=self.target_angle)

        if r_p:
            p_e = PointEnvironment(reference_point=r_p, boundary=self.updated_boundary,
                                   neighbor_num=self.neighbor_num, radius_num=self.radius_num,
                                   area_ratio=self.current_area / self.original_area,
                                   radius=self.radius, static=static)
            self.current_point_environment = p_e

            state = p_e.state
            return np.array(state).astype(np.float32)

        else:
            # self.last_point_environment = None
            return None

    def seed(self, seed=None):
        # Old-gym seeding hook; kept so SB3's Gym->Gymnasium compatibility layer
        # (which calls env.seed(seed) on reset) works. Meshing is deterministic
        # given the boundary, so there is no RNG state to set here.
        return [seed]

    def close(self):
        close_render()

    def render(self, mode='human'):
        print(f'Generated elements: {len(self.generated_quads)}')
        render_boundary(self.boundary)

    def action_2_point(self, action):
        if isinstance(action, np.ndarray):
            x, y = action[0], action[1]
        else:
            n_action = [action / (self.max_radius * 10), (action % (self.max_radius * 10)) / 10]
            x = round(math.cos(math.radians(n_action[0])), 1) * n_action[1]
            y = round(math.sin(math.radians(n_action[0])), 1) * n_action[1]
        return self.detransformation([round(x, 4), round(y, 4)])

    def save_history_info(self, filename):
        with open(filename, 'w') as fw:
            json.dump(self.history_info, fw)
