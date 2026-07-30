import math
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

from general.mesh import MeshGeneration
from general.components import Vertex, Mesh
from general.point_environment import PointEnvironment
from general.lin_alg import transformation, detransformation
from general.boundary_renderer import MeshFrame

base_path = Path(__file__).parent.parent
output_path = base_path / "general" / "output"

class BoudaryEnv(MeshGeneration, gym.Env):
    TYPE_THRESHOLD = 0.3

    def __init__(self, boundary, experiment_version=None, env_name=None):
        super(BoudaryEnv, self).__init__(boundary)
        self.original_boundary = self.boundary.deep_copy()
        self.original_area = self.boundary.poly_area()
        self.current_area = self.original_area
        self.max_radius = 2 # change from 3 to 2
        self.action_space = spaces.Box(np.array([-1, -1.5, 0]), np.array([1, 1.5, 1.5]), dtype=np.float32)

        self.neighbor_num = 6 # from 4 to 6
        self.radius_num = 3
        self.radius = 4
        self.observation_space = spaces.Box(low=-999, high=999,
                                            shape=(2 * (self.neighbor_num + self.radius_num), ), dtype=np.float32)
        self.current_point_environment: Any = None
        self.not_valid_points = []
        self.last_not_valid_points = []

        self.target_angle = 0
        self.rewarding = []
        self.estimated_area_range: Any = None
        self.window_size = (500, 500)
        self.current_state = None

        # loggging
        self.experiment_version = experiment_version if experiment_version else 'test'
        self.env_name = env_name if env_name is not None else 1
        self.history_info = {
            -1: [],
            1: [],
            0: []
        }

    def reset(self, static=False):  # pyright: ignore[reportIncompatibleMethodOverride]
        self.viewer = None
        self.boundary = self.original_boundary.deep_copy()
        self.updated_boundary = self.boundary.copy()
        self.original_vertices = [v for v in self.boundary.vertices]
        self.generated_meshes = []
        self.not_valid_points = []
        self.rewarding = []
        self.current_area = self.original_area
        self.current_point_environment = None
        self.candidate_vertices = None
        self.test_candidate_vertices = []
        self.failed_num = 0
        state = self.find_next_state(static=static)
        self.current_state = state
        self.estimated_area_range = self.estimate_area_range()
        return state

    def get_middle_points(self, state):
        v1 = np.asarray([state[self.neighbor_num], state[self.neighbor_num + 1]], dtype=float)
        v2 = np.asarray([state[self.neighbor_num + 2], state[self.neighbor_num + 3]], dtype=float)
        return v1, v2

    def transformation(self):
        arra = self.current_point_environment.state
        p0, p1 = self.get_middle_points(arra)

        return transformation(arra, self.current_point_environment.base_length, p0, p1)

    def detransformation(self, point, is_move=False):
        v1, v2 = self.get_middle_points(self.current_point_environment.points_as_array(self.current_point_environment.neighbors))

        de_point = detransformation(point, self.current_point_environment.base_length if not is_move else 1, v1, v2)
        return Vertex(round(de_point[0], 4), round(de_point[1], 4))

    def rule_element(self, rule, index, new_point=None):
        v = self.updated_boundary.vertices
        n = len(v)
        if rule == -1:
            return Mesh([v[index - 1], v[index], v[(index + 1) % n], v[(index + 2) % n]])
        if rule == 1:
            return Mesh([v[index - 2], v[index - 1], v[index], v[(index + 1) % n]])
        return Mesh([new_point, v[index - 1], v[index], v[(index + 1) % n]])

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
                mesh = self.rule_element(-1, index)
                rule = -1
            elif rule_type >= 0.5: #1 - self.TYPE_THRESHOLD:
                mesh = self.rule_element(1, index)
                rule = 1
            else:
                # reward -= 0.1 * math.fabs(rule_type)
                if self.is_point_inside_area(new_point):
                    mesh = self.rule_element(-1, index) if self.find_same_point(new_point) \
                        else self.rule_element(0, index, new_point)
                else:
                    reward += -1 / len(self.generated_meshes) if len(self.generated_meshes) else -1
                    mesh = None
                rule = 0

            if mesh is not None:
                if self.validate_mesh(mesh, quality_method=0) and \
                        not self.check_intersection_with_boundary(mesh, reference_point): # intersection check remove for type 1 2
                    mesh.connect_vertices()
                    self.generated_meshes.append(mesh)

                    # update boundary and reference points
                    # remove_references, add_references = self.update_boundary(reference_point, mesh)

                    self.update_boundary(reference_point, mesh)
                    # self.boundary.show()
                    mesh_area = mesh.compute_area()[0]
                    self.current_area -= mesh_area

                    # if len(self.generated_meshes) % 5 == 0:
                    # self.boundary.save_intermediate_boundary_fig(f"{output_path}/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_boundary.png",
                    #                                              mesh.vertices, style='k.-', dpi=200, r_vertices=self.updated_boundary.vertices)
                    # self.boundary.save_vertices_into_fig(f"{output_path}/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_action.png",
                    #                                              mesh.vertices, style='r.-', dpi=200)
                    # self.updated_boundary.savefig(
                    #     f"{output_path}/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_left_boundary.png",
                    #     style='b.-')
                    # self.boundary.savefig(f"{output_path}/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_boundary.png", style='k.-')

                    quality = self.get_quality(mesh, 2)

                    speed_penalty = self.get_speed_penalty(mesh_area)
                    reward += quality + speed_penalty

                    self.history_info[rule].append(reward)

                    failed = False
                    if len(self.updated_boundary.vertices) <= 5:
                        reward += 10
                        done = True
                        if len(self.updated_boundary.vertices) == 4:
                            mesh = Mesh(self.updated_boundary.vertices)
                            mesh.connect_vertices()
                            self.generated_meshes.append(mesh)
                        # self.boundary.save_intermediate_boundary_fig(
                        #     f"{output_path}/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_boundary.png",
                        #     self.updated_boundary.vertices, style='k.-', dpi=200)
                        # self.boundary.save_vertices_into_fig(
                        #     f"{output_path}/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_action.png",
                        #     self.updated_boundary.vertices, style='r.-', dpi=200)
                    else:
                        done = False
                else:
                    reward += -1 / len(self.generated_meshes) if len(self.generated_meshes) else -1
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

    def move(self, new_point, type, lr_1=None, lr_2=None):
        x, y = self.current_point_environment.base_length * self.radius * new_point[0] * math.cos(new_point[1]), \
               self.current_point_environment.base_length * self.radius * new_point[0] * math.sin(new_point[1])
        # x, y = np.clip(self.radius * new_point[0] * math.cos(new_point[1]), -1.5, 1.5), \
        #        np.clip(self.radius * new_point[0] * math.sin(new_point[1]), -1.5, 1.5)
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
            mesh = None

            if type <= self.TYPE_THRESHOLD:
                mesh = self.rule_element(-1, index)
                # reward -= 0.1 * math.fabs(rule_type + 1)
            elif type >= 1 - self.TYPE_THRESHOLD:
                mesh = self.rule_element(1, index)
            else:
                if self.is_point_inside_area(new_point):
                    mesh = self.rule_element(0, index, new_point)

            if mesh is None:
                pass
            elif self.validate_mesh(mesh, quality_method=0) and \
                not self.check_intersection_with_boundary(mesh, reference_point):

                mesh.connect_vertices()
                not_valid_element = False
                self.generated_meshes.append(mesh)

                # self.boundary.save_intermediate_boundary_fig(f"boundary/{len(self.generated_meshes)}_boundary.png",
                #                                              self.updated_boundary.vertices, style='b.-', dpi=300)

                self.update_boundary(reference_point, mesh)

                # self.boundary.save_intermediate_boundary_fig(f"{output_path}/data_augmentation/test/{len(self.generated_meshes)}_left_boundary.png",
                #                                              mesh.vertices, style='k.-', dpi=400, r_vertices=self.updated_boundary.vertices)

                next_state = self.find_next_state(self.not_valid_points, static=True)

                if len(self.updated_boundary.vertices) <= 5:
                    done = True
                    if len(self.updated_boundary.vertices) == 4:
                        mesh = Mesh(self.updated_boundary.vertices)
                        self.generated_meshes.append(mesh)

                        # self.boundary.save_intermediate_boundary_fig(
                        #     f"{output_path}/data_augmentation/test/{len(self.generated_meshes)}_left_boundary.png",
                        #     self.updated_boundary.vertices, style='k.-', dpi=400)

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
                        self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices,
                                         lr_1, lr_2, iteration=400)
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

    def get_speed_penalty(self, mesh_area):
        min_area = self.estimated_area_range[0] ** 2 #* 0.5 #*1.5
        critical_area = self.estimated_area_range[1] ** 2 #* 0.5# *1.5

        if min_area <= mesh_area < critical_area:
            speed_penalty = ((mesh_area - critical_area) / (critical_area - min_area))
        elif mesh_area < min_area:
            speed_penalty = -1
        else:
            speed_penalty = 0
        return speed_penalty

    def find_next_state(self, not_valid_points=None, static=False):
        r_p = self.find_reference_point(not_valid_points, target_angle=self.target_angle)

        if r_p:
            # print(r_p)
            p_e = PointEnvironment(reference_point=r_p, boundary=self.updated_boundary,
                                   neighbor_num=self.neighbor_num, radius_num=self.radius_num,
                                   average_edge_length=self.average_edge_length,
                                   area_ratio=self.current_area / self.original_area,
                                   radius=self.radius, static=static)
            self.current_point_environment = p_e

            # state = ([round(elem, 4) for elem in self.transformation()])
            state = p_e.state

            # if len(p_e.radius_neighbors):
            #     [state.append(round(elem, 4)) for elem in self.transformation(p_e.state, self.points_as_array(p_e.radius_neighbors))]
            # state.append(p_e.available_radius)
            # add area ratio
            # state.append(self.current_area / self.original_area)
            return np.array(state).astype(np.float32)       # return self.points_as_array(neighbors)

        else:
            # self.last_point_environment = None
            return None

    def seed(self, seed=None):
        # Old-gym seeding hook; kept so SB3's Gym->Gymnasium compatibility layer
        # (which calls env.seed(seed) on reset) works. Meshing is deterministic
        # given the boundary, so there is no RNG state to set here.
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def render(self, mode='human'):
        print(f'Generated elements: {len(self.generated_meshes)}')
        if self.viewer is None:
            # self.viewer.set_bounds(-1, 12, -6.5, 6.5)
            # self.times = 1
            self.viewer = MeshFrame(self.window_size)
            min_x, max_x = min([v.x for v in self.original_vertices]), max([v.x for v in self.original_vertices])
            min_y, max_y = min([v.y for v in self.original_vertices]), max([v.y for v in self.original_vertices])
            self.times = int(min(self.window_size[0] / (max_x-min_x), self.window_size[1] / (max_y - min_y)))
            self.min_x, self.min_y = min_x, min_y

        all_segts = self.boundary.all_segments()
        for line in all_segts:
            self.viewer.draw_line(((line.point1.x - self.min_x) * self.times, (line.point1.y - self.min_y) * self.times),
                                  ((line.point2.x - self.min_x) * self.times, (line.point2.y - self.min_y) * self.times),
                                  color=(0, 0, 1)) #color=(0, 0, 1)
        time.sleep(.0001)
        self.viewer.render()

    def find_same_point(self, point):
        for p in self.updated_boundary.vertices:
            if p.distance_to(point) < 0.001:
                return p

    def action_2_point(self, action):
        if isinstance(action, np.ndarray):
            x, y = action[0], action[1]
            # x, y = 1.5 * 1.4142 * action[0] * math.cos(math.pi * action[1]), \
            #        1.5 * 1.4142 * action[0] * math.sin(math.pi * action[1])
        else:
            n_action = [action / (self.max_radius * 10), (action % (self.max_radius * 10)) / 10]
            x = round(math.cos(math.radians(n_action[0])), 1) * n_action[1]
            y = round(math.sin(math.radians(n_action[0])), 1) * n_action[1]
        return self.detransformation([round(x, 4), round(y, 4)])

    def plot_sample(self, state, action):
        L = len(state)
        N = int((L - 6) / 2)
        left = [-i - 2 for i in reversed(range(0, N, 2))]
        right = [i for i in range(0, N, 2)]
        all = left + right

        medium = [N + i for i in range(0, 6, 2)]

        n_x = [state[j] * math.cos(state[j + 1]) for j in all]
        n_x.insert(int(N/2), 0)
        r_x = [state[j] * math.cos(state[j + 1]) for j in medium]
        n_y = [state[j] * math.sin(state[j + 1]) for j in all]
        n_y.insert(int(N/2), 0)
        r_y = [state[j] * math.sin(state[j + 1]) for j in medium]
        plt.plot(n_x, n_y, 'k.-')
        plt.plot(r_x, r_y, 'y.-')
        target_x, target_y = action[1] * math.cos(action[2]), action[1] * math.sin(action[2])
        plt.plot(target_x, target_y, 'r.')
        plt.title(f'type: {action[0]}')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def save_history_info(self, filename):
        with open(filename, 'w') as fw:
            json.dump(self.history_info, fw)
