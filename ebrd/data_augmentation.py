import numpy as np
import math
import matplotlib.pyplot as plt
from general.components import Segment, Vertex, Mesh
import json
import seaborn as sns
import random
sns.set_theme(style="darkgrid")
from multiprocessing import Process, Manager
from matplotlib.gridspec import SubplotSpec
import statistics
import pandas as pd
import time


class MeshAugmentation:

    def __init__(self, observation_space, action_space):
        '''
        Sample initiation
        :param observation_space: [<radius, angle>, <radius, angle>, ...]
        :param action_space: [<type, radius, angle>]
        '''
        self.action_space = action_space
        self.observation_space = observation_space
        self.type_label = [0, 0.5, 1]
        # self.type_label = [-1, 0, 1]

    def sample_state(self, angle, type):
        if type == self.type_label[1]:
            obs = [self.random(0.01, 0.45), 0,
                   self.random(0.1, 0.77), self.random(-0.5, 0.9) * math.pi,
                   self.random(0.33, 1), self.random(-0.5, 0.9) * math.pi,
                   self.random(0.15, 1), self.random(0, 1 / 3 * angle),
                   self.random(0.15, 1), self.random(1 / 3 * angle, 2 / 3 * angle),
                   self.random(0.15, 1), self.random(2 / 3 * angle, angle),
                   self.random(0.33, 1), self.random(0.1, 1.5) * math.pi,
                   self.random(0.1, 0.77), self.random(0.01, 1.5) * math.pi,
                   self.random(0.01, 0.45), angle,
                   ]
        elif type == self.type_label[0]:
            obs = [self.random(0.01, 0.45), 0,
                   self.random(0.1, 0.77), self.random(-0.5 * math.pi, angle),
                   self.random(0.33, 1), self.random(-0.5, 0.9) * math.pi,
                   self.random(0.15, 1), self.random(0, 1 / 3 * angle),
                   self.random(0.15, 1), self.random(1 / 3 * angle, 2 / 3 * angle),
                   self.random(0.15, 1), self.random(2 / 3 * angle, angle),
                   self.random(0.33, 1), self.random(0.1, 1.5) * math.pi,
                   self.random(0.1, 0.77), self.random(0.01, angle),
                   self.random(0.01, 0.45), angle,
                   ]
        elif type == self.type_label[2]:
            obs = [self.random(0.01, 0.45), 0,
                   self.random(0.1, 0.77), self.random(0.01, angle),
                   self.random(0.33, 1), self.random(-0.5, 0.9) * math.pi,
                   self.random(0.15, 1), self.random(0, 1 / 3 * angle),
                   self.random(0.15, 1), self.random(1 / 3 * angle, 2 / 3 * angle),
                   self.random(0.15, 1), self.random(2 / 3 * angle, angle),
                   self.random(0.33, 1), self.random(0.1, 1.5) * math.pi,
                   self.random(0.1, 0.77), self.random(0.01, 1.5) * math.pi,
                   self.random(0.01, 0.45), angle,
                   ]
        return obs

    def sample_action(self, obs, angle, type, m):
        acts = []
        left, right = False, False
        if type == self.type_label[1]:
            while len(acts) < m:
                act = [self.type_label[1], self.random(0.01, 0.56), self.random(0.01, angle)]
                acts.append(act)

        elif type == self.type_label[0]:
            acts.append([self.type_label[0], obs[-4], obs[-3]])
        else:
            acts.append([self.type_label[2], obs[2], obs[3]])
        return acts

    def get_quality_distrition(self, N, method):
        distribution = {}
        if method == 'normal':
            distribution = {
                # '0.4': [0.05 * N, 0],
                # '0.5': [0.2 * N, 0],
                # '0.6': [0.25 * N, 0],
                # '0.7': [0.3 * N, 0],
                # '0.8': [0.2 * N, 0],
                # '1':   [0.05 * N, 0],
                # '0.5':  [0.05 * N, 0],
                # '0.55': [0.05 * N, 0],
                # '0.6':  [0.05 * N, 0],
                # '0.65': [0.1 * N, 0],
                # '0.7':  [0.15 * N, 0],
                '0.75': [0.2 * N, 0],
                '0.8':  [0.5 * N, 0],
                '1':    [0.3 * N, 0],
            }
        elif method == 'uniform':
            distribution = {
                '0.4': [0.2 * N, 0],
                '0.5': [0.2 * N, 0],
                '0.6': [0.2 * N, 0],
                '0.7': [0.2 * N, 0],
                '0.8': [0.2 * N, 0],
                '1':   [0.2 * N, 0],
            }

        return distribution

    def get_angle_distribution(self, N, method='uniform'):
        distribution = {}
        if method == 'uniform':
            m = 11
            distribution = {
                # '0.2': [N / m, 0],
                '0.4': [math.ceil(N / m), 0],
                '0.6': [math.ceil(N / m), 0],
                '0.8': [math.ceil(N / m), 0],
                '1': [math.ceil(N / m), 0],
                '1.2': [math.ceil(N / m), 0],
                '1.4': [math.ceil(N / m), 0],
                '1.6': [math.ceil(N / m), 0],
                '1.8': [math.ceil(N / m), 0],
                '2': [math.ceil(N / m), 0],
                '2.2': [math.ceil(N / m), 0],
                '2.4': [math.ceil(N / m), 0],
            }

        return distribution

    def sampling(self, n, threshold, file_name=None):
        # actions = [self.action_space.sample() for i in range(n)]
        # observations = [self.observation_space.sample() for i in range(n)]
        observations, actions, metrics = [], [], []
        start_time = time.time()

        t1 = self.sampling_4_type(0.5 * n, threshold, self.type_label[1])
        t2 = self.sampling_4_type(0.25 * n, threshold, self.type_label[0])
        t3 = self.sampling_4_type(0.25 * n, threshold, self.type_label[2])

        observations.extend(t1[0])
        actions.extend(t1[1])
        metrics.extend(t1[2])

        observations.extend(t2[0])
        actions.extend(t2[1])
        metrics.extend(t2[2])

        observations.extend(t3[0])
        actions.extend(t3[1])
        metrics.extend(t3[2])

        # print(f"Type -1: {l}; Type 0: {m}; TYpe 1: {r}")
        # data = {'observation': observations, 'action': actions, 'quality': metrics}
        # with open('D:\\meshingData\\ANN\\data_augmentation\\1\samples.json', 'w+') as f:
        #     json.dump(data, f)

        # actions = np.asarray(actions)
        # training_data = {'samples': observations,
        #                  'output_types': actions[:, 0:1].tolist(),
        #                  'outputs': actions[:, 1:].tolist()}

        observations, actions = np.asarray(observations), np.asarray(actions)
        # training_data = {'samples': observations[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]].tolist(),
        #                  'output_types': actions[:, 0:1].tolist(),
        #                  'outputs': actions[:, 1:].tolist()}
        training_data = {'samples': observations.tolist(),
                         'output_types': actions[:, 0:1].tolist(),
                         'outputs': actions[:, 1:].tolist()}

        print("Sampling completed in: ", time.time() - start_time, 's')

        if file_name is not None:
            with open(file_name, 'w+') as f:
                json.dump(training_data, f)
        return training_data

    def get_Q_key(self, quality):
        # if quality <= 0.5:
        #     return "0.5"
        # elif 0.5 < quality <= 0.55:
        #     return "0.55"
        # if quality <= 0.6:
        #     return "0.6"
        # elif 0.6 < quality <= 0.65:
        #     return "0.65"
        # elif 0.65 < quality <= 0.7:
        #     return "0.7"
        # elif 0.7 < quality <= 0.75:
        if quality <= 0.75:
            return "0.75"
        elif 0.75 < quality <= 0.8:
            return "0.8"
        elif quality > 0.8:
            return "1"

    def get_angle_key(self, angle):
        if angle < 0.2 or angle > 2.4:
            return None
        elif 0.2 <= angle <= 0.4:
            return '0.4'
        elif 0.4 < angle <= 0.6:
            return '0.6'
        elif 0.6 < angle <= 0.8:
            return '0.8'
        elif 0.8 < angle <= 1:
            return '1'
        elif 1 < angle <= 1.2:
            return '1.2'
        elif 1.2 < angle <= 1.4:
            return '1.4'
        elif 1.4 < angle <= 1.6:
            return '1.6'
        elif 1.6 < angle <= 1.8:
            return '1.8'
        elif 1.8 < angle <= 2:
            return '2'
        elif 2 < angle <= 2.2:
            return '2.2'
        else:
            return '2.4'

    def sampling_4_type(self, n, threshold, type):
        observations, actions, metrics = [], [], []
        i = 0
        _i = 0
        # quadlity_ditribution = self.get_quality_distrition(n, method='normal')
        angle_distribution = self.get_angle_distribution(n)

        for k, v in angle_distribution.items():
            while angle_distribution[k][1] < v[0]:

            # while len(observations) < n:
            # angle = self.random(0.1, 0.9 * math.pi)
                angle = self.random(float(k) - 0.2, float(k))
                obs = self.sample_state(angle, type=type)

                if self.check_obs_quality(obs):
                    continue

                # for type 0.5
                sample_acts = self.sample_action(obs, angle, type=type, m=20)
                for act in sample_acts:
                    ele_quality, boundary_quality = self.compute_quality([obs, act])
                    quality = math.sqrt(ele_quality * boundary_quality)

                    lam = 0.618
                    quality = (1-lam) * ele_quality + lam * boundary_quality

                    max_ele_quality, max_boun_quality = self.estimate_max_quality([obs, act], method=3)
                    max_quality = (1-lam) * max_ele_quality + lam * max_boun_quality

                    _i += 1

                    if quality / max_quality >= threshold and ele_quality / max_ele_quality >= threshold: #
                        # print(f"product: {quality}, weight: {0.5 * ele_quality + 0.5 * boundary_quality}")
                        # print(f"m2: {self.estimate_max_quality([obs, act], method=2)}, m3:{self.estimate_max_quality([obs, act], method=3)}")
                        # print(f"quality: {quality}, \n "#max ele Q: {max_ele_quality} max Q: {max_quality}
                        #       f"element Q: {ele_quality}, max ele Q: {max_ele_quality}, ratio: {ele_quality / max_ele_quality}\n" #= {ele_quality / max_ele_quality} max ele Q: {max_ele_quality}
                        #       f"boundary Q: {boundary_quality},") #max bound: {max_boun_quality},
                        # self.plot_sample(obs, act)

                        # _key = self.get_Q_key(quality)
                        # # print(quadlity_ditribution)
                        # if quadlity_ditribution[_key][1] >= quadlity_ditribution[_key][0]:
                        #     continue
                        # quadlity_ditribution[_key][1] += 1

                        # _key = self.get_angle_key(angle)
                        # if _key is None:
                        #     continue
                        # print(angle_distribution)
                        # if angle_distribution[_key][1] >= angle_distribution[_key][0]:
                        #     continue
                        # angle_distribution[_key][1] += 1
                        angle_distribution[k][1] += 1
                        if angle_distribution[k][1] >= angle_distribution[k][0]:
                            break

                        # print(angle_distribution)

                        observations.append(obs)
                        actions.append(act)
                        # metrics.append(quality)
                        if i % 1000 == 0:
                            print(f"Sampling length: {i} out of {n} for type {type} with {_i} tries")
                        i += 1
        print(f"Sampling length: {i} out of {n} for type {type} with {_i} tries")
        return observations, actions, metrics
    
    def check_obs_quality(self, obs):
        points = [Vertex(obs[i] * math.cos(obs[i+1]), obs[i] * math.sin(obs[i+1])) for i in range(0, len(obs) - 1, 2)]

        n_points = [points[-3], points[-2], points[-1], Vertex(0, 0), points[0], points[1], points[2]]
        r_points = [points[3], points[4], points[5]]

        for i in range(1, len(n_points)):
            for j in range(i + 2, len(n_points)):
                s1 = Segment(n_points[i], n_points[i - 1])
                s2 = Segment(n_points[j], n_points[j - 1])
                if s1.is_cross(s2):
                    return True

            for k in range(1, len(r_points)):
                s1 = Segment(n_points[i], n_points[i - 1])
                s2 = Segment(r_points[k], r_points[k - 1])
                if s1.is_cross(s2):
                    return True

        # check

        lengths = [n_points[i].distance_to(n_points[i-1]) for i in range(1, len(n_points))]
        if max(lengths) / min(lengths) >= 5:
            return True
        return False

    def random(self, low, high):
        return np.random.random() * (high - low) + low

    def plot_sample(self, state, action, frame_on=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
        # if action[0] == 0:
        #     title = 1
        # elif action[0] == 1:
        #     title = 2
        # else:
        #     title = 0
        title = action[0]
        plt.title(f'type: {title}')
        if not frame_on:
            # ax.set_frame_on(False)
            plt.xticks([])
            plt.yticks([])
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def sample_2_vertices(self, sample):
        state, action = sample

        reference_vertex = Vertex(0, 0)

        L = len(state)
        N = int((L-6) / 2)
        left = [-i-2 for i in reversed(range(0, N, 2))]
        right = [i for i in range(0, N, 2)]
        all = left + right

        n_points = [Vertex(state[j] * math.cos(state[j + 1]), state[j] * math.sin(state[j + 1]))
                    for j in all]
        n_points.insert(int(N/2), reference_vertex)

        medium = [N + i for i in range(0, 6, 2)]

        r_points = [Vertex(state[j] * math.cos(state[j + 1]), state[j] * math.sin(state[j + 1]))
                    for j in medium]

        # base_length = sum([n_points[i].distance_to(n_points[i - 1])
        #                    for i in range(2, len(n_points) - 1)]) / (len(n_points) - 3)
        base_length = 1

        rule_type, new_point = action[0], Vertex(base_length * action[1] * math.cos(action[2]),
                                                 base_length * action[1] * math.sin(action[2]))

        return n_points, r_points, rule_type, new_point

    def estimate_max_quality(self, sample, method=2):
        # if sample[0][-1] > 1:
        #     return 1

        n_points, r_points, rule_type, new_point = self.sample_2_vertices(sample)
        index = int(len(n_points) / 2)

        left_v = n_points[index - 1]
        right_v = n_points[index + 1]

        if method == 1:
            v1, v2 = left_v.get_perpendicular_vertex(right_v)

            if v1.to_find_clockwise_angle(left_v, right_v) < math.pi:
                new_point = v2
            else:
                new_point = v1

            mesh = Mesh([new_point,
                         right_v,
                         n_points[index],
                         left_v
                         ])
            element_quality = mesh.get_quality(type='strong') #type='area' type='strong'
            boundary_quality = self.compute_boundary_quality(mesh, n_points, r_points)
            # max_quality = math.sqrt(element_quality * boundary_quality)
            return element_quality, boundary_quality

        elif method == 2:
            # max angle quality
            angle = n_points[index].to_find_clockwise_angle(left_v, right_v)
            rem_angle = (2 * math.pi - angle) / 3
            # angle_quality = min(math.pi - angle, angle) / max(math.pi - angle, angle)

            angle_product = 1
            for _angle in [angle, rem_angle, rem_angle, rem_angle]:
                angle_product *= 1 - (
                            math.fabs(math.degrees(_angle) - 90) / 90)
            # print(angle_product)
            if angle_product < 0:
                angle_quality = 0
            else:
                angle_quality = math.pow(angle_product, 1 / 4)

            # max edge quality
            l1, l2 = n_points[index].distance_to(left_v), n_points[index].distance_to(right_v)
            # edge_quality = min(l1, l2) / max(l1, l2)
            edge_product = 1
            area = l2 * l1 * math.sin(angle)
            for edge in [l1, l2]:
                edge_product *= math.pow(edge / math.sqrt(area), 1 if math.sqrt(area) - edge > 0 else -1)
            edge_quality = math.pow(edge_product, 1 / 2)
            # print(f"Angle: {angle_quality}; Edge: {edge_quality}; {angle_quality * edge_quality}")
            # max_quality =

            # max boundary quality
            segments = [Segment(r_points[0], r_points[1]), Segment(r_points[1], r_points[2])]
            dists = []
            for s in segments:
                dists.append(s.distance(n_points[index]))

            return math.sqrt(angle_quality * edge_quality), math.sqrt(min(1, max(dists) / ((math.sqrt(2)/2 + 0.5) * (l1 + l2))))
        elif method == 3:
            # max angle quality
            angle = n_points[index].to_find_clockwise_angle(left_v, right_v)
            rem_angle = (2 * math.pi - angle) / 3
            angle_quality = min(angle, rem_angle) / max(angle, rem_angle)

            # max edge quality
            l1, l2 = n_points[index].distance_to(left_v), n_points[index].distance_to(right_v)
            # edge_quality = min(l1, l2) / max(l1, l2)
            edge_product = 1
            area = l2 * l1 * math.sin(angle)
            for edge in [l1, l2]:
                edge_product *= math.pow(edge / math.sqrt(area), 1 if math.sqrt(area) - edge > 0 else -1)
            edge_quality = math.pow(edge_product, 1 / 4)

            segments = [Segment(r_points[0], r_points[1]), Segment(r_points[1], r_points[2])]
            dists = []
            for s in segments:
                dists.append(s.distance(n_points[index]))

            return math.sqrt(angle_quality * edge_quality), math.sqrt(
                min(1, max(dists) / ((math.sqrt(2) / 2 + 0.5) * (l1 + l2))))

    def compute_quality(self, sample):
        '''
        Compute the quality for the sample
        :param sample: [observation, action]
        :return: the integration of element quality and boundary quality
        '''
        n_points, r_points, rule_type, new_point = self.sample_2_vertices(sample)

        index = int(len(n_points) / 2)
        if rule_type == self.type_label[0]:
            mesh = Mesh([ # vertices in clockwise direction
                n_points[index - 2],
                n_points[(index + 1) % len(n_points)],
                n_points[index],
                n_points[index - 1]
            ])
        elif rule_type == self.type_label[2]:
            mesh = Mesh([
                n_points[index - 1],
                n_points[(index + 2) % len(n_points)],
                n_points[
                    (index + 1) % len(n_points)],
                n_points[index]
            ])
        else:
            mesh = Mesh([new_point,
                         n_points[
                             (index + 1) % len(n_points)],
                         n_points[index],
                         n_points[index - 1]
                         ])

        if self.validate_mesh(mesh, quality_method=0) and \
                not self.check_intersection(mesh, n_points, r_points) and \
                not self.check_internal_element(mesh, r_points + [n_points[0], n_points[-1]] if len(n_points) > 5
                                                    else r_points):
            element_quality = mesh.get_quality(type='strong') #type='area' type='strong'
            boundary_quality = self.compute_boundary_quality(mesh, n_points, r_points)

            # print(f"Element quality: {mesh.get_quality(type='strong')}; Boundary quality: {boundary_quality}")
            # return element_quality * boundary_quality if need_boundary else element_quality
            return element_quality, boundary_quality
        else:
            return 0, 0

    def validate_mesh(self, mesh, quality_method=0):
        if not mesh.is_valid(quality_method):
            return False

        return True

    def check_intersection(self, mesh, neighbors, r_neighbors):
        for i in range(4):
            s1 = Segment(mesh.vertices[i], mesh.vertices[i - 1])
            for j in range(1, len(neighbors)):
                s = Segment(neighbors[j], neighbors[j-1])
                if mesh.vertices[i] not in [neighbors[j], neighbors[j-1]] and \
                    mesh.vertices[i-1] not in [neighbors[j], neighbors[j-1]]:
                    if s1.is_cross(s):
                        return True

            for j in range(1, len(r_neighbors)):
                s = Segment(r_neighbors[j], r_neighbors[j-1])
                if mesh.vertices[i] not in [r_neighbors[j], r_neighbors[j-1]] and \
                    mesh.vertices[i-1] not in [r_neighbors[j], r_neighbors[j-1]]:
                    if s1.is_cross(s):
                        return True
        return False

    def check_internal_element(self, mesh, radius_points):
        centroid = mesh.get_centriod()

        for v in radius_points:
            s1 = Segment(v, centroid)
            crossed = False
            for i in range(4):
                s = Segment(mesh.vertices[i], mesh.vertices[i - 1])
                if s1.is_cross(s):
                    crossed = True
                    break
            if not crossed:
                return True
        return False

    def compute_boundary_quality(self, mesh, neighbors, r_neighbors):
        new_v = [(i, v) for i, v in enumerate(mesh.vertices) if v not in neighbors]

        if len(new_v):
            # Has newly generated vertex
            id, new_v = new_v[0]
            # Angle quality
            left_v = mesh.vertices[id-1]
            right_v = mesh.vertices[(id + 1) % 4]
            left_angle = left_v.to_find_clockwise_angle(neighbors[neighbors.index(left_v) - 1], new_v)
            right_angle = right_v.to_find_clockwise_angle(new_v, neighbors[neighbors.index(right_v) + 1])
            angles = []
            if left_angle < math.pi / 2:
                angles.append(left_angle)
            if right_angle < math.pi / 2:
                angles.append(right_angle)
                # product *= 3 * angle / math.pi
            q1 = 2 * min(angles) / math.pi if len(angles) else 1

            # Distance quality
            dist = new_v.distance_to(left_v) + new_v.distance_to(right_v)
            if len(neighbors) > 5:
                segments = [Segment(neighbors[0], neighbors[1]), Segment(neighbors[-1], neighbors[-2]),
                            Segment(r_neighbors[0], r_neighbors[1]), Segment(r_neighbors[1], r_neighbors[2])]
            else:
                segments = [Segment(r_neighbors[0], r_neighbors[1]), Segment(r_neighbors[1], r_neighbors[2])]
            dists = []
            for s in segments:
                dists.append(s.distance(new_v))

            if len(dists):
                m_d = min(dists)
                q2 = m_d / dist if m_d < dist else 1
            else:
                q2 = 1

            # compute smoothness
            targt_len = dist / 2
            k = int(len(neighbors) / 2)
            _dists = [neighbors[k - 2], neighbors[k - 1], new_v, neighbors[k + 1], neighbors[k + 2]]
            mean_dist = sum([_dists[i].distance_to(_dists[i - 1]) for i in range(1, len(_dists))]) / (len(_dists) - 1)

            smoothness = min(mean_dist, targt_len) / max(mean_dist, targt_len)
            return math.pow(q1 * q2 * smoothness, 1 / 3)
            # return math.sqrt(q1*q2)
        else:
            indices = sorted([neighbors.index(v) for v in mesh.vertices])
            left_ind, right_ind = indices[0], indices[-1]

            angles = []

            if left_ind == 0:
                right_angle = neighbors[right_ind].to_find_clockwise_angle(neighbors[left_ind],
                                                                           neighbors[right_ind + 1])
                if right_angle < math.pi / 2:
                    angles.append(right_angle)

                _dists = [neighbors[left_ind], neighbors[right_ind], neighbors[right_ind + 1]]
            elif right_ind == len(neighbors) - 1:
                left_angle = neighbors[left_ind].to_find_clockwise_angle(neighbors[left_ind - 1],
                                                                         neighbors[right_ind])

                if left_angle < math.pi / 2:
                    angles.append(left_angle)
                _dists = [neighbors[left_ind - 1], neighbors[left_ind], neighbors[right_ind]]
            else:
                left_angle = neighbors[left_ind].to_find_clockwise_angle(neighbors[left_ind - 1],
                                                                         neighbors[right_ind])
                right_angle = neighbors[right_ind].to_find_clockwise_angle(neighbors[left_ind],
                                                                           neighbors[right_ind + 1])
                if left_angle < math.pi / 2:
                    angles.append(left_angle)
                if right_angle < math.pi / 2:
                    angles.append(right_angle)
                    # product *= 3 * angle / math.pi
                _dists = [neighbors[left_ind - 1], neighbors[left_ind], neighbors[right_ind], neighbors[right_ind + 1]]

            # angle quality
            angle_quality = 2 * min(angles) / math.pi if len(angles) else 1

            # compute smoothness
            targt_len = neighbors[left_ind].distance_to(neighbors[right_ind])
            mean_dist = sum([_dists[i].distance_to(_dists[i - 1]) for i in range(1, len(_dists))]) / (
                    len(_dists) - 1)

            smoothness = min(mean_dist, targt_len) / max(mean_dist, targt_len)
            boundary_quality = math.sqrt(angle_quality * smoothness)
            # boundary_quality = angle_quality * smoothness
            return boundary_quality

    @staticmethod
    def load_samples(filename):
        with open(filename, 'r') as fr:
            data = json.load(fr)

        return data

    def samples_2_plot_data(self, samples, quality_threshold=0, include_quality=True):
        data = {
            'Type 0': {
                'Neighbors': {'x': [], 'y': []},
                'RNeighbors': {'x': [], 'y': []},
                'Angles': [],
                'Quality': []
            },
            'Type 1': {
                'Neighbors': {'x': [], 'y': []},
                'RNeighbors': {'x': [], 'y': []},
                'Vertex': {'x': [], 'y': []},
                'Angles': [],
                "Quality": []
            },
            'Type 2': {
                'Neighbors': {'x': [], 'y': []},
                'RNeighbors': {'x': [], 'y': []},
                'Angles': [],
                "Quality": []
            }
        }

        actions = np.asarray(list(map(list.__add__, samples['output_types'], samples['outputs'])))
        observations = np.asarray(samples['samples'])
        # print(f"Actions: \n min: {actions.min(axis=0)}; "
        #       f"\n max: {actions.max(axis=0)}")
        # print(f"Observations: \n min: {observations.min(axis=0)}; \n max: {observations.max(axis=0)}")
        for i in range(len(observations)):
            # print(self.compute_quality([observations[i], actions[i]], need_boundary=False))
            if 'quality' in samples.keys():
                if samples['quality'][i] < quality_threshold:
                    continue

            n_points, r_points, rule_type, new_point = self.sample_2_vertices([observations[i],
                                                                               actions[i]])
            if include_quality:
                element_quality, boundary_quality = self.compute_quality([observations[i], actions[i]])
                quality = math.sqrt(element_quality * boundary_quality)

            else:
                quality = 0

            if rule_type == self.type_label[0]:
                data['Type 0']['Neighbors']['x'].extend([p.x for p in n_points])
                data['Type 0']['Neighbors']['y'].extend([p.y for p in n_points])
                data['Type 0']['RNeighbors']['x'].extend([p.x for p in r_points])
                data['Type 0']['RNeighbors']['y'].extend([p.y for p in r_points])
                data['Type 0']['Angles'].append(observations[i][-1])
                data['Type 0']['Quality'].append(quality)

            elif rule_type == self.type_label[2]:
                data['Type 2']['Neighbors']['x'].extend([p.x for p in n_points])
                data['Type 2']['Neighbors']['y'].extend([p.y for p in n_points])
                data['Type 2']['RNeighbors']['x'].extend([p.x for p in r_points])
                data['Type 2']['RNeighbors']['y'].extend([p.y for p in r_points])
                data['Type 2']['Angles'].append(observations[i][-1])
                data['Type 2']['Quality'].append(quality)
            else:
                data['Type 1']['Neighbors']['x'].extend([p.x for p in n_points])
                data['Type 1']['Neighbors']['y'].extend([p.y for p in n_points])
                data['Type 1']['RNeighbors']['x'].extend([p.x for p in r_points])
                data['Type 1']['RNeighbors']['y'].extend([p.y for p in r_points])
                data['Type 1']['Vertex']['x'].append(new_point.x)
                data['Type 1']['Vertex']['y'].append(new_point.y)
                data['Type 1']['Angles'].append(observations[i][-1])
                data['Type 1']['Quality'].append(quality)

        return data

    def scatter_plot(self, samples):
        data = self.samples_2_plot_data(samples)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        i = 0
        for k, v in data.items():

            axs[0, i].plot(v['Neighbors']['x'], v['Neighbors']['y'], 'b.',
                        v['RNeighbors']['x'], v['RNeighbors']['y'], 'y.')
            axs[0, i].set_title(k + ": vertex distribution")
            if k == 'Type 1':
                axs[0, i].plot(v['Vertex']['x'], v['Vertex']['y'], 'r.')

            # plot angle distribution
            axs[1, i].hist(v['Angles'], 15)
            axs[1, i].set_title(k + ': angle distribution')
            # Quality distribution
            axs[2, i].hist(v['Quality'], 10)
            axs[2, i].set_title(k + ': quality distribution')

            i += 1
        fig.tight_layout()
        plt.show()

    def comparison_scatter_plot(self, samples1, samples2, include_quality=False):
        data1 = self.samples_2_plot_data(samples1, include_quality=include_quality)
        data2 = self.samples_2_plot_data(samples2, include_quality=include_quality)

        # subfigs[0].suptitle(f'Samples extracted from Gmsh')

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        i = 0
        # fig.suptitle("Angle distribution")
        for k in ["Type 0", "Type 1", "Type 2"]:
            v = data1[k]
            title = k

            # axs[0, i].plot(v['Neighbors']['x'], v['Neighbors']['y'], 'b.',
            #                v['RNeighbors']['x'], v['RNeighbors']['y'], 'y.')
            # axs[0, i].set_title(title, fontsize=18)
            # if k == 'Type 1':
            #     axs[0, i].plot(v['Vertex']['x'], v['Vertex']['y'], 'r.')

            axs[0, i].hist(random.sample(v['Angles'], 10000) if len(v['Angles']) > 10000 else v['Angles'], 15)
            axs[0, i].set_title(title, fontsize=18)
            axs[0, i].set_xlim(0.5, 2.5)

            i += 1

        # subfigs[1].suptitle(f'Samples generated by FreeMesh-DG')
        # axs = subfigs[1].subplots(nrows=1, ncols=3)
        i = 0
        for k in ["Type 0", "Type 1", "Type 2"]:
            v = data2[k]
            title = k

            # axs[1, i].plot(v['Neighbors']['x'], v['Neighbors']['y'], 'b.',
            #                v['RNeighbors']['x'], v['RNeighbors']['y'], 'y.')
            # axs[1, i].set_title(title, fontsize=18)
            # if k == 'Type 1':
            #     axs[1, i].plot(v['Vertex']['x'], v['Vertex']['y'], 'r.')

            axs[1, i].hist(random.sample(v['Angles'], 10000) if len(v['Angles']) > 10000 else v['Angles'], 15)
            axs[1, i].set_title(title, fontsize=18)
            axs[1, i].set_xlim(0.5, 2.5)
            i += 1

        grid = plt.GridSpec(2, 3)
        self.create_subtitle(fig, grid[0, ::], '(a) Samples extracted from Gmsh', fontsize=20)
        self.create_subtitle(fig, grid[1, ::], '(b) Samples generated by FreeMesh-DG', fontsize=20)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str, fontsize: int):
        "Sign sets of subplots with title"
        row = fig.add_subplot(grid)
        # the '\n' is important
        row.set_title(f'{title}\n', fontweight='semibold', fontsize=fontsize)
        # hide subplot
        row.set_frame_on(False)
        row.axis('off')

    def resampling(self, file_name, target_name, n):
        data = MeshAugmentation.load_samples(file_name)
        # actions = np.asarray(list(map(list.__add__, data['output_types'], data['outputs'])))
        # observations = np.asarray(data['samples'])
        type_indices = {'-1': [],
                        '0': [],
                        '1': []}
        for i, ot in enumerate(data['output_types']):
            if ot[0] == 0:
                type_indices['-1'].append(i)
            elif ot[0] == 0.5:
                type_indices['0'].append(i)
            else:
                type_indices['1'].append(i)

        type_indices['-1'] = random.choices(type_indices['-1'], k=int(0.3*n))
        type_indices['0'] = random.choices(type_indices['0'], k=int(0.4*n))
        type_indices['1'] = random.choices(type_indices['1'], k=int(0.3*n))

        resampling_data = {'samples': [],
                         'output_types': [],
                         'outputs': []}

        for k, v in type_indices.items():
            # if k == '-1':
            #
            # elif k == '0':
            #     pass
            # else:
            #     pass
            resampling_data['samples'].extend([data['samples'][i] for i in v])
            resampling_data['output_types'].extend([data['output_types'][i] for i in v])
            resampling_data['outputs'].extend([data['outputs'][i] for i in v])

        with open(target_name, 'w+') as f:
            json.dump(resampling_data, f)
        return resampling_data

    def test(self):
        angle = math.pi / 9
        sample = [1, 0, 2, 0, 3, 0, 6, angle/6, 6, angle/2, 6, 5*angle/6, 3, angle, 2, angle, 1, angle]
        actions = [
            [0.5, 1, angle/2],
            [0.5, 1.2, angle/2],
            [0.5, 1.5, angle / 2],
            [0.5, 1.8, angle / 2],
            [0.5, 2, angle / 2],
            [0.5, 3, angle / 2],
        ]
        for a in actions:
            e_q, b_q = self.compute_quality([sample, a])
            print(f"Element Q: {e_q}, Boundary Q: {b_q}, Q: {math.sqrt(e_q * b_q)}")

    def data_evaluation(self, files):
        datasets = {}
        metrics_data = {
            "Element quality": pd.DataFrame({"value": [], "threshold": []}),
            'Boundary quality': pd.DataFrame({"value": [], "threshold": []}),
            'Quality': pd.DataFrame({"value": [], "threshold": []}),
            'Angle': pd.DataFrame({"value": [], "threshold": []}),
            'Averaged segment length': pd.DataFrame({"value": [], "threshold": []}),
        }

        for k, v in files.items():
            if v is None:
                continue
            data = self.load_samples(v)
            actions = np.asarray(list(map(list.__add__, data['output_types'], data['outputs'])))
            observations = np.asarray(data['samples'])

            metrics = {
                'Element quality': {"value": [], "threshold": []},
                'Boundary quality': {"value": [], "threshold": []},
                'Angle': {"value": [], "threshold": []},
                'Averaged segment length': {"value": [], "threshold": []},
                'Quality': {"value": [], "threshold": []}
            }
            for i in range(len(observations)):
                # print(self.compute_quality([observations[i], actions[i]], need_boundary=False))

                n_points, r_points, rule_type, new_point = self.sample_2_vertices([observations[i],
                                                                                   actions[i]])

                # element_quality, boundary_quality = self.compute_quality([observations[i], actions[i]])
                element_quality, boundary_quality = 1, 1
                metrics['Quality']["value"].append(math.sqrt(element_quality*boundary_quality))
                metrics['Quality']["threshold"].append(k)

                metrics['Element quality']["value"].append(element_quality)
                metrics['Element quality']["threshold"].append(k)

                metrics['Boundary quality']["value"].append(boundary_quality)
                metrics['Boundary quality']["threshold"].append(k)

                metrics['Angle']["value"].append(observations[i][-1])
                metrics['Angle']["threshold"].append(k)

                metrics['Averaged segment length']["value"].append(sum([n_points[j].distance_to(n_points[j-1])
                                              for j in range(1, len(n_points))]) / len(n_points))
                metrics['Averaged segment length']["threshold"].append(k)

            for name, r in metrics.items():
                metrics_data[name] = pd.concat([metrics_data[name], pd.DataFrame(r)])

            datasets[k] = metrics
            # for m, v in metrics.items():
            #     print(f"{k}: {m} mean {statistics.mean(v)}, std {statistics.stdev(v)}, max: {max(v)}, min: {min(v)}")

        fig, axs = plt.subplots(2, 3, sharex=True)
        i = 0
        for k, v in metrics_data.items():
            # if i == 0:
            #     sns.boxplot(ax=axs[i], x="method", y="value", data=v, hue='method', showfliers=False)
            # else:
            if i == 4:
                sns.boxplot(ax=axs[i // 3, i % 3], x="threshold", y="value", data=v)
                # axs[i // 3, i % 3].legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)
                # axs[i // 3, i % 3].legend(loc='lower right')
                # axs[i // 3, i % 3].legend(bbox_to_anchor=(1, 0.4),
                #            bbox_transform=plt.gcf().transFigure)
            else:
                sns.boxplot(ax=axs[i // 3, i % 3], x="threshold", y="value", data=v)

            if k == 'Element quality':
                title = '(a) Element quality'
            elif k == 'Boundary quality':
                title = '(b) Boundary quality'
            elif k == 'Quality':
                title = '(c) Quality'
            elif k == 'Angle':
                title = '(d) Angle'
            elif k == 'Averaged segment length':
                title = '(e) Averaged segment length'

            axs[i // 3, i % 3].set_title(title)
            axs[i // 3, i % 3].xaxis.set_label_text('foo')
            axs[i // 3, i % 3].xaxis.label.set_visible(False)
            axs[i // 3, i % 3].yaxis.set_label_text('foo')
            axs[i // 3, i % 3].yaxis.label.set_visible(False)
            i += 1
        axs[1, 2].axis('off')
        fig.tight_layout()
        plt.show()
        return datasets


def sampling_worker(i, n, threshold, data):
    mg = MeshAugmentation([], [])
    data[i] = mg.sampling(n, threshold)


def sampling_main(pool, N, threshold, file_name=None, is_plot=False):
    manager = Manager()
    return_dict = manager.dict()

    jobs = []
    for i in range(pool):
        p = Process(target=sampling_worker,
                    args=(i, N / pool, threshold, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    sample_data = {'samples': [],
                   'output_types': [],
                   'outputs': []}

    for p in return_dict.values():
        for k, v in p.items():
            sample_data[k].extend(v)

    if file_name is not None:
        with open(file_name, 'w+') as f:
            json.dump(sample_data, f)

    if is_plot:
        mg = MeshAugmentation([], [])
        mg.scatter_plot(sample_data)

    return sample_data

if __name__ == '__main__':
#     # observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#     #                                np.array([1/2, 0, 1, 2 * math.pi,
#     #                                          1, math.pi, 1, math.pi, 1, math.pi,
#     #                                          1, 3 * math.pi / 2, 1/2, math.pi]), dtype=np.float32)
#     # action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1/3, math.pi]), dtype=np.float32)
    mg = MeshAugmentation([], [])
    # mg.test()
    # res = mg.sampling(1000000)
    # for i in range(len(res[0])):
    #     # mg.plot_sample(res[0][i], res[1][i])
    #     print(mg.compute_quality([res[0][i], res[1][i]]))
    # test()
    # data = mg.load_samples('D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_2.7_100.json')
    # data = mg.load_samples('D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_6.json')
    # data = mg.sampling(40000, threshold=0.7,
    #                    file_name='D:\meshingData\ANN\data_augmentation\\1103\\training_samples_1_40k_07.json')

    # mg.scatter_plot(data)
    # mg.resampling('D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_0.7.json',
    #               target_name='D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_0.7_1.json',
    #               distr=[300, 400, 300])

#     # multiprocessing for sampling
#     sampling_main(10, 1000, 0.85,
#                   file_name='D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_1.7_0.6.json',
#                   is_plot=True)

    ### draw sample comparison graphes
    data1 = mg.load_samples('D:\meshingData\ANN\data_augmentation\\1\\data_aug.json')
    data2 = mg.load_samples('D:\meshingData\ANN\data_augmentation\\1\\training_samples_1_40k_07.json')
    mg.comparison_scatter_plot(data1, data2, include_quality=False)

    # mg.scatter_plot(data2)

    #### Dataset evaluation
    # files = {
    #     '0.6': 'D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_1.7_0.6.json',
    #     '0.7': 'D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_1.7_0.7.json',
    #     '0.75': 'D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_1.7_0.75.json',
    #     '0.8': 'D:\meshingData\ANN\data_augmentation\\1\\training_samples_data_aug_1.7_0.7999999999999999.json',
    # }
    # mg.data_evaluation(files)