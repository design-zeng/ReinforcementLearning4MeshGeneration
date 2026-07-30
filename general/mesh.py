import math
import json
import itertools
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from general.components import *

class MeshGeneration:
    def __init__(self, boundary):
        self.boundary = boundary
        self.generated_meshes = []
        self.updated_boundary = boundary.copy()
        self.all_vertices = boundary.vertices
        self.original_vertices = list(boundary.vertices)
        self.candidate_vertices: Any = None
        self.test_candidate_vertices: Any = []
        self.maximum_reference_angle = math.pi * 0.972
        self.num_ref_neighbor = 4
        self.rp_index = 0
        self.average_edge_length = self.boundary.average_edge_length()

    @staticmethod
    def calculate_crossing_segments(closed_curve_vertices, ray_segment):
        '''
        http://geomalgorithms.com/a03-_inclusion.html#:~:text=Inclusion%20of%20a%20Point%20in%20a%20Polygon&text=%2D%20which%20counts%20the%20number%20of,%22even%2Dodd%22%20test.
        :param closed_curve_vertices:
        :param ray_segment:
        :return:
        '''
        count = 0
        for i in range(len(closed_curve_vertices)):
            seg = Segment(closed_curve_vertices[i], closed_curve_vertices[i - 1])
            # excludes horizontal segments
            orientation = round(closed_curve_vertices[i].y - closed_curve_vertices[i - 1].y, 4)
            if orientation == 0:
                continue
            if seg.is_cross(ray_segment):
                if round(closed_curve_vertices[i].y - ray_segment.point2.y, 4) == 0:
                    next_orientation = round(closed_curve_vertices[(i + 1) % len(closed_curve_vertices)].y - closed_curve_vertices[i].y, 4)
                    if next_orientation == 0:
                        continue
                    elif next_orientation * orientation < 0:
                        continue
                    else:
                        if orientation < 0:
                            count += 1
                        else:
                            continue
                else:
                    if round(closed_curve_vertices[i - 1].y - ray_segment.point2.y, 4) == 0:
                        pre_orientation = round(closed_curve_vertices[i - 1].y - closed_curve_vertices[i - 2].y, 4)

                        if pre_orientation == 0:
                            continue
                        elif pre_orientation * orientation < 0:
                            continue
                        else:
                            if orientation < 0:
                                continue
                            else:
                                count += 1
                    else:
                        count += 1
        return count

    def count_crossing_segments(self, ray_segment):
        return self.calculate_crossing_segments(self.updated_boundary.vertices, ray_segment)

    def is_inside(self, ray_segment):
        return self.count_crossing_segments(ray_segment) % 2 != 0

    def remove_reference_candidates(self, points):
        if isinstance(points, list):
            for p in points:
                i = 0
                while i < len(self.candidate_vertices):
                    if self.candidate_vertices[i][0] == p:
                        del self.candidate_vertices[i]
                    else:
                        i += 1
                i = 0
                while i < len(self.test_candidate_vertices):
                    if self.test_candidate_vertices[i][0] == p:
                        del self.test_candidate_vertices[i]
                    else:
                        i += 1
        else:
            raise ValueError("Points should be a list!")

    def add_reference_candidates(self, points):
        if isinstance(points, list):
            # v_angles = []
            for v in points:
                angle_dist = self.check_boundary_point(v)
                if angle_dist is not None:
                    i = 0
                    while i < len(self.candidate_vertices):
                        if angle_dist <= self.candidate_vertices[i][1]:
                            self.candidate_vertices.insert(i, (v, angle_dist))
                            break
                        # if angle_dist[1] <= self.candidate_vertices[i][2]:
                        #     self.candidate_vertices.insert(i, (v, angle_dist[0], angle_dist[1]))
                        #     break
                        i += 1
                    else:
                        self.candidate_vertices.append((v, angle_dist))
                    # v_angles.append([v, angle, dist])
            # self.test_candidate_vertices.extend(sorted(v_angles, key=lambda x: x[1]))
        else:
            raise ValueError("Points should be a list!")

    def check_boundary_point(self, vertex, index=None):
        if index is None:
            index = self.updated_boundary.vertices.index(vertex)

        if self.num_ref_neighbor % 2 != 0:
            raise ValueError('Wrong number of reference neighbors')
        sum_angle = 0
        if self.num_ref_neighbor // 2 == 2:
            lam = 0.618
            weights = [lam, 1-lam]
        else:
            weights = [2/self.num_ref_neighbor for i in range(self.num_ref_neighbor // 2)]

        for i in range(self.num_ref_neighbor // 2):
            clockwise_angle = vertex.to_find_clockwise_angle(
                self.updated_boundary.vertices[(index + 1 + i) % len(self.updated_boundary.vertices)],
                self.updated_boundary.vertices[index - 1 - i]
            )
            if i == 0:
                if clockwise_angle >= self.maximum_reference_angle or clockwise_angle == 0:  # 175
                    return
            sum_angle += clockwise_angle * weights[i]

        return math.degrees(sum_angle)

    def find_reference_candidates(self, target_angle):
        candidate_vertices = []
        for i, vertex in enumerate(self.updated_boundary.vertices):
            angle_dist = self.check_boundary_point(vertex, i)
            if angle_dist is not None:
                candidate_vertices.append((vertex, angle_dist))
        self.candidate_vertices = sorted(candidate_vertices, key=lambda x: math.fabs(x[1] - target_angle))

    def find_reference_point(self, not_valid_points=None, target_angle=0):
        if self.candidate_vertices is None:
            self.find_reference_candidates(target_angle)

        assert self.candidate_vertices is not None
        if len(self.candidate_vertices):
            if not_valid_points:
                for v in self.candidate_vertices:
                    if not self.is_vertex_inside_list(v[0], not_valid_points):
                        return v[0]
            else:
                return self.candidate_vertices[0][0]

    def compute_boundary_quality(self, add_v):
        index = self.updated_boundary.vertices.index(add_v)
        angles = []
        # product = 1
        for i in [1, -1]:
            angle = self.updated_boundary.vertices[(index + i) % len(self.updated_boundary.vertices)].to_find_clockwise_angle(
                self.updated_boundary.vertices[(index + i + 1) % len(self.updated_boundary.vertices)],
                self.updated_boundary.vertices[index + i - 1])
            if angle < math.pi / 3:
                angles.append(angle)
                # product *= 3 * angle / math.pi
        # return math.pow(product, 1 / 2)
        q1 = 3 * min(angles) / math.pi if len(angles) else 1
        # q1 = product

        close_vs = []
        dist = add_v.distance_to(self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)]) + add_v.distance_to(self.updated_boundary.vertices[index - 1])
        for i, v in enumerate(self.updated_boundary.vertices):
            if v in [self.updated_boundary.vertices[index],
                     self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                     self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                     self.updated_boundary.vertices[index - 1],
                     self.updated_boundary.vertices[index - 2]]:
                continue
            if add_v.distance_to(v) < dist:
                if i - 1 in close_vs:
                    continue
                close_vs.append(i)
        dists = []
        for i in close_vs:
            seg = Segment(self.updated_boundary.vertices[(i + 1) % len(self.updated_boundary.vertices)],
                          self.updated_boundary.vertices[i])
            dists.append(seg.distance(add_v))

        # compute smoothness

        targt_len = dist / 2

        _dists = [(index + i) % len(self.updated_boundary.vertices) for i in range(-2, 3)]
        mean_dist = sum([self.updated_boundary.vertices[_dists[i]].distance_to(
            self.updated_boundary.vertices[_dists[i + 1]]) for i in range(len(_dists) - 1)]) / (len(_dists) - 1)

        smoothness = min(mean_dist, targt_len) / max(mean_dist, targt_len)

        if len(dists):
            m_d = min(dists)
            q2 = m_d / (0.5 * dist) if m_d < 0.5 * dist else 1
        else:
            q2 = 1

        # print(f"Smoothness: {smoothness}; Angle: {q1}; Dist: {q2}; Boundary quality: {smoothness * q1 * q2}")
        pow = 1/3
        return math.pow(smoothness * q1 * q2, pow)

    def compute_ele_boundary_quality(self, element):
        new_vs = [v for v in element.vertices
                  if len(v.get_connected_vertices()) == 2 and v in self.updated_boundary.vertices]

        if len(new_vs):
            return self.compute_boundary_quality(new_vs[0]) # * self.compute_boundary_narrowness(new_vs[0])
        else:
            targt_vs = [v for v in element.vertices if v in self.updated_boundary.vertices]

            if not len(targt_vs):
                print("No enough vertices to compute boundary quality!")
                return 1

            angles, dists = [], []
            # product = 1
            for i, v in enumerate(targt_vs):
                index = self.updated_boundary.vertices.index(v)
                angle = v.to_find_clockwise_angle(
                    self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[index - 1])
                if angle < math.pi / 3:
                    angles.append(angle)
                    # product *= 2 * angle / math.pi
            # return math.pow(product, 1 / len(targt_vs))
            # compute the smoothness of surrounding segments
            index_1, index_r = self.updated_boundary.vertices.index(targt_vs[0]), \
                               self.updated_boundary.vertices.index(targt_vs[1])
            index = index_1 if index_1 < index_r else index_r

            targt_len = targt_vs[0].distance_to(targt_vs[1])

            dists = [(index + i) % len(self.updated_boundary.vertices) for i in range(-2, 4)]
            mean_dist = sum([
                self.updated_boundary.vertices[dists[i]].distance_to(
                    self.updated_boundary.vertices[dists[i+1]]
                )
                for i in range(len(dists) - 1)
            ]) / (len(dists) - 1)

            smoothness = min(mean_dist, targt_len) / max(mean_dist, targt_len)
            angle_quality = 3 * min(angles) / math.pi if len(angles) else 1
            # print(f"Smoothness: {smoothness}, Angle: {angle_quality}; Boundary quality: {smoothness * angle_quality}")
            # print(f"Boundary quality 2: {2 * angle_quality / 3 + smoothness / 3}")
            # print(f"Boundary quality 3: {math.pow(angle_quality * smoothness, 1/2)}")
            pow = 1/2
            return math.pow(angle_quality * smoothness, pow)

    def is_vertex_inside_list(self, vertex, points):
        return any(p.distance_to(vertex) < 0.001 for p in points)

    def count_segts_in_boundary(self, vertex):
        count = 0
        for seg in vertex.segments:
            if seg.point1 in self.updated_boundary.vertices and seg.point2 in self.updated_boundary.vertices:
                count += 1
        return count

    def get_neighbors(self, reference_point, num_points=4):
        return self.updated_boundary.get_neighbors(reference_point, num_points=num_points)

    def check_intersection_with_boundary(self, mesh, reference_point):
        max_dist = max([reference_point.distance_to(v) for v in mesh.vertices if v is not reference_point])
        neighboring_vertices = [v for v in self.updated_boundary.vertices
                                if reference_point.distance_to(v) < max_dist and v not in mesh.vertices]

        _index = mesh.vertices.index(reference_point)
        checking_sesg = [Segment(mesh.vertices[_index - 1], mesh.vertices[_index - 2]),
                         Segment(mesh.vertices[_index - 2], mesh.vertices[_index - 3])]
        for v in neighboring_vertices:
            index = self.updated_boundary.vertices.index(v)
            for c_g in checking_sesg:
                if self.updated_boundary.vertices[index - 1] not in mesh.vertices:
                    if c_g.is_cross(Segment(v, self.updated_boundary.vertices[index - 1])):
                        return True

                if self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)] not in mesh.vertices:
                    if c_g.is_cross(Segment(v, self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)])):
                        return True

        return False

    def validate_mesh(self, mesh, quality_method=0):
        return mesh.is_valid(quality_method)

    def is_point_inside_area(self, vertex):
        remote_dist = 10000
        ray_segment = Segment(vertex, Vertex(remote_dist, vertex.y))
        return self.is_inside(ray_segment)

    def find_related_meshes(self, vertex):
        return list({m for m in self.generated_meshes if vertex in m.vertices})

    def update_boundary(self, reference_point, mesh):
        new_vertices = []
        for v in mesh.vertices:
            if v not in self.updated_boundary.vertices:
                new_vertices.append(v)

        if len(new_vertices) == 1:
            id = self.updated_boundary.vertices.index(mesh.vertices[mesh.vertices.index(new_vertices[0]) - 2])
            self.updated_boundary.vertices.insert(id, new_vertices[0])
            self.remove_point(mesh.vertices[mesh.vertices.index(new_vertices[0]) - 2])
            if not new_vertices[0] in self.boundary.vertices:
                self.boundary.vertices.append(new_vertices[0])
            # update candidate reference points
            ref_neighbors = []
            [ref_neighbors.extend([
                    self.updated_boundary.vertices[(id + i + 1) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[id - i - 1]
                ])
                for i in range(self.num_ref_neighbor // 2)
            ]
            self.remove_reference_candidates(ref_neighbors + [mesh.vertices[mesh.vertices.index(new_vertices[0]) - 2]])
            self.add_reference_candidates(ref_neighbors)

            self.rp_index += 1

        elif len(new_vertices) == 0:
            removable_vertices = []
            for v in mesh.vertices:
                if self.count_segts_in_boundary(v) < 3:
                    removable_vertices.append(v)
            for v in removable_vertices:
                self.updated_boundary.vertices.remove(v)

            # update candidate reference points
            id = max([self.updated_boundary.vertices.index(v) for v in mesh.vertices
                      if v not in removable_vertices])
            ref_neighbors = []
            [ref_neighbors.extend([
                    self.updated_boundary.vertices[(id + i) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[id - i - 1]
                ])
                for i in range(self.num_ref_neighbor // 2)
            ]

            self.remove_reference_candidates(removable_vertices + ref_neighbors)
            self.add_reference_candidates(ref_neighbors)

            self.rp_index = max([self.updated_boundary.vertices.index(v) for v in mesh.vertices
                                 if v not in removable_vertices])

    def estimate_area_range(self):
        lengths = [l[1] for l in self.boundary.sort_segments_by_length()]

        # min_L = lengths[0]
        # return min_L ** 2, 1.2 * min_L ** 2
        L = sum(lengths) / len(lengths)
        max_L = min(lengths[-2], 2 * L)
        min_L = min(L / math.sqrt(2), lengths[1])
        # if len(self.candidate_vertices):
        #     angle = self.candidate_vertices[0][1]
        # else:
        #     raise ValueError('Empty angle in the candidate vertices!')
        # return min_L ** 2, ((max_L + 3 * min_L) / 4) ** 2, L
        return min_L, (max_L + 3 * min_L) / 4

    def smooth_pave(self, vertices, current_boundary_vertices, lr_1=None, lr_2=None, iteration=400, interior=False):
        # self.smooth_current_boundary(current_boundary_vertices, lr_1=lr_1, lr_2=lr_2, iteration=iteration)
        if not interior:
            self.smooth_current_boundary_3()
        self.smooth_fixed_vertices([v for v in vertices if v not in current_boundary_vertices], iteration)
        self.find_reference_candidates(target_angle=0)

    def middle_vertex(self, vertex, left_v, right_v, target_angle):
        m_v = (left_v + right_v) / 2
        A = right_v.x - left_v.x
        B = right_v.y - left_v.y
        D = left_v.distance_to(m_v) / math.tan(math.radians(target_angle / 2))

        if B == 0:
            x1, x2 = m_v.x, m_v.x
            y1, y2 = m_v.y + D, m_v.y - D
        elif A == 0:
            x1, x2 = m_v.x + D, m_v.x - D
            y1, y2 = m_v.y, m_v.y
        else:
            M = -A / B
            N = A * m_v.x / B + m_v.y

            x1, x2 = (-2 * M * N + 2 * m_v.x + 2 * M * m_v.y + math.sqrt(math.fabs((-2 * M * N + 2 * m_v.x + 2 * M * m_v.y) ** 2 - 4 * (M ** 2 + 1) * ((N - m_v.y) ** 2 + m_v.x ** 2 - D ** 2)))) / (2 * (M ** 2 + 1)), (-2 * M * N + 2 * m_v.x + 2 * M * m_v.y - math.sqrt(math.fabs((-2 * M * N + 2 * m_v.x + 2 * M * m_v.y) ** 2 - 4 * (M ** 2 + 1) * ((N - m_v.y) ** 2 + m_v.x ** 2 - D ** 2)))) / (2 * (M ** 2 + 1))
            y1, y2 = M * x1 + N, M * x2 + N
        V1, V2 = Vertex(x1, y1), Vertex(x2, y2)
        if V1.distance_to(vertex) < V2.distance_to(vertex):
            return V1
        else:
            return V2

    def side_vertex(self, vertex, next_v, nn_v, angle, dist):
        a = next_v.x
        b = next_v.y
        A = nn_v.x - next_v.x
        B = nn_v.y - next_v.y
        W = dist * next_v.distance_to(nn_v) * math.cos(math.radians(angle))
        if B == 0:
            x1, x2 = W/A + a, W/A + a
            y1, y2 = b + math.sqrt(dist**2-(W/A)**2), b - math.sqrt(dist**2-(W/A)**2)
        elif A == 0:
            x1, x2 = a + math.sqrt(dist**2-(W/B)**2), a - math.sqrt(dist**2-(W/B)**2)
            y1, y2 = W/B + b, W/B + b
        else:
            M = -A / B
            N = (W + A * a + B * b) / B
            x1, x2 = (2 * M * b - 2 * M * N + 2 * a + math.sqrt(math.fabs((2 * M * b - 2 * M * N + 2 * a) ** 2 - 4 * (M ** 2 + 1) * ((N - b) ** 2 + a ** 2 - dist ** 2)))) / (2 * (M ** 2 + 1)), (2 * M * b - 2 * M * N + 2 * a - math.sqrt(math.fabs((2 * M * b - 2 * M * N + 2 * a) ** 2 - 4 * (M ** 2 + 1) * ((N - b) ** 2 + a ** 2 - dist ** 2)))) / (2 * (M ** 2 + 1))
            y1, y2 = M * x1 + N, M * x2 + N
        V1, V2 = Vertex(x1, y1), Vertex(x2, y2)
        if V1.distance_to(vertex) < V2.distance_to(vertex):
            return V1
        else:
            return V2

    def inner_vertex(self, vertex, angle):
        index = self.updated_boundary.vertices.index(vertex)
        left_v = self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)]
        right_v = self.updated_boundary.vertices[index - 1]

        m_v = (left_v + right_v) / 2
        d = m_v.distance_to(right_v) * math.tan(math.radians(angle))
        a = m_v.x
        b = m_v.y
        A = vertex.x - m_v.x
        B = vertex.y - m_v.y
        s = math.sqrt(d ** 2 / (A ** 2 + B ** 2))
        n_v = Vertex(a + s * A, b + s * B)
        return n_v

    def indention_vertex(self, vertex, left_v, right_v, angle, dist):
        a = vertex.x
        b = vertex.y
        A = left_v.x - vertex.x
        B = left_v.y - vertex.y
        W = dist * vertex.distance_to(left_v) * math.cos(math.radians(angle))

        if B == 0:
            x1, x2 = W/A + a, W/A + a
            y1, y2 = b + math.sqrt(dist**2-(W/A)**2), b - math.sqrt(dist**2-(W/A)**2)
        elif A == 0:
            x1, x2 = a + math.sqrt(dist**2-(W/B)**2), a - math.sqrt(dist**2-(W/B)**2)
            y1, y2 = W/B + b, W/B + b
        else:
            M = -A / B
            N = (W + A * a + B * b) / B
            x1, x2 = (2 * M * b - 2 * M * N + 2 * a + math.sqrt(math.fabs((2 * M * b - 2 * M * N + 2 * a) ** 2 - 4 * (M ** 2 + 1) * ((N - b) ** 2 + a ** 2 - dist ** 2)))) / (2 * (M ** 2 + 1)), (2 * M * b - 2 * M * N + 2 * a - math.sqrt(math.fabs((2 * M * b - 2 * M * N + 2 * a) ** 2 - 4 * (M ** 2 + 1) * ((N - b) ** 2 + a ** 2 - dist ** 2)))) / (2 * (M ** 2 + 1))
            y1, y2 = M * x1 + N, M * x2 + N
        V1, V2 = Vertex(x1, y1), Vertex(x2, y2)
        if V1.to_find_clockwise_angle(left_v, right_v) < V2.to_find_clockwise_angle(left_v, right_v):
            return V1
        else:
            return V2

    def find_side_vertex(self, vertex, _next_v, next_v, nn_v, v_angle):
        dist = (vertex.distance_to(_next_v) + vertex.distance_to(next_v) + next_v.distance_to(nn_v)) / 3

        target_angle = 45
        failed = False
        while True:
            n_v = self.side_vertex(vertex, next_v, nn_v, target_angle, dist)

            connected_vs = vertex.get_connected_vertices()
            if target_angle <= v_angle:
                failed = True
                break
            clockwise_boundaey = self.clockwise_vertices(vertex, connected_vs)
            if self.is_inside_boundary(vertex, n_v, clockwise_boundaey, _next_v, next_v):
                break
            else:
                target_angle -= 5

        if not failed:
            return n_v
        return vertex

    def smooth_current_boundary_3(self):
        i = 0
        while i < len(self.updated_boundary.vertices):

            index = i
            if self.updated_boundary.vertices[index] in self.original_vertices:
                i += 1
                continue

            v_angle = math.degrees(self.updated_boundary.vertices[index].to_find_clockwise_angle(
                self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                self.updated_boundary.vertices[index - 1]
            ))

            if v_angle <= 90:
                target_angle = v_angle if v_angle >= 45 else 45
                failed = False
                while True:
                    new_v = self.middle_vertex(
                        self.updated_boundary.vertices[index],
                        self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                        self.updated_boundary.vertices[index - 1],
                        target_angle
                    )
                    # print('Target angle:', target_angle)
                    connected_vs = self.updated_boundary.vertices[index].get_connected_vertices()
                    if target_angle >= 135:
                        failed = True
                        break
                    clockwise_boundaey = self.clockwise_vertices(self.updated_boundary.vertices[index], connected_vs)
                    if self.is_inside_boundary(
                        self.updated_boundary.vertices[index],
                        new_v,
                        clockwise_boundaey,
                        self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                        self.updated_boundary.vertices[index - 1]
                    ):
                        break
                    else:
                        target_angle += 5
                if not failed:
                    self.updated_boundary.vertices[index].x = new_v.x
                    self.updated_boundary.vertices[index].y = new_v.y

            elif 90 < v_angle <= 180:
                left_angle = self.updated_boundary.compute_boundary_angle(
                    self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)]
                )
                right_angle = self.updated_boundary.compute_boundary_angle(
                    self.updated_boundary.vertices[index - 1]
                )

                if right_angle < 45:
                    n_v = self.find_side_vertex(self.updated_boundary.vertices[index],
                                                self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                                                self.updated_boundary.vertices[index - 1],
                                                self.updated_boundary.vertices[index - 2], right_angle)

                    self.updated_boundary.vertices[index].x = n_v.x
                    self.updated_boundary.vertices[index].y = n_v.y
                elif left_angle < 45:
                    n_v = self.find_side_vertex(self.updated_boundary.vertices[index],
                                                self.updated_boundary.vertices[index - 1],
                                                self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                                                self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)], left_angle)

                    self.updated_boundary.vertices[index].x = n_v.x
                    self.updated_boundary.vertices[index].y = n_v.y

                else:
                    n_v = self.find_indention_vertex(self.updated_boundary.vertices[index], v_angle)
                    self.updated_boundary.vertices[index].x = n_v.x
                    self.updated_boundary.vertices[index].y = n_v.y

            elif 180 < v_angle <= 270:
                n_v = self.find_indention_vertex(self.updated_boundary.vertices[index], v_angle)
                self.updated_boundary.vertices[index].x = n_v.x
                self.updated_boundary.vertices[index].y = n_v.y

            else:
                n_v = self.inner_vertex(self.updated_boundary.vertices[index], 45)
                self.updated_boundary.vertices[index].x = n_v.x
                self.updated_boundary.vertices[index].y = n_v.y

                n_v = self.find_indention_vertex(self.updated_boundary.vertices[index], v_angle)
                self.updated_boundary.vertices[index].x = n_v.x
                self.updated_boundary.vertices[index].y = n_v.y

            i += 1

    def find_indention_vertex(self, vertex, v_angle):
        index = self.updated_boundary.vertices.index(vertex)
        left_v = self.updated_boundary.vertices[
            (index + 1) % len(
                self.updated_boundary.vertices)]
        right_v = self.updated_boundary.vertices[index - 1]
        dist = (vertex.distance_to(left_v) + vertex.distance_to(right_v)) / 2

        c_neighbors = Boundary2D.get_closet_points(
            self.updated_boundary.vertices,
            self.updated_boundary.vertices[index],
            [
                self.updated_boundary.vertices[index - 2],
                right_v,
                left_v,
                self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)]
            ],
            dist
        )

        neighbors = self.find_closest_segments(self.updated_boundary, vertex, dist)

        if len(neighbors) or len(c_neighbors):
            failed = False
            times = 4
            while True:
                n_v = self.indention_vertex(vertex, left_v, right_v, (360 - v_angle) / 2, dist / times)
                # print('Target angle:', target_angle)
                connected_vs = vertex.get_connected_vertices()
                if times >= 10:
                    failed = True
                    break
                clockwise_boundaey = self.clockwise_vertices(vertex, connected_vs)
                if self.is_inside_boundary(vertex, n_v, clockwise_boundaey, left_v, right_v):
                    break
                else:
                    times += 1
            if not failed:
                return n_v
        return vertex

    def is_inside_boundary(self, original_v, vertex, boundary, left_v, right_v):
        for i in range(len(boundary)):
            if left_v in [boundary[i], boundary[i-1]] and \
                    right_v in [boundary[i], boundary[i-1]]:
                continue

            if (vertex.to_find_clockwise_angle(boundary[i], boundary[i-1]) < math.pi) != \
                (original_v.to_find_clockwise_angle(boundary[i], boundary[i - 1]) < math.pi):
                return False
        return True

    def clockwise_vertices(self, inner_v, vertices):
        for i in range(1, len(vertices)):
            max_angle = -1
            flag = i
            for j in range(i, len(vertices)):
                current_angle = inner_v.to_find_clockwise_angle(vertices[j], vertices[i-1])
                if current_angle > max_angle:
                    max_angle = current_angle
                    flag = j
            if flag != i:
                vertices[i], vertices[flag] = vertices[flag], vertices[i]
        final_vertices = []
        for i in range(len(vertices)):
            inter_v = [v for v in vertices[i].get_connected_vertices()
                       if v in vertices[i-1].get_connected_vertices() and v is not inner_v]
            if len(inter_v):
                final_vertices.append(vertices[i-1])
                final_vertices.append(inter_v[0])
            else:
                final_vertices.append(vertices[i-1])

        return final_vertices

    def find_closest_segments(self, boundary, vertex, dist):
        closet_segments = []
        for i in range(len(boundary.vertices)):
            if vertex in [boundary.vertices[i-1], boundary.vertices[i]]:
                continue
            s = Segment(boundary.vertices[i-1], boundary.vertices[i])
            _, _dist, inner = s.perpendicular_point(vertex)
            if inner and _dist <= dist:
                closet_segments.append(s)
        return closet_segments

    def smooth_fixed_vertices(self, vertices, iteration):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        #  modified after running, needs to be checked later on
        while diffs > 0.001 and i_iteration < iteration:
            # while diffs > 0.001:
            i_iteration += 1
            new_sum_coordinates = 0
            for vertex in vertices:
                if vertex in self.original_vertices:
                    continue
                x = 0
                y = 0
                count = 0
                connected_vertices = vertex.get_connected_vertices()

                for connect_v in connected_vertices:
                    x += connect_v.x + vertex.x
                    y += connect_v.y + vertex.y
                    count += 1
                if count == 0:
                    continue
                vertex.x = x / (2 * count)
                vertex.y = y / (2 * count)

                new_sum_coordinates += vertex.x + vertex.y

            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates
        print(f"Smoothing fixed vertices,Iteration numbers: {i_iteration}, the diff of smoothing is {diffs}!")

    def smooth(self, vertices, lr_1=0.999, lr_2=0.999, iteration=400):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        #  modified after running, needs to be checked later on
        while diffs > 0.001 and i_iteration < iteration:
            # while diffs > 0.001:
            i_iteration += 1
            new_sum_coordinates = 0
            for vertex in vertices:
                if vertex in self.original_vertices:
                    continue
                x = 0
                y = 0
                count = 0
                connected_vertices = vertex.get_connected_vertices()
                near_meshes = self.find_related_meshes(vertex)
                if len(near_meshes) == 1:
                    lr = lr_1
                    if len(connected_vertices) == 2:
                        origins = connected_vertices[0].get_common_vertex(connected_vertices[1])
                        origin = [v for v in origins if v is not vertex]
                        origin = Boundary2D.compute_dist(origin, vertex)[0][0]

                        p_dist = Boundary2D.compute_dist(
                            [
                                v for v in self.updated_boundary.vertices
                                if v not in connected_vertices and v is not vertex
                            ],
                            origin
                        )
                        if not len(p_dist):
                            continue
                        cloest_p, dist = p_dist[0]

                        estimate_vertex = Mesh.estimate_4th_vertex(origin, connected_vertices[0], connected_vertices[1], suggest_dist=dist)

                        vertex.x = estimate_vertex.x
                        vertex.y = estimate_vertex.y


                    else:
                        for connect_v in connected_vertices:
                            vertex.x = lr * vertex.x + (1 - lr) * connect_v.x
                            vertex.y = lr * vertex.y + (1 - lr) * connect_v.y

                elif len(near_meshes) == 2:
                    update_boundary_vertices = []
                    inside_vertices = []
                    [update_boundary_vertices.append(v) if v in self.updated_boundary.vertices
                     else inside_vertices.append(v) for v in connected_vertices]

                    if len(inside_vertices) == 1 and len(update_boundary_vertices) == 2:
                        inside_vertex = inside_vertices[0]

                        origins = inside_vertex.get_common_vertex(update_boundary_vertices[0])
                        common_v1s = [v for v in origins if v is not vertex]
                        common_v1s = Boundary2D.compute_dist(common_v1s, vertex)[0]

                        origins = inside_vertex.get_common_vertex(update_boundary_vertices[1])
                        common_v2s = [v for v in origins if v is not vertex]
                        common_v2s = Boundary2D.compute_dist(common_v2s, vertex)[0]

                        common_v1 = common_v1s[0]
                        common_v2 = common_v2s[0]

                        estimate_vertex_1 = Mesh.estimate_4th_vertex(common_v1, update_boundary_vertices[0], inside_vertex, factor=0.7)

                        estimate_vertex_2 = Mesh.estimate_4th_vertex(common_v2, update_boundary_vertices[1], inside_vertex, factor=0.7)

                        vertex.x = (estimate_vertex_1.x + estimate_vertex_2.x) / 2
                        vertex.y = (estimate_vertex_1.y + estimate_vertex_2.y) / 2

                    else:
                        lr = lr_2
                        for connect_v in connected_vertices:
                            vertex.x = lr * vertex.x + (1 - lr) * connect_v.x
                            vertex.y = lr * vertex.y + (1 - lr) * connect_v.y
                else:
                    for connect_v in connected_vertices:
                        x += connect_v.x + vertex.x
                        y += connect_v.y + vertex.y
                        count += 1
                    if count == 0:
                        continue
                    vertex.x = x / (2 * count)
                    vertex.y = y / (2 * count)
            for vertex in vertices:
                new_sum_coordinates += vertex.x + vertex.y
            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates
        print(f"Iteration numbers: {i_iteration}, the diff of smoothing is {diffs}!")

        # update reference points
        self.find_reference_candidates(target_angle=0)

    def get_nodes(self, root, exclusion, layer, path, paths, N):
        if root is None:
            return

        if len(path) < N:
            path.append(root)
        else:
            path[-layer-1] = root
        if layer == 0:
            paths.append([v for v in path])
            return
        else:
            nodes = [v for v in root.get_connected_vertices() if v not in exclusion and v not in path[:N-layer]]
            for i in range(len(nodes)):
                self.get_nodes(nodes[i], exclusion, layer-1, path, paths, N)

    def extract_samples_2(self, meshes, n_neighbor, n_radius, radius, index=1, quality_threshold=0.7):
        all_samples, outputs, types = [], [], []
        for id, element in enumerate(meshes):
            print(f"Extracting element {id} out of {len(meshes)}")
            if self.get_quality(element, index=index) >= quality_threshold:
                for i in range(4):
                    rp = element.vertices[i]
                    l_p = element.vertices[(i + 1) % 4]
                    r_p = element.vertices[(i - 1)]
                    target = element.vertices[(i - 2)]
                    l_p_path, l_p_paths, exclusion = [], [], [rp, r_p]
                    self.get_nodes(l_p, exclusion, n_neighbor - 1, l_p_path, l_p_paths, n_neighbor)

                    r_p_path, r_p_paths, exclusion = [], [], [rp, l_p]
                    self.get_nodes(r_p, exclusion, n_neighbor - 1, r_p_path, r_p_paths, n_neighbor)

                    radius_neighbors = self.get_radius_neighbors(rp, l_p, r_p, [rp, l_p, r_p, target], radius=radius, N=n_radius)

                    samples = list(itertools.product(
                        [
                            _path for _path in r_p_paths
                            if len(_path) == n_neighbor
                        ],
                        radius_neighbors,
                        [
                            _path for _path in l_p_paths
                            if len(_path) == n_neighbor
                        ]
                    ))

                    for rr, mm, ll in samples:
                        if target in rr and target in ll:
                            continue
                        _sample = []
                        base_length = (rp.distance_to(rr[0]) +
                            sum([rr[j].distance_to(rr[j-1]) for j in range(1, len(rr))]) +
                            sum([ll[j].distance_to(ll[j-1]) for j in range(1, len(ll))]) +
                            rp.distance_to(ll[0])) / (2 * n_neighbor)

                        [_sample.extend([rp.distance_to(p) / (base_length * radius),
                                         rp.to_find_clockwise_angle(p, r_p) % round(2 * math.pi, 4)])
                            for p in rr]
                        [_sample.extend([rp.distance_to(p) / (base_length * radius),
                                         rp.to_find_clockwise_angle(p, r_p) % round(2 * math.pi, 4)])
                            for p in mm]
                        [_sample.extend([rp.distance_to(p) / (base_length * radius),
                                         rp.to_find_clockwise_angle(p, r_p) % round(2 * math.pi, 4)])
                            for p in reversed(ll)]
                        _target = [rp.distance_to(target) / (base_length * radius),
                                   rp.to_find_clockwise_angle(target, r_p) % round(2 * math.pi, 4)]

                        if target in rr:
                            types.append([1])
                        elif target in ll:
                            types.append([0])
                        else:
                            types.append([0.5])
                        all_samples.append(_sample)
                        outputs.append(_target)
        print("Done!")
        return all_samples, types, outputs

    def get_radius_neighbors(self, base_point, start_point, end_point, exclusion, radius, N=3):
        def radius_neighbors_with_angle(start_angle, end_angle):
            # find closest point in circle
            base_length = radius * (0.5 * base_point.distance_to(start_point) + 0.5 * base_point.distance_to(end_point))
            closet_neighbors = self.boundary.get_closet_points(
                self.boundary.get_points_within_angle(
                    self.boundary.vertices, base_point, start_point, start_angle, end_angle
                ),
                base_point,
                exclusion=exclusion,
                S_T=base_length
            )

            _angle = base_point.to_find_clockwise_angle(start_point, Vertex(base_point.x + 1, base_point.y))

            closet_neighbors.append(base_point +
                                    Vertex(base_length * math.cos(_angle - (start_angle + end_angle) / 2),
                                           base_length * math.sin((_angle - (start_angle + end_angle) / 2))))
            return closet_neighbors

        angle = base_point.to_find_clockwise_angle(start_point, end_point)
        angles = [i * angle / N for i in range(N+1)]
        neighbors = [radius_neighbors_with_angle(angles[i-1], angles[i]) for i in range(1, N+1)]
        # left_neighbors = radius_neighbors_with_angle(0.01, angle / 3)
        # middle_neighbors = radius_neighbors_with_angle(angle / 3, 2 * angle / 3)
        # right_neighbors = radius_neighbors_with_angle(2 * angle / 3, angle * 0.99)
        all_combinations = list(itertools.product(*reversed(neighbors)))
        return all_combinations

    def save_samples(self, file_name, res, _type=1):
        if _type == 1:
            res['samples'] = [self.points_as_array(s) for s in res['samples']]
            res['outputs'] = [self.points_as_array(s) for s in res['outputs']]
        with open(file_name, 'w') as fw:
            json.dump(res, fw)

    def points_as_array(self, points):
        flated_points = []
        for point in points:
            flated_points.append(point.x)
            flated_points.append(point.y)
        return flated_points

    def remove_point(self, point):
        self.updated_boundary.vertices.remove(point)

    def compute_element_quality(self, element):
        q1, q2 = element.get_quality_3()
        return math.pow(q1 * q2, 1/2)

    def get_quality(self, element, index=0):
        if index == 0:
            return element.get_quality()
        elif index == 1:
            return self.compute_element_quality(element)
        elif index == 2:
            b_reward = self.compute_ele_boundary_quality(element)
            # e_reward = self.compute_element_quality(element)
            e_reward = element.get_quality(type='robust')
            # print(f'boundary quality: {b_reward}; element quality: {e_reward}')
            # print(2 * e_reward / 9, 4 * (b_reward - 1) / 9)
            # return math.sqrt(e_reward * b_reward)
            return e_reward + 1 * (b_reward - 1)
            # return e_reward * b_reward
        elif index == 3:
            return element.get_quality(type='stretch')
        elif index == 4:
            return element.get_quality(type='robust')
        elif index == 5:
            return element.get_quality(type='strong')
        elif index == 6:
            b_reward = self.compute_ele_boundary_quality(element)
            # e_reward = self.compute_element_quality(element)
            e_reward = element.get_quality(type='area')
            print(f'boundary quality: {b_reward}; element quality: {e_reward}')
            # print(2 * e_reward / 9, 4 * (b_reward - 1) / 9)
            # return math.sqrt(e_reward * b_reward)
            return e_reward + 1 * (b_reward - 1)
        raise ValueError(f"Unknown quality index: {index}")

    def generate_meshes_canvas(self, meshes, quality, indexing, type, style):
        self.boundary.plot(style=style, linewidth=1)
        for id, m in enumerate(meshes):
            center = m.get_centriod(diff=True)
            if quality and indexing:
                _quality = round(self.get_quality(element=m, index=type), 4)
                plt.text(center.x, center.y, f"{id}; {_quality}", fontsize=6)
            elif quality:
                _quality = round(self.get_quality(element=m, index=type), 4)
                plt.text(center.x, center.y, str(_quality), fontsize=6)
            elif indexing:
                plt.text(center.x, center.y, str(id), fontsize=4)

    def save_meshes(self, name, meshes, quality=False, indexing=False, type=0, dpi=300, style='k.-'):
        plt.clf()
        self.generate_meshes_canvas(meshes, quality, indexing, type, style=style)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=-0, hspace=0, wspace=0)
        plt.savefig(name, dpi=dpi)
        plt.close('all')

    def write_generated_elements_2_file(self, filename, format='inp'):
        if len(self.generated_meshes) == 0:
            print("There are no elements generated!")
            return

        nodes = list(self.original_vertices)  # boundary nodes (referenced by the B21 edge elements)
        for ele in self.generated_meshes:
            nodes.extend([n for n in ele.vertices if n not in nodes])

        with open(filename, 'w') as fw:
            fw.write("*NODE, NSET=ALLNODES\n")
            for id, node in enumerate(nodes):
                fw.write(f"{id+1}, {node.x}, {node.y}" + "\n")

            i = 0
            for i in range(1, len(self.original_vertices)):
                fw.write(f'*ELEMENT, TYPE=B21, ELSET=EB{i}\n {i+1}, {nodes.index(self.original_vertices[i-1]) + 1}, {nodes.index(self.original_vertices[i]) + 1}\n')
            fw.write(f'*ELEMENT, TYPE=S4R, ELSET=EB{i+1} \n')
            for id, ele in enumerate(self.generated_meshes):
                fw.write(f"{id+1}, {nodes.index(ele.vertices[0]) + 1}, "
                         f"{nodes.index(ele.vertices[1]) + 1}, "
                         f"{nodes.index(ele.vertices[2]) + 1}, "
                         f"{nodes.index(ele.vertices[3]) + 1}" + "\n")
        print("Document writing is finished!")

def connect_vertices(points):
    for i in range(len(points)):
        segmt = Segment(points[i - 1], points[i])
        points[i - 1].assign_segment(segmt)
        points[i].assign_segment(segmt)
