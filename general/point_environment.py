import math
from typing import Any

import numpy as np

from general.components import Vertex, Boundary2D, Segment


class PointEnvironment(object):
    def __init__(self, reference_point, boundary, neighbor_num=4, radius_num=3,
                 average_edge_length=1, area_ratio=1, radius=6, static=False):
        self.reference_point = reference_point
        self.neighbors: Any = None
        self.radius_neighbors: Any = None
        self.base_length: Any = None
        self.available_radius: Any = None
        self.state: Any = None
        self.boundary = boundary
        self.neighbor_num = neighbor_num
        self.radius_num = radius_num
        self.average_edge_length =average_edge_length
        self.theta = 0
        self.radius = radius
        self.state_vertices = [None for i in range(self.neighbor_num + self.radius_num)]
        # self.get_available_radius()
        self.area_ratio = area_ratio
        self.static = static
        self.get_state(state_type=2)

    def get_neighbors(self, boundary):
        vertices = boundary.get_neighbors(self.reference_point, num_points=self.neighbor_num)
        self.neighbors = vertices
        self.base_length = round(sum([vertices[i].distance_to(vertices[i-1])
                                      for i in range(1, len(vertices))]) / self.neighbor_num, 4)

            # def get_available_radius(self):
    #     if len(self.radius_neighbors):
    #         all_distance = [self.reference_point.distance_to(v) / 2 for v in self.radius_neighbors]
    #         self.available_radius = round(min(all_distance) / self.base_length, 1)
    #     else:
    #         self.available_radius = self.radius

    def get_state(self, state_type=2):
        self.get_neighbors(self.boundary)
        self.state = self.get_radius_points().flatten()

    def clip_angle(self, angle, max_angle):
        return min(angle, max_angle + math.pi / 2)

    def get_radius_points(self):
        r_points = np.full([self.radius_num + self.neighbor_num, 2], 1, dtype=np.float32)
        index = self.boundary.vertices.index(self.reference_point)
        right_p = self.boundary.vertices[index - 1]
        # rr_p = self.boundary.vertices[index - 2]
        left_p = self.boundary.vertices[(index + 1) % len(self.boundary.vertices)]
        # ll_p = self.boundary.vertices[(index + 2) % len(self.boundary.vertices)]
        target_length = self.base_length * self.radius

        theta = self.reference_point.to_find_clockwise_angle(left_p, right_p)
        self.theta = theta

        for i in range(self.neighbor_num // 2):
            if i == 0:
                # r_points[i] = [(self.reference_point.distance_to(right_p) / self.radius) / self.base_length,
                #                self.base_length / self.average_edge_length]
                # the second value is the absolute distance between current base length and the mean length of the boundary
                if not self.static:
                    r_points[i] = [(self.reference_point.distance_to(right_p) / self.radius) / self.base_length,
                                   self.area_ratio] #self.area_ratio
                else:
                    r_points[i] = [(self.reference_point.distance_to(right_p) / self.radius) / self.base_length,
                                   0]  # self.area_ratio
                r_points[self.radius_num + self.neighbor_num - i - 1] = [
                    (self.reference_point.distance_to(left_p) / self.radius) / self.base_length,
                    theta]
                self.state_vertices[i] = right_p
                self.state_vertices[self.radius_num + self.neighbor_num - i - 1] = left_p

            else:
                _angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[index - i - 1], right_p)
                r_points[i] = [(self.reference_point.distance_to(self.boundary.vertices[index - i - 1]) / self.radius) / self.base_length,
                               _angle if _angle < math.pi else max(_angle, 1.5 * math.pi) - 2 * math.pi]
                _angle = self.reference_point.to_find_clockwise_angle(
                    self.boundary.vertices[(index + 1 + i) % len(self.boundary.vertices)], right_p)
                r_points[self.radius_num + self.neighbor_num - i - 1] = [(self.reference_point.distance_to(
                    self.boundary.vertices[(index + i + 1) % len(self.boundary.vertices)]) / self.radius) / self.base_length,
                             min(_angle, theta + math.pi / 2)]
                self.state_vertices[i] = self.boundary.vertices[index - i - 1]
                self.state_vertices[self.radius_num + self.neighbor_num - i - 1] = \
                    self.boundary.vertices[(index + i + 1) % len(self.boundary.vertices)]


        rotation_angle = self.reference_point.to_find_clockwise_angle(right_p, self.reference_point + Vertex(1, 0))
        angles = [i * theta / (2 * self.radius_num) for i in range(1, 2 * self.radius_num, 2)]
        # p_s = []
        for i, a in enumerate(angles):
            r_points[self.neighbor_num // 2 + i][1] = self.clip_angle(a, theta)
        #     p_s.append(self.reference_point + Vertex.rotate_counterclockwise(Vertex(target_length * math.cos(a),
        #                                                  target_length * math.sin(a)), rotation_angle))
        p_s = self.reference_point + Vertex.rotate_counterclockwise(Vertex(target_length * math.cos(theta / 2),
                                                         target_length * math.sin(theta / 2)), rotation_angle)
        shortest_edge = [1, 0] # dist, id

        for i in range(index - 1, index - len(self.boundary.vertices), -1):
            d = self.reference_point.distance_to(self.boundary.vertices[i])
            if self.boundary.vertices[i] in [right_p, left_p]:
                continue
            else:
                angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i], right_p)
                l_angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i + 1], right_p)

            if angle == 0:
                continue
            k = int(angle / (theta / self.radius_num))
            # h = int(l_angle / (theta / self.radius_num))
            if k < self.radius_num and d < target_length:
                if r_points[k + self.neighbor_num // 2][0] > (d / self.radius) / self.base_length:
                    r_points[k + self.neighbor_num // 2][0] = (d / self.radius) / self.base_length
                    r_points[k + self.neighbor_num // 2][1] = self.clip_angle(angle, theta)
                    self.state_vertices[k + self.neighbor_num // 2] = self.boundary.vertices[i]

            seg = Segment(self.boundary.vertices[i], self.boundary.vertices[i + 1])
            # for c_k in p_s:
            ll = Segment(self.reference_point, p_s)
            flag, vv = ll.intersection_vertex(seg)
            if flag is not None:
                if flag:
                    _d = self.reference_point.distance_to(vv)
                    if shortest_edge[0] > (_d / self.radius) / self.base_length:
                        shortest_edge[0] = (_d / self.radius) / self.base_length
                        shortest_edge[1] = i

        if shortest_edge[0] != 1 and shortest_edge[0] < r_points[(self.radius_num + self.neighbor_num) // 2][0]:
            _i = shortest_edge[1]

            for i in range(self.radius_num):
                r_points[self.neighbor_num // 2 + i] = [(self.reference_point.distance_to(
                    self.boundary.vertices[i - self.radius_num // 2 + _i]) / self.radius) / self.base_length,
                                 self.reference_point.to_find_clockwise_angle(
                                     self.boundary.vertices[i - self.radius_num // 2 + _i],
                                     right_p)]
                self.state_vertices[self.neighbor_num // 2 + i] = \
                    self.boundary.vertices[i - self.radius_num // 2 + _i]

        self.state_vertices.insert(0, self.reference_point)
        return np.asarray([[round(v[0], 4), round(v[1], 4)] for v in r_points])

    def points_as_array(self, points):
        flattened_points = []
        for point in points:
            flattened_points.append(point.x)
            flattened_points.append(point.y)
        return np.asarray(flattened_points)
