import math
from typing import Any

from general.components import *


def _circle_line_x(lin, M, N, a, b, r):
    disc = math.sqrt(math.fabs(lin ** 2 - 4 * (M ** 2 + 1) * ((N - b) ** 2 + a ** 2 - r ** 2)))
    denom = 2 * (M ** 2 + 1)
    return (lin + disc) / denom, (lin - disc) / denom


def middle_vertex(vertex, left_v, right_v, target_angle):
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

        lin = -2 * M * N + 2 * m_v.x + 2 * M * m_v.y
        x1, x2 = _circle_line_x(lin, M, N, m_v.x, m_v.y, D)
        y1, y2 = M * x1 + N, M * x2 + N
    V1, V2 = Vertex(x1, y1), Vertex(x2, y2)
    if V1.distance_to(vertex) < V2.distance_to(vertex):
        return V1
    else:
        return V2


def side_vertex(vertex, next_v, nn_v, angle, dist):
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
        lin = 2 * M * b - 2 * M * N + 2 * a
        x1, x2 = _circle_line_x(lin, M, N, a, b, dist)
        y1, y2 = M * x1 + N, M * x2 + N
    V1, V2 = Vertex(x1, y1), Vertex(x2, y2)
    if V1.distance_to(vertex) < V2.distance_to(vertex):
        return V1
    else:
        return V2


def indention_vertex(vertex, left_v, right_v, angle, dist):
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
        lin = 2 * M * b - 2 * M * N + 2 * a
        x1, x2 = _circle_line_x(lin, M, N, a, b, dist)
        y1, y2 = M * x1 + N, M * x2 + N
    V1, V2 = Vertex(x1, y1), Vertex(x2, y2)
    if V1.to_find_clockwise_angle(left_v, right_v) < V2.to_find_clockwise_angle(left_v, right_v):
        return V1
    else:
        return V2


def is_inside_boundary(original_v, vertex, boundary, left_v, right_v):
    for i in range(len(boundary)):
        if left_v in [boundary[i], boundary[i-1]] and \
                right_v in [boundary[i], boundary[i-1]]:
            continue

        if (vertex.to_find_clockwise_angle(boundary[i], boundary[i-1]) < math.pi) != \
            (original_v.to_find_clockwise_angle(boundary[i], boundary[i - 1]) < math.pi):
            return False
    return True


def clockwise_vertices(inner_v, vertices):
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


class SmoothingMixin:
    # Provided by Mesher; this mixin is only ever combined into it.
    updated_boundary: Any
    original_vertices: Any
    find_related_meshes: Any

    def smooth_pave(self, vertices, current_boundary_vertices, lr_1=None, lr_2=None, iteration=400, interior=False):
        # self.smooth_current_boundary(current_boundary_vertices, lr_1=lr_1, lr_2=lr_2, iteration=iteration)
        if not interior:
            self.smooth_current_boundary_3()
        self.smooth_fixed_vertices([v for v in vertices if v not in current_boundary_vertices], iteration)
        self.updated_boundary.find_reference_candidates(target_angle=0)

    @staticmethod
    def estimate_4th_vertex(origin, left, right, factor=0.5, suggest_dist=None):
        distance = (origin.distance_to(left) + origin.distance_to(right)) * factor
        if suggest_dist is not None:
            distance = min(distance, 0.6 * suggest_dist)
        s = Segment.get_ray_segment(Segment(origin, left), Segment(origin, right), distance)
        return s.point2

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

    def find_side_vertex(self, vertex, _next_v, next_v, nn_v, v_angle):
        dist = (vertex.distance_to(_next_v) + vertex.distance_to(next_v) + next_v.distance_to(nn_v)) / 3

        target_angle = 45
        failed = False
        while True:
            n_v = side_vertex(vertex, next_v, nn_v, target_angle, dist)

            connected_vs = vertex.get_connected_vertices()
            if target_angle <= v_angle:
                failed = True
                break
            clockwise_boundary = clockwise_vertices(vertex, connected_vs)
            if is_inside_boundary(vertex, n_v, clockwise_boundary, _next_v, next_v):
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
                    new_v = middle_vertex(
                        self.updated_boundary.vertices[index],
                        self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                        self.updated_boundary.vertices[index - 1],
                        target_angle
                    )
                    connected_vs = self.updated_boundary.vertices[index].get_connected_vertices()
                    if target_angle >= 135:
                        failed = True
                        break
                    clockwise_boundary = clockwise_vertices(self.updated_boundary.vertices[index], connected_vs)
                    if is_inside_boundary(
                        self.updated_boundary.vertices[index],
                        new_v,
                        clockwise_boundary,
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

        c_neighbors = Boundary.get_closest_points(
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

        neighbors = self.updated_boundary.find_closest_segments(vertex, dist)

        if len(neighbors) or len(c_neighbors):
            failed = False
            times = 4
            while True:
                n_v = indention_vertex(vertex, left_v, right_v, (360 - v_angle) / 2, dist / times)
                connected_vs = vertex.get_connected_vertices()
                if times >= 10:
                    failed = True
                    break
                clockwise_boundary = clockwise_vertices(vertex, connected_vs)
                if is_inside_boundary(vertex, n_v, clockwise_boundary, left_v, right_v):
                    break
                else:
                    times += 1
            if not failed:
                return n_v
        return vertex

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
                        origin = Boundary.compute_dist(origin, vertex)[0][0]

                        p_dist = Boundary.compute_dist(
                            [
                                v for v in self.updated_boundary.vertices
                                if v not in connected_vertices and v is not vertex
                            ],
                            origin
                        )
                        if not len(p_dist):
                            continue
                        cloest_p, dist = p_dist[0]

                        estimate_vertex = self.estimate_4th_vertex(origin, connected_vertices[0], connected_vertices[1], suggest_dist=dist)

                        vertex.x = estimate_vertex.x
                        vertex.y = estimate_vertex.y


                    else:
                        for connect_v in connected_vertices:
                            vertex.x = lr * vertex.x + (1 - lr) * connect_v.x
                            vertex.y = lr * vertex.y + (1 - lr) * connect_v.y

                elif len(near_meshes) == 2:
                    update_boundary_vertices = []
                    inside_vertices = []
                    for v in connected_vertices:
                        if v in self.updated_boundary.vertices:
                            update_boundary_vertices.append(v)
                        else:
                            inside_vertices.append(v)

                    if len(inside_vertices) == 1 and len(update_boundary_vertices) == 2:
                        inside_vertex = inside_vertices[0]

                        origins = inside_vertex.get_common_vertex(update_boundary_vertices[0])
                        common_v1s = [v for v in origins if v is not vertex]
                        common_v1s = Boundary.compute_dist(common_v1s, vertex)[0]

                        origins = inside_vertex.get_common_vertex(update_boundary_vertices[1])
                        common_v2s = [v for v in origins if v is not vertex]
                        common_v2s = Boundary.compute_dist(common_v2s, vertex)[0]

                        common_v1 = common_v1s[0]
                        common_v2 = common_v2s[0]

                        estimate_vertex_1 = self.estimate_4th_vertex(common_v1, update_boundary_vertices[0], inside_vertex, factor=0.7)

                        estimate_vertex_2 = self.estimate_4th_vertex(common_v2, update_boundary_vertices[1], inside_vertex, factor=0.7)

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

        self.updated_boundary.find_reference_candidates(target_angle=0)
