import math
import itertools
from types import SimpleNamespace

import numpy as np

from general.geometry import (Vertex, Segment, Polygon, Quad, clip_angle,
                              middle_vertex, side_vertex, indention_vertex,
                              stays_inside_ring, clockwise_vertices)


class Boundary(Polygon):
    def __init__(self, vertices, num_ref_neighbor=4, maximum_reference_angle=math.pi * 0.972):
        super().__init__(vertices)
        self.num_ref_neighbor = num_ref_neighbor
        self.maximum_reference_angle = maximum_reference_angle
        self.candidate_vertices = None

    def sort_segments_by_length(self, reverse=False):
        return sorted(((seg, seg.length()) for seg in self.all_segments()),
                      key=lambda x: x[1], reverse=reverse)

    def find_closest_segments(self, vertex, dist):
        closest = []
        for i in range(len(self.vertices)):
            prev, curr = self.vertices[i - 1], self.vertices[i]
            if vertex in (prev, curr):
                continue
            seg = Segment(prev, curr)
            _, perp_dist, inner = seg.perpendicular_point(vertex)
            if inner and perp_dist <= dist:
                closest.append(seg)
        return closest

    def count_boundary_segments(self, vertex):
        return sum(1 for seg in vertex.segments
                   if seg.point1 in self.vertices and seg.point2 in self.vertices)

    def reference_angle_score(self, vertex, index=None):
        if index is None:
            index = self.vertices.index(vertex)
        n = len(self.vertices)
        half = self.num_ref_neighbor // 2
        if half == 2:
            lam = 0.618
            weights = [lam, 1 - lam]
        else:
            weights = [2 / self.num_ref_neighbor] * half

        sum_angle = 0
        for i in range(half):
            clockwise_angle = vertex.clockwise_angle(
                self.vertices[(index + 1 + i) % n], self.vertices[index - 1 - i])
            if i == 0 and (clockwise_angle >= self.maximum_reference_angle or clockwise_angle == 0):
                return None
            sum_angle += clockwise_angle * weights[i]
        return math.degrees(sum_angle)

    def find_reference_candidates(self, target_angle):
        candidates = [(vertex, ad) for i, vertex in enumerate(self.vertices)
                      if (ad := self.reference_angle_score(vertex, i)) is not None]
        self.candidate_vertices = sorted(candidates, key=lambda x: math.fabs(x[1] - target_angle))

    def find_reference_point(self, not_valid_points=None, target_angle=0):
        if self.candidate_vertices is None:
            self.find_reference_candidates(target_angle)
        if not self.candidate_vertices:
            return None
        if not not_valid_points:
            return self.candidate_vertices[0][0]
        for vertex, _ in self.candidate_vertices:
            if not self.coincides_with_any(vertex, not_valid_points):
                return vertex
        return None

    def add_reference_candidates(self, points):
        for vertex in points:
            angle_dist = self.reference_angle_score(vertex)
            if angle_dist is None:
                continue
            i = 0
            while i < len(self.candidate_vertices):
                if angle_dist <= self.candidate_vertices[i][1]:
                    self.candidate_vertices.insert(i, (vertex, angle_dist))
                    break
                i += 1
            else:
                self.candidate_vertices.append((vertex, angle_dist))

    def remove_reference_candidates(self, points):
        for p in points:
            i = 0
            while i < len(self.candidate_vertices):
                if self.candidate_vertices[i][0] == p:
                    del self.candidate_vertices[i]
                else:
                    i += 1

    def rule_element(self, rule, index, new_point=None):
        v = self.vertices
        n = len(v)
        if rule == -1:
            return Quad([v[index - 1], v[index], v[(index + 1) % n], v[(index + 2) % n]])
        if rule == 1:
            return Quad([v[index - 2], v[index - 1], v[index], v[(index + 1) % n]])
        return Quad([new_point, v[index - 1], v[index], v[(index + 1) % n]])

    def check_intersection_with_boundary(self, quad, reference_point):
        max_dist = max(reference_point.distance_to(v) for v in quad.vertices if v is not reference_point)
        neighbouring = [v for v in self.vertices
                        if reference_point.distance_to(v) < max_dist and v not in quad.vertices]

        idx = quad.vertices.index(reference_point)
        checking_segs = [Segment(quad.vertices[idx - 1], quad.vertices[idx - 2]),
                         Segment(quad.vertices[idx - 2], quad.vertices[idx - 3])]
        n = len(self.vertices)
        for v in neighbouring:
            index = self.vertices.index(v)
            prev, nxt = self.vertices[index - 1], self.vertices[(index + 1) % n]
            for seg in checking_segs:
                if prev not in quad.vertices and seg.is_cross(Segment(v, prev)):
                    return True
                if nxt not in quad.vertices and seg.is_cross(Segment(v, nxt)):
                    return True
        return False

    def update_boundary(self, reference_point, quad, mesh_boundary):
        new_vertices = [v for v in quad.vertices if v not in self.vertices]
        half = self.num_ref_neighbor // 2

        if len(new_vertices) == 1:
            new_v = new_vertices[0]
            replaced = quad.vertices[quad.vertices.index(new_v) - 2]
            idx = self.vertices.index(replaced)
            self.vertices.insert(idx, new_v)
            self.vertices.remove(replaced)
            if new_v not in mesh_boundary.vertices:
                mesh_boundary.vertices.append(new_v)

            ref_neighbors = []
            for i in range(half):
                ref_neighbors += [self.vertices[(idx + i + 1) % len(self.vertices)],
                                  self.vertices[idx - i - 1]]
            self.remove_reference_candidates(ref_neighbors + [replaced])
            self.add_reference_candidates(ref_neighbors)

        elif len(new_vertices) == 0:
            removable = [v for v in quad.vertices if self.count_boundary_segments(v) < 3]
            for v in removable:
                self.vertices.remove(v)

            idx = max(self.vertices.index(v) for v in quad.vertices if v not in removable)
            ref_neighbors = []
            for i in range(half):
                ref_neighbors += [self.vertices[(idx + i) % len(self.vertices)],
                                  self.vertices[idx - i - 1]]
            self.remove_reference_candidates(removable + ref_neighbors)
            self.add_reference_candidates(ref_neighbors)

    def estimate_area_range(self):
        lengths = [length for _, length in self.sort_segments_by_length()]
        L = sum(lengths) / len(lengths)
        max_L = min(lengths[-2], 2 * L)
        min_L = min(L / math.sqrt(2), lengths[1])
        return min_L, (max_L + 3 * min_L) / 4

    def compute_boundary_quality(self, added_vertex):
        v = self.vertices
        n = len(v)
        index = v.index(added_vertex)

        angles = [a for i in (1, -1)
                  if (a := v[(index + i) % n].clockwise_angle(v[(index + i + 1) % n], v[index + i - 1])) < math.pi / 3]
        q1 = 3 * min(angles) / math.pi if angles else 1

        dist = added_vertex.distance_to(v[(index + 1) % n]) + added_vertex.distance_to(v[index - 1])
        excluded = {v[index], v[(index + 1) % n], v[(index + 2) % n], v[index - 1], v[index - 2]}
        close_vs = []
        for i, vv in enumerate(v):
            if vv not in excluded and added_vertex.distance_to(vv) < dist and i - 1 not in close_vs:
                close_vs.append(i)
        dists = [Segment(v[(i + 1) % n], v[i]).distance(added_vertex) for i in close_vs]

        target_len = dist / 2
        ring = [(index + i) % n for i in range(-2, 3)]
        mean_dist = sum(v[ring[i]].distance_to(v[ring[i + 1]]) for i in range(len(ring) - 1)) / (len(ring) - 1)
        smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)

        if dists and min(dists) < 0.5 * dist:
            q2 = min(dists) / (0.5 * dist)
        else:
            q2 = 1

        return math.pow(smoothness * q1 * q2, 1 / 3)

    def compute_element_boundary_quality(self, element):
        v = self.vertices
        n = len(v)
        new_vs = [x for x in element.vertices
                  if len(x.get_connected_vertices()) == 2 and x in v]

        if new_vs:
            return self.compute_boundary_quality(new_vs[0])

        target_vs = [x for x in element.vertices if x in v]
        if not target_vs:
            return 1

        angles = []
        for x in target_vs:
            index = v.index(x)
            angle = x.clockwise_angle(v[(index + 1) % n], v[index - 1])
            if angle < math.pi / 3:
                angles.append(angle)
        index = min(v.index(target_vs[0]), v.index(target_vs[1]))

        target_len = target_vs[0].distance_to(target_vs[1])
        ring = [(index + i) % n for i in range(-2, 4)]
        mean_dist = sum(v[ring[i]].distance_to(v[ring[i + 1]]) for i in range(len(ring) - 1)) / (len(ring) - 1)

        smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)
        angle_quality = 3 * min(angles) / math.pi if angles else 1
        return math.pow(angle_quality * smoothness, 1 / 2)

    def reference_state(self, reference_point, neighbor_num, radius_num, area_ratio, radius, static):
        neighbors = self.get_neighbors(reference_point, num_points=neighbor_num)
        base_length = round(sum(neighbors[i].distance_to(neighbors[i - 1])
                                for i in range(1, len(neighbors))) / neighbor_num, 4)
        r_points = self._radius_points(reference_point, neighbor_num, radius_num,
                                       area_ratio, radius, static, base_length)
        return SimpleNamespace(reference_point=reference_point, neighbors=neighbors,
                               base_length=base_length, state=r_points.flatten())

    def _radius_points(self, rp, neighbor_num, radius_num, area_ratio, radius, static, base_length):
        bv = self.vertices
        n = len(bv)
        r_points = np.full([radius_num + neighbor_num, 2], 1, dtype=np.float32)
        index = bv.index(rp)
        right_p = bv[index - 1]
        left_p = bv[(index + 1) % n]
        target_length = base_length * radius
        theta = rp.clockwise_angle(left_p, right_p)

        def nd(p):
            return (rp.distance_to(p) / radius) / base_length

        for i in range(neighbor_num // 2):
            if i == 0:
                r_points[i] = [nd(right_p), area_ratio if not static else 0]
                r_points[radius_num + neighbor_num - i - 1] = [nd(left_p), theta]
            else:
                _angle = rp.clockwise_angle(bv[index - i - 1], right_p)
                r_points[i] = [nd(bv[index - i - 1]),
                               _angle if _angle < math.pi else max(_angle, 1.5 * math.pi) - 2 * math.pi]
                _angle = rp.clockwise_angle(bv[(index + 1 + i) % n], right_p)
                r_points[radius_num + neighbor_num - i - 1] = [
                    nd(bv[(index + i + 1) % n]), min(_angle, theta + math.pi / 2)]

        rotation_angle = rp.clockwise_angle(right_p, rp + Vertex(1, 0))
        for i in range(radius_num):
            a = (2 * i + 1) * theta / (2 * radius_num)
            r_points[neighbor_num // 2 + i][1] = clip_angle(a, theta)
        p_s = rp + Vertex.rotate_counterclockwise(
            Vertex(target_length * math.cos(theta / 2), target_length * math.sin(theta / 2)), rotation_angle)
        shortest_edge = [1, 0]

        for i in range(index - 1, index - n, -1):
            d = rp.distance_to(bv[i])
            if bv[i] in [right_p, left_p]:
                continue
            angle = rp.clockwise_angle(bv[i], right_p)
            if angle == 0:
                continue
            k = int(angle / (theta / radius_num))
            if k < radius_num and d < target_length:
                if r_points[k + neighbor_num // 2][0] > (d / radius) / base_length:
                    r_points[k + neighbor_num // 2][0] = (d / radius) / base_length
                    r_points[k + neighbor_num // 2][1] = clip_angle(angle, theta)

            seg = Segment(bv[i], bv[i + 1])
            ll = Segment(rp, p_s)
            flag, vv = ll.intersection_vertex(seg)
            if flag:
                _d = rp.distance_to(vv)
                if shortest_edge[0] > (_d / radius) / base_length:
                    shortest_edge[0] = (_d / radius) / base_length
                    shortest_edge[1] = i

        if shortest_edge[0] != 1 and shortest_edge[0] < r_points[(radius_num + neighbor_num) // 2][0]:
            _i = shortest_edge[1]
            for i in range(radius_num):
                r_points[neighbor_num // 2 + i] = [
                    nd(bv[i - radius_num // 2 + _i]),
                    rp.clockwise_angle(bv[i - radius_num // 2 + _i], right_p)]

        return np.asarray([[round(v[0], 4), round(v[1], 4)] for v in r_points])

    def get_radius_neighbors(self, base_point, start_point, end_point, exclusion, radius, N=3):
        def radius_neighbors_with_angle(start_angle, end_angle):
            base_length = radius * (0.5 * base_point.distance_to(start_point) + 0.5 * base_point.distance_to(end_point))
            closest_neighbors = self.get_closest_points(
                self.get_points_within_angle(
                    self.vertices, base_point, start_point, start_angle, end_angle
                ),
                base_point,
                exclusion=exclusion,
                max_dist=base_length
            )

            _angle = base_point.clockwise_angle(start_point, Vertex(base_point.x + 1, base_point.y))

            closest_neighbors.append(base_point +
                                    Vertex(base_length * math.cos(_angle - (start_angle + end_angle) / 2),
                                           base_length * math.sin((_angle - (start_angle + end_angle) / 2))))
            return closest_neighbors

        angle = base_point.clockwise_angle(start_point, end_point)
        angles = [i * angle / N for i in range(N+1)]
        neighbors = [radius_neighbors_with_angle(angles[i-1], angles[i]) for i in range(1, N+1)]
        # left_neighbors = radius_neighbors_with_angle(0.01, angle / 3)
        # middle_neighbors = radius_neighbors_with_angle(angle / 3, 2 * angle / 3)
        # right_neighbors = radius_neighbors_with_angle(2 * angle / 3, angle * 0.99)
        all_combinations = list(itertools.product(*reversed(neighbors)))
        return all_combinations

    def smooth_front(self, original_vertices):
        n = len(self.vertices)
        for i in range(n):
            cur = self.vertices[i]
            if cur in original_vertices:
                continue
            left = self.vertices[(i + 1) % n]
            right = self.vertices[i - 1]
            v_angle = math.degrees(cur.clockwise_angle(left, right))

            if v_angle <= 90:
                new_v = self.find_middle_vertex(cur, left, right, v_angle)
            elif v_angle <= 180:
                left_angle = self.compute_boundary_angle(left)
                right_angle = self.compute_boundary_angle(right)
                if right_angle < 45:
                    new_v = self.find_side_vertex(cur, left, right, self.vertices[i - 2], right_angle)
                elif left_angle < 45:
                    new_v = self.find_side_vertex(cur, right, left, self.vertices[(i + 2) % n], left_angle)
                else:
                    new_v = self.find_indention_vertex(cur, v_angle)
            elif v_angle <= 270:
                new_v = self.find_indention_vertex(cur, v_angle)
            else:
                inner = self.inner_vertex(cur, 45)
                cur.x, cur.y = inner.x, inner.y
                new_v = self.find_indention_vertex(cur, v_angle)

            cur.x, cur.y = new_v.x, new_v.y

    def find_middle_vertex(self, vertex, left_v, right_v, v_angle):
        target_angle = v_angle if v_angle >= 45 else 45
        while True:
            n_v = middle_vertex(vertex, left_v, right_v, target_angle)
            if target_angle >= 135:
                return vertex
            clockwise_boundary = clockwise_vertices(vertex, vertex.get_connected_vertices())
            if stays_inside_ring(vertex, n_v, clockwise_boundary, left_v, right_v):
                return n_v
            target_angle += 5

    def find_side_vertex(self, vertex, _next_v, next_v, next_next_v, v_angle):
        dist = (vertex.distance_to(_next_v) + vertex.distance_to(next_v) + next_v.distance_to(next_next_v)) / 3
        target_angle = 45
        while True:
            n_v = side_vertex(vertex, next_v, next_next_v, target_angle, dist)
            if target_angle <= v_angle:
                return vertex
            clockwise_boundary = clockwise_vertices(vertex, vertex.get_connected_vertices())
            if stays_inside_ring(vertex, n_v, clockwise_boundary, _next_v, next_v):
                return n_v
            target_angle -= 5

    def find_indention_vertex(self, vertex, v_angle):
        index = self.vertices.index(vertex)
        n = len(self.vertices)
        left_v = self.vertices[(index + 1) % n]
        right_v = self.vertices[index - 1]
        dist = (vertex.distance_to(left_v) + vertex.distance_to(right_v)) / 2

        c_neighbors = self.get_closest_points(
            self.vertices, vertex,
            [self.vertices[index - 2], right_v, left_v, self.vertices[(index + 2) % n]], dist)
        neighbors = self.find_closest_segments(vertex, dist)
        if not (neighbors or c_neighbors):
            return vertex

        times = 4
        while True:
            n_v = indention_vertex(vertex, left_v, right_v, (360 - v_angle) / 2, dist / times)
            if times >= 10:
                return vertex
            clockwise_boundary = clockwise_vertices(vertex, vertex.get_connected_vertices())
            if stays_inside_ring(vertex, n_v, clockwise_boundary, left_v, right_v):
                return n_v
            times += 1

    def inner_vertex(self, vertex, angle):
        index = self.vertices.index(vertex)
        left_v = self.vertices[(index + 1) % len(self.vertices)]
        right_v = self.vertices[index - 1]

        m_v = (left_v + right_v) / 2
        d = m_v.distance_to(right_v) * math.tan(math.radians(angle))
        diff = vertex - m_v
        s = math.sqrt(d ** 2 / (diff.x ** 2 + diff.y ** 2))
        return m_v + diff * s
