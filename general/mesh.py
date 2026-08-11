import json
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
            print("No enough vertices to compute boundary quality!")
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


class Mesh:
    def __init__(self, boundary):
        self.boundary = boundary
        self.updated_boundary = boundary.copy()
        self.all_vertices = boundary.vertices
        self.original_vertices = list(boundary.vertices)
        self.average_edge_length = self.boundary.average_edge_length()

        self.original_boundary = boundary.deep_copy()
        self.original_area = boundary.poly_area()
        self.generated_quads = []

    def reset(self):
        self.boundary = self.original_boundary.deep_copy()
        self.updated_boundary = self.boundary.copy()
        self.original_vertices = list(self.boundary.vertices)
        self.generated_quads = []

    def can_commit_quad(self, quad, reference_point):
        return quad.is_valid(0) and \
            not self.updated_boundary.check_intersection_with_boundary(quad, reference_point)

    def commit_quad(self, quad, reference_point):
        quad.connect_vertices()
        self.generated_quads.append(quad)
        self.updated_boundary.update_boundary(reference_point, quad, self.boundary)

    def find_related_quads(self, vertex):
        return list({m for m in self.generated_quads if vertex in m.vertices})

    def get_quality(self, element, index=0):
        if index == 1:
            return quad_quality(element, 'edge_angle')
        elif index == 2:
            b_reward = self.updated_boundary.compute_element_boundary_quality(element)
            e_reward = quad_quality(element, 'robust')
            return e_reward + 1 * (b_reward - 1)
        elif index == 4:
            return quad_quality(element, 'robust')
        elif index == 5:
            return quad_quality(element, 'strong')
        raise ValueError(f"Unknown quality index: {index}")

    def write_elements_to_file(self, filename, format='inp'):
        if len(self.generated_quads) == 0:
            print("There are no elements generated!")
            return

        nodes = list(self.original_vertices)  # boundary nodes (referenced by the B21 edge elements)
        for ele in self.generated_quads:
            nodes.extend([n for n in ele.vertices if n not in nodes])

        with open(filename, 'w') as fw:
            fw.write("*NODE, NSET=ALLNODES\n")
            for idx, node in enumerate(nodes):
                fw.write(f"{idx+1}, {node.x}, {node.y}" + "\n")

            i = 0
            for i in range(1, len(self.original_vertices)):
                fw.write(f'*ELEMENT, TYPE=B21, ELSET=EB{i}\n {i+1}, {nodes.index(self.original_vertices[i-1]) + 1}, {nodes.index(self.original_vertices[i]) + 1}\n')
            fw.write(f'*ELEMENT, TYPE=S4R, ELSET=EB{i+1} \n')
            for idx, ele in enumerate(self.generated_quads):
                fw.write(f"{idx+1}, {nodes.index(ele.vertices[0]) + 1}, "
                         f"{nodes.index(ele.vertices[1]) + 1}, "
                         f"{nodes.index(ele.vertices[2]) + 1}, "
                         f"{nodes.index(ele.vertices[3]) + 1}" + "\n")
        print("Document writing is finished!")

    # --- mesh relaxation (smoothing) ---

    def smooth_pave(self, vertices, current_boundary_vertices, lr_1=None, lr_2=None, iteration=400, interior=False):
        # self.smooth_current_boundary(current_boundary_vertices, lr_1=lr_1, lr_2=lr_2, iteration=iteration)
        if not interior:
            self.updated_boundary.smooth_front(self.original_vertices)
        self.smooth_fixed_vertices([v for v in vertices if v not in current_boundary_vertices], iteration)
        self.updated_boundary.find_reference_candidates(target_angle=0)

    @staticmethod
    def estimate_4th_vertex(origin, left, right, factor=0.5, suggest_dist=None):
        distance = (origin.distance_to(left) + origin.distance_to(right)) * factor
        if suggest_dist is not None:
            distance = min(distance, 0.6 * suggest_dist)
        s = Segment.get_ray_segment(Segment(origin, left), Segment(origin, right), distance)
        return s.point2

    def smooth_fixed_vertices(self, vertices, iteration):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        while diffs > 0.001 and i_iteration < iteration:
            i_iteration += 1
            new_sum_coordinates = 0
            for vertex in vertices:
                if vertex in self.original_vertices:
                    continue
                connected = vertex.get_connected_vertices()
                if not connected:
                    continue
                count = len(connected)
                vertex.x = sum(c.x + vertex.x for c in connected) / (2 * count)
                vertex.y = sum(c.y + vertex.y for c in connected) / (2 * count)
                new_sum_coordinates += vertex.x + vertex.y
            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates
        print(f"Smoothing fixed vertices,Iteration numbers: {i_iteration}, the diff of smoothing is {diffs}!")

    def smooth(self, vertices, lr_1=0.999, lr_2=0.999, iteration=400):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        while diffs > 0.001 and i_iteration < iteration:
            i_iteration += 1
            for vertex in vertices:
                if vertex not in self.original_vertices:
                    self._smooth_vertex(vertex, lr_1, lr_2)
            new_sum_coordinates = sum(v.x + v.y for v in vertices)
            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates
        print(f"Iteration numbers: {i_iteration}, the diff of smoothing is {diffs}!")

        self.updated_boundary.find_reference_candidates(target_angle=0)

    def _smooth_vertex(self, vertex, lr_1, lr_2):
        connected = vertex.get_connected_vertices()
        n_meshes = len(self.find_related_quads(vertex))

        if n_meshes == 1 and len(connected) == 2:
            origins = [v for v in connected[0].get_common_vertex(connected[1]) if v is not vertex]
            origin = Boundary.sorted_by_distance(origins, vertex)[0][0]
            p_dist = Boundary.sorted_by_distance(
                [v for v in self.updated_boundary.vertices
                 if v not in connected and v is not vertex], origin)
            if not p_dist:
                return
            estimate = self.estimate_4th_vertex(origin, connected[0], connected[1], suggest_dist=p_dist[0][1])
            vertex.x, vertex.y = estimate.x, estimate.y
            return

        if n_meshes == 2:
            on_front = [v for v in connected if v in self.updated_boundary.vertices]
            interior = [v for v in connected if v not in self.updated_boundary.vertices]
            if len(interior) == 1 and len(on_front) == 2:
                inside = interior[0]
                origins1 = [v for v in inside.get_common_vertex(on_front[0]) if v is not vertex]
                origins2 = [v for v in inside.get_common_vertex(on_front[1]) if v is not vertex]
                e1 = self.estimate_4th_vertex(Boundary.sorted_by_distance(origins1, vertex)[0][0], on_front[0], inside, factor=0.7)
                e2 = self.estimate_4th_vertex(Boundary.sorted_by_distance(origins2, vertex)[0][0], on_front[1], inside, factor=0.7)
                vertex.x = (e1.x + e2.x) / 2
                vertex.y = (e1.y + e2.y) / 2
                return

        if n_meshes not in (1, 2):
            x = y = count = 0
            for c in connected:
                x += c.x + vertex.x
                y += c.y + vertex.y
                count += 1
            if count == 0:
                return
            vertex.x = x / (2 * count)
            vertex.y = y / (2 * count)
            return

        lr = lr_1 if n_meshes == 1 else lr_2
        for c in connected:
            vertex.x = lr * vertex.x + (1 - lr) * c.x
            vertex.y = lr * vertex.y + (1 - lr) * c.y

    # --- offline sample extraction (training data for the imitation network) ---

    @staticmethod
    def collect_neighbor_paths(root, exclusion, layer, path, paths, N):
        if root is None:
            return

        if len(path) < N:
            path.append(root)
        else:
            path[-layer-1] = root
        if layer == 0:
            paths.append(list(path))
            return
        else:
            nodes = [v for v in root.get_connected_vertices() if v not in exclusion and v not in path[:N-layer]]
            for i in range(len(nodes)):
                Mesh.collect_neighbor_paths(nodes[i], exclusion, layer-1, path, paths, N)

    def extract_samples(self, quads, n_neighbor, n_radius, radius, index=1, quality_threshold=0.7):
        all_samples, outputs, types = [], [], []
        for idx, element in enumerate(quads):
            print(f"Extracting element {idx} out of {len(quads)}")
            if self.get_quality(element, index=index) >= quality_threshold:
                for i in range(4):
                    rp = element.vertices[i]
                    l_p = element.vertices[(i + 1) % 4]
                    r_p = element.vertices[(i - 1)]
                    target = element.vertices[(i - 2)]
                    l_p_path, l_p_paths, exclusion = [], [], [rp, r_p]
                    self.collect_neighbor_paths(l_p, exclusion, n_neighbor - 1, l_p_path, l_p_paths, n_neighbor)

                    r_p_path, r_p_paths, exclusion = [], [], [rp, l_p]
                    self.collect_neighbor_paths(r_p, exclusion, n_neighbor - 1, r_p_path, r_p_paths, n_neighbor)

                    radius_neighbors = self.boundary.get_radius_neighbors(rp, l_p, r_p, [rp, l_p, r_p, target], radius=radius, N=n_radius)

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

                        def encode(p):
                            return [rp.distance_to(p) / (base_length * radius),
                                    rp.clockwise_angle(p, r_p) % round(2 * math.pi, 4)]

                        for p in rr:
                            _sample.extend(encode(p))
                        for p in mm:
                            _sample.extend(encode(p))
                        for p in reversed(ll):
                            _sample.extend(encode(p))
                        _target = encode(target)

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

    @staticmethod
    def save_samples(file_name, res, _type=1):
        if _type == 1:
            res['samples'] = [Vertex.points_as_array(s) for s in res['samples']]
            res['outputs'] = [Vertex.points_as_array(s) for s in res['outputs']]
        with open(file_name, 'w') as fw:
            json.dump(res, fw)


def edge_angle_quality(quad):
    length_of_edges = quad.length_4_segments()
    area = quad.area()
    if area <= 0:
        q1 = 0
    else:
        s = math.sqrt(area)
        product = 1
        for edge in length_of_edges:
            product *= math.pow(edge / s, 1 if s - edge > 0 else -1)
        q1 = math.pow(product, 1 / 4)

    angle_product = 1
    for a in quad.corner_angles():
        angle_product *= 1 - (math.fabs(math.degrees(a) - 90) / 90)
    if angle_product < 0:
        q2 = 0
    else:
        q2 = math.pow(angle_product, 1/4)

    return q1, q2


def quad_quality(quad, quality_type='robust'):
    if quality_type == 'stretch':
        return math.sqrt(2) * min(quad.length_4_segments()) / max(quad.vertices[0].distance_to(quad.vertices[2]), quad.vertices[1].distance_to(quad.vertices[3]))
    elif quality_type == 'robust':
        q1 = quad_quality(quad, 'stretch')
        angles = quad.corner_angles()
        q2 = min(angles) / max(angles)
        return math.sqrt(q1 * q2)
    elif quality_type == 'edge_angle':
        q1, q2 = edge_angle_quality(quad)
        return math.sqrt(q1 * q2)
    elif quality_type == 'taper':
        p0, p1, p2, p3 = quad.vertices[0], quad.vertices[-1], quad.vertices[-2], quad.vertices[-3]
        x1 = (p1 - p0) + (p2 - p3)
        x2 = (p2 - p1) + (p3 - p0)
        x12 = (p0 - p1) + (p2 - p3)
        return x12.length() / min(x1.length(), x2.length())
    elif quality_type == 's_jacobian':
        p0, p1, p2, p3 = quad.vertices[0], quad.vertices[-1], quad.vertices[-2], quad.vertices[-3]
        l0, l1, l2, l3 = p1-p0, p2-p1, p3-p2, p0-p3
        a3 = l2.cross(l3)
        a2 = l1.cross(l2)
        a1 = l0.cross(l1)
        a0 = l3.cross(l0)
        return min([a0 / (l0.length() * l3.length()),
                 a1 / (l0.length() * l1.length()),
                 a2 / (l1.length() * l2.length()),
                 a3 / (l2.length() * l3.length())])
    elif quality_type == 'strong':
        q1, _ = edge_angle_quality(quad)
        angles = [math.fabs(a) for a in quad.corner_angles()]
        q2 = min(angles) / max(angles)
        return math.sqrt(q1 * q2)
    raise ValueError(f"Unknown quality type: {quality_type}")
