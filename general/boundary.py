import math

from general.components import Vertex, Segment, Polygon, Quad
from general.math_utils import _circle_line_x


def _circle_line_vertices(a, b, A, B, W, dist):
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
    return Vertex(x1, y1), Vertex(x2, y2)


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
    W = dist * next_v.distance_to(nn_v) * math.cos(math.radians(angle))
    V1, V2 = _circle_line_vertices(next_v.x, next_v.y, nn_v.x - next_v.x, nn_v.y - next_v.y, W, dist)
    if V1.distance_to(vertex) < V2.distance_to(vertex):
        return V1
    else:
        return V2


def indention_vertex(vertex, left_v, right_v, angle, dist):
    W = dist * vertex.distance_to(left_v) * math.cos(math.radians(angle))
    V1, V2 = _circle_line_vertices(vertex.x, vertex.y, left_v.x - vertex.x, left_v.y - vertex.y, W, dist)
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


class Boundary(Polygon):
    def __init__(self, vertices, num_ref_neighbor=4, maximum_reference_angle=math.pi * 0.972):
        super().__init__(vertices)
        self.candidate_vertices = None
        self.num_ref_neighbor = num_ref_neighbor
        self.maximum_reference_angle = maximum_reference_angle

    def deep_copy(self):
        copied = type(self)([vertex.copy() for vertex in self.vertices])
        copied.connect_vertices()
        return copied

    def all_segments(self):
        segts = []
        for vertex in self.vertices:
            if vertex.segments:
                for segt in vertex.segments:
                    if segt not in segts and (segt.point1 in self.vertices and segt.point2 in self.vertices):
                        segts.append(segt)
        return segts

    def sort_segments_by_length(self, reverse=False):
        return sorted(((seg, seg.length()) for seg in self.all_segments()),
                      key=lambda x: x[1], reverse=reverse)

    def get_neighbors(self, vertex, num_points=4):
        if num_points % 2 != 0:
            raise ValueError("The neighbor number is not even!")
        half = num_points // 2
        p_num = len(self.vertices)
        index = self.vertices.index(vertex)

        vertices = [self.vertices[(index + i) % p_num] for i in reversed(range(half + 1))]
        vertices += [self.vertices[index - i] for i in range(1, half + 1)]
        return vertices

    def find_closest_segments(self, vertex, dist):
        closest_segments = []
        for i in range(len(self.vertices)):
            if vertex in [self.vertices[i-1], self.vertices[i]]:
                continue
            s = Segment(self.vertices[i-1], self.vertices[i])
            _, _dist, inner = s.perpendicular_point(vertex)
            if inner and _dist <= dist:
                closest_segments.append(s)
        return closest_segments

    def find_same_point(self, point):
        return next((p for p in self.vertices if p.distance_to(point) < 0.001), None)

    def count_segts_in_boundary(self, vertex):
        return sum(1 for seg in vertex.segments
                   if seg.point1 in self.vertices and seg.point2 in self.vertices)

    def check_boundary_point(self, vertex, index=None):
        if index is None:
            index = self.vertices.index(vertex)

        if self.num_ref_neighbor % 2 != 0:
            raise ValueError('Wrong number of reference neighbors')
        sum_angle = 0
        if self.num_ref_neighbor // 2 == 2:
            lam = 0.618
            weights = [lam, 1-lam]
        else:
            weights = [2/self.num_ref_neighbor] * (self.num_ref_neighbor // 2)

        for i in range(self.num_ref_neighbor // 2):
            clockwise_angle = vertex.to_find_clockwise_angle(
                self.vertices[(index + 1 + i) % len(self.vertices)],
                self.vertices[index - 1 - i]
            )
            if i == 0:
                if clockwise_angle >= self.maximum_reference_angle or clockwise_angle == 0:  # 175
                    return
            sum_angle += clockwise_angle * weights[i]

        return math.degrees(sum_angle)

    def find_reference_candidates(self, target_angle):
        candidate_vertices = [(vertex, ad) for i, vertex in enumerate(self.vertices)
                              if (ad := self.check_boundary_point(vertex, i)) is not None]
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

    def add_reference_candidates(self, points):
        if isinstance(points, list):
            for v in points:
                angle_dist = self.check_boundary_point(v)
                if angle_dist is not None:
                    i = 0
                    while i < len(self.candidate_vertices):
                        if angle_dist <= self.candidate_vertices[i][1]:
                            self.candidate_vertices.insert(i, (v, angle_dist))
                            break
                        i += 1
                    else:
                        self.candidate_vertices.append((v, angle_dist))
        else:
            raise ValueError("Points should be a list!")

    def remove_reference_candidates(self, points):
        if isinstance(points, list):
            for p in points:
                i = 0
                while i < len(self.candidate_vertices):
                    if self.candidate_vertices[i][0] == p:
                        del self.candidate_vertices[i]
                    else:
                        i += 1
        else:
            raise ValueError("Points should be a list!")

    @staticmethod
    def is_vertex_inside_list(vertex, points):
        return any(p.distance_to(vertex) < 0.001 for p in points)

    def check_intersection_with_boundary(self, quad, reference_point):
        max_dist = max([reference_point.distance_to(v) for v in quad.vertices if v is not reference_point])
        neighboring_vertices = [v for v in self.vertices
                                if reference_point.distance_to(v) < max_dist and v not in quad.vertices]

        _index = quad.vertices.index(reference_point)
        checking_segs = [Segment(quad.vertices[_index - 1], quad.vertices[_index - 2]),
                         Segment(quad.vertices[_index - 2], quad.vertices[_index - 3])]
        for v in neighboring_vertices:
            index = self.vertices.index(v)
            for c_g in checking_segs:
                if self.vertices[index - 1] not in quad.vertices:
                    if c_g.is_cross(Segment(v, self.vertices[index - 1])):
                        return True

                if self.vertices[(index + 1) % len(self.vertices)] not in quad.vertices:
                    if c_g.is_cross(Segment(v, self.vertices[(index + 1) % len(self.vertices)])):
                        return True

        return False

    def remove_point(self, point):
        self.vertices.remove(point)

    def update_boundary(self, reference_point, quad, mesh_boundary):
        new_vertices = []
        for v in quad.vertices:
            if v not in self.vertices:
                new_vertices.append(v)

        if len(new_vertices) == 1:
            id = self.vertices.index(quad.vertices[quad.vertices.index(new_vertices[0]) - 2])
            self.vertices.insert(id, new_vertices[0])
            self.remove_point(quad.vertices[quad.vertices.index(new_vertices[0]) - 2])
            if not new_vertices[0] in mesh_boundary.vertices:
                mesh_boundary.vertices.append(new_vertices[0])
            ref_neighbors = []
            for i in range(self.num_ref_neighbor // 2):
                ref_neighbors.extend([
                    self.vertices[(id + i + 1) % len(self.vertices)],
                    self.vertices[id - i - 1]
                ])
            self.remove_reference_candidates(ref_neighbors + [quad.vertices[quad.vertices.index(new_vertices[0]) - 2]])
            self.add_reference_candidates(ref_neighbors)

        elif len(new_vertices) == 0:
            removable_vertices = []
            for v in quad.vertices:
                if self.count_segts_in_boundary(v) < 3:
                    removable_vertices.append(v)
            for v in removable_vertices:
                self.vertices.remove(v)

            id = max([self.vertices.index(v) for v in quad.vertices
                      if v not in removable_vertices])
            ref_neighbors = []
            for i in range(self.num_ref_neighbor // 2):
                ref_neighbors.extend([
                    self.vertices[(id + i) % len(self.vertices)],
                    self.vertices[id - i - 1]
                ])

            self.remove_reference_candidates(removable_vertices + ref_neighbors)
            self.add_reference_candidates(ref_neighbors)

    def rule_element(self, rule, index, new_point=None):
        v = self.vertices
        n = len(v)
        if rule == -1:
            return Quad([v[index - 1], v[index], v[(index + 1) % n], v[(index + 2) % n]])
        if rule == 1:
            return Quad([v[index - 2], v[index - 1], v[index], v[(index + 1) % n]])
        return Quad([new_point, v[index - 1], v[index], v[(index + 1) % n]])

    def estimate_area_range(self):
        # e_min/e_max robustified to 2nd shortest/longest edge, capped at 2*mean, not raw extremes
        lengths = [l[1] for l in self.sort_segments_by_length()]
        L = sum(lengths) / len(lengths)
        max_L = min(lengths[-2], 2 * L)
        min_L = min(L / math.sqrt(2), lengths[1])
        return min_L, (max_L + 3 * min_L) / 4

    def compute_boundary_quality(self, add_v):
        v = self.vertices
        n = len(v)
        index = v.index(add_v)
        angles = []
        # product = 1
        for i in [1, -1]:
            angle = v[(index + i) % n].to_find_clockwise_angle(
                v[(index + i + 1) % n],
                v[index + i - 1])
            if angle < math.pi / 3:
                angles.append(angle)
                # product *= 3 * angle / math.pi
        # return math.pow(product, 1 / 2)
        q1 = 3 * min(angles) / math.pi if len(angles) else 1
        # q1 = product

        close_vs = []
        dist = add_v.distance_to(v[(index + 1) % n]) + add_v.distance_to(v[index - 1])
        for i, vv in enumerate(v):
            if vv in [v[index],
                      v[(index + 1) % n],
                      v[(index + 2) % n],
                      v[index - 1],
                      v[index - 2]]:
                continue
            if add_v.distance_to(vv) < dist:
                if i - 1 in close_vs:
                    continue
                close_vs.append(i)
        dists = []
        for i in close_vs:
            seg = Segment(v[(i + 1) % n], v[i])
            dists.append(seg.distance(add_v))

        target_len = dist / 2

        _dists = [(index + i) % n for i in range(-2, 3)]
        mean_dist = sum([v[_dists[i]].distance_to(
            v[_dists[i + 1]]) for i in range(len(_dists) - 1)]) / (len(_dists) - 1)

        smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)

        if len(dists):
            m_d = min(dists)
            q2 = m_d / (0.5 * dist) if m_d < 0.5 * dist else 1
        else:
            q2 = 1

        pow = 1/3
        return math.pow(smoothness * q1 * q2, pow)

    def compute_ele_boundary_quality(self, element):
        v = self.vertices
        n = len(v)
        new_vs = [x for x in element.vertices
                  if len(x.get_connected_vertices()) == 2 and x in v]

        if len(new_vs):
            return self.compute_boundary_quality(new_vs[0]) # * self.compute_boundary_narrowness(new_vs[0])
        else:
            target_vs = [x for x in element.vertices if x in v]

            if not len(target_vs):
                print("No enough vertices to compute boundary quality!")
                return 1

            angles, dists = [], []
            # product = 1
            for i, x in enumerate(target_vs):
                index = v.index(x)
                angle = x.to_find_clockwise_angle(
                    v[(index + 1) % n],
                    v[index - 1])
                if angle < math.pi / 3:
                    angles.append(angle)
                    # product *= 2 * angle / math.pi
            # return math.pow(product, 1 / len(target_vs))
            index_1, index_r = v.index(target_vs[0]), v.index(target_vs[1])
            index = index_1 if index_1 < index_r else index_r

            target_len = target_vs[0].distance_to(target_vs[1])

            dists = [(index + i) % n for i in range(-2, 4)]
            mean_dist = sum([
                v[dists[i]].distance_to(v[dists[i+1]])
                for i in range(len(dists) - 1)
            ]) / (len(dists) - 1)

            smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)
            angle_quality = 3 * min(angles) / math.pi if len(angles) else 1
            pow = 1/2
            return math.pow(angle_quality * smoothness, pow)
