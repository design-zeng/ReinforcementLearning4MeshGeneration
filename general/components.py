import math

import numpy as np


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.segments = None

    def distance_to(self, vertex):
        return (self - vertex).length()

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def cross(self, other):
        return self.x * other.y - other.x * self.y

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    @staticmethod
    def points_as_array(vertices):
        return [coord for vertex in vertices for coord in (vertex.x, vertex.y)]

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        return Vertex(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Vertex(self.x / other, self.y / other)

    def __str__(self):
        return f"({self.x, self.y})"

    def assign_segment(self, segment):
        if self.segments:
            self.segments.append(segment)
        else:
            self.segments = [segment]

    def to_find_clockwise_angle(self, vertex1, vertex2):
        v1 = vertex1 - self
        v2 = vertex2 - self

        theta = - math.atan2(v1.cross(v2), v1.dot(v2))

        return round(theta, 4) if math.copysign(1, theta) >= 0 else round(2 * math.pi + theta, 4)

    def has_segment_with_vertex(self, vertex):
        return bool(self.segments) and any(seg.has_vertex(vertex) for seg in self.segments)

    def get_connected_vertices(self):
        if not self.segments:
            return []

        vertices = []
        for seg in self.segments:
            other = seg.point2 if seg.point1 is self else seg.point1
            if other not in vertices:
                vertices.append(other)
        return vertices

    def copy(self):
        return Vertex(self.x, self.y)

    @staticmethod
    def rotate_counterclockwise(vertex, angle, origin=None):
        o = np.array([0.0, 0.0]) if origin is None else np.array([origin.x, origin.y])
        c, s = math.cos(angle), math.sin(angle)
        q = o + np.array([[c, -s], [s, c]]) @ (np.array([vertex.x, vertex.y]) - o)
        return Vertex(float(q[0]), float(q[1]))

    def get_common_vertex(self, another_vertex):
        connected_v = self.get_connected_vertices()
        another_vertex_v = another_vertex.get_connected_vertices()
        return [v for v in connected_v if v in another_vertex_v]


class Segment:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def is_cross(self, another_segment):
        def straddle(seg, other):
            sa = round(math.sin(seg.point1.to_find_clockwise_angle(other.point1, seg.point2)), 4)
            sb = round(math.sin(seg.point1.to_find_clockwise_angle(other.point2, seg.point2)), 4)
            if sa == 0 and sb == 0:
                long_seg, short_seg = (seg, other) if seg.length() > other.length() else (other, seg)
                m = (long_seg.point1 + long_seg.point2) / 2
                return min(m.distance_to(short_seg.point1), m.distance_to(short_seg.point2)) <= long_seg.length() / 2

            vm = seg.point2 - seg.point1
            return (other.point1 - seg.point1).cross(vm) * (other.point2 - seg.point1).cross(vm) <= 0

        return straddle(self, another_segment) and straddle(another_segment, self)

    def seg_angle(self):
        d = self.point2 - self.point1
        return math.atan2(d.y, d.x)

    @staticmethod
    def get_ray_segment(segment, another_segment, remote_dist):
        ray_point1 = (segment.point1 + another_segment.point1) / 2
        ray_point2 = (segment.point2 + another_segment.point2) / 2

        average_segment = Segment(ray_point1, ray_point2)
        return Segment.build_ray(average_segment, remote_dist)

    @staticmethod
    def build_ray(segment, remote_dist):
        theta = segment.seg_angle()
        p1 = segment.point1.copy()
        p2 = Vertex(p1.x + remote_dist * math.cos(theta), p1.y + remote_dist * math.sin(theta))
        return Segment(p1, p2)

    def has_vertex(self, vertex):
        return vertex == self.point1 or vertex == self.point2

    def perpendicular_point(self, vertex):
        d = self.point2 - self.point1
        w = vertex - self.point1
        s = d.dot(w) / d.length() ** 2
        target = self.point1 + d * s
        return target, vertex.distance_to(target), 0 <= s <= 1

    def length(self):
        return self.point1.distance_to(self.point2)

    def intersection_vertex(self, another_seg):
        u = self.point2 - self.point1
        w = another_seg.point2 - another_seg.point1
        d = u.cross(w)
        if d == 0:
            return None, None
        qp = another_seg.point1 - self.point1
        s = qp.cross(w) / d
        h = qp.cross(u) / d
        is_inside = 0 < s < 1 and 0 < h < 1
        return is_inside, self.point1 + u * s

    def distance(self, another):
        if isinstance(another, Vertex):
            d = self.point2 - self.point1
            w = another - self.point1
            s = max(0.0, min(1.0, d.dot(w) / d.length() ** 2))
            return another.distance_to(self.point1 + d * s)
        elif isinstance(another, Segment):
            return min(self.distance(another.point1), self.distance(another.point2),
                       another.distance(self.point1), another.distance(self.point2))
        else:
            raise ValueError('Not recognized object type!')


class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def copy(self):
        return type(self)(list(self.vertices))

    def connect_vertices(self):
        for i in range(len(self.vertices)):
            prev, curr = self.vertices[i - 1], self.vertices[i]
            if curr.has_segment_with_vertex(prev):
                continue
            seg = Segment(prev, curr)
            prev.assign_segment(seg)
            curr.assign_segment(seg)

    @staticmethod
    def compute_dist(vertices, vertex):
        dists = [(v, vertex.distance_to(v)) for v in vertices if vertex is not v]
        return sorted(dists, key=lambda x: x[1])

    @staticmethod
    def get_closest_points(vertices, vertex, exclusion=None, S_T=None):
        exclusion = exclusion or []
        selected = []
        for v, d in Polygon.compute_dist(vertices, vertex):
            if d > S_T:
                break
            if v not in exclusion:
                selected.append(v)
        return selected

    @staticmethod
    def get_points_within_angle(vertices, base_point, start_point, start_angle, end_angle):
        target_points = [v for v in vertices
            if start_angle < base_point.to_find_clockwise_angle(start_point, v) < end_angle]
        return target_points

    def contains_point(self, vertex):
        ray_segment = Segment(vertex, Vertex(10000, vertex.y))
        count = 0
        n = len(self.vertices)
        ray_y = ray_segment.point2.y
        for i in range(n):
            v_i, v_p = self.vertices[i], self.vertices[i - 1]
            orientation = round(v_i.y - v_p.y, 4)
            if orientation == 0:
                continue
            if not Segment(v_i, v_p).is_cross(ray_segment):
                continue
            if round(v_i.y - ray_y, 4) == 0:
                no = round(self.vertices[(i + 1) % n].y - v_i.y, 4)
                count += (no * orientation > 0 and orientation < 0)
            elif round(v_p.y - ray_y, 4) == 0:
                po = round(v_p.y - self.vertices[i - 2].y, 4)
                count += (po * orientation > 0 and orientation > 0)
            else:
                count += 1
        return count % 2 != 0

    def average_edge_length(self):
        n = len(self.vertices)
        if n == 0:
            return 0
        dist = sum(self.vertices[i].distance_to(self.vertices[i - 1]) for i in range(n))
        return round(dist / n, 4)

    def get_perimeter(self):
        return sum(self.vertices[i - 1].distance_to(self.vertices[i])
                   for i in range(1, len(self.vertices)))

    def compute_boundary_angle(self, vertex):
        if vertex not in self.vertices:
            return None
        index = self.vertices.index(vertex)
        right_v = self.vertices[index - 1]
        left_v = self.vertices[(index + 1) % len(self.vertices)]
        return math.degrees(vertex.to_find_clockwise_angle(left_v, right_v))

    def poly_area(self):
        xy = np.array([[v.x, v.y] for v in self.vertices])
        return 0.5 * np.abs(np.dot(xy[:, 0],np.roll(xy[:, 1],1))-np.dot(xy[:, 1],np.roll(xy[:, 0],1)))


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


class Quad(Polygon):
    def __init__(self, vertices):
        super().__init__(vertices)
        self.segments = None
        self.max_degree = 0.99 * math.pi
        self.min_degree = 0.01 * math.pi

    def length_4_segments(self):
        return [self.vertices[i - 1].distance_to(self.vertices[i]) for i in range(len(self.vertices))]

    def is_valid(self, quality_method=0):
        if self.segments_crossed():
            return False

        if quality_method == 0:
            for i in range(len(self.vertices)):
                degree = self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])
                if degree > self.max_degree or degree < self.min_degree:
                    return False
        return True

    def segments_crossed(self):
        v = self.vertices
        return (Segment(v[0], v[1]).is_cross(Segment(v[2], v[3])) or
                Segment(v[0], v[3]).is_cross(Segment(v[1], v[2])))

    def get_centroid(self, diff=False):
        centroid = sum(self.vertices, Vertex(0, 0)) / len(self.vertices)
        if diff:
            _diff = sum(self.length_4_segments()) / 4
            centroid.x -= _diff / 3
            centroid.y -= _diff * 0.1
        return centroid

    def inner_angles(self):
        return [math.degrees(math.fabs(self.vertices[i].to_find_clockwise_angle(
            self.vertices[(i + 1) % 4], self.vertices[i - 1]) - math.pi / 2)) for i in range(4)]

    def get_quality(self, quality_type='robust'):
        if quality_type == 'stretch':
            return math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]), self.vertices[1].distance_to(self.vertices[3]))
        elif quality_type == 'robust':
            q1 = math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]), self.vertices[1].distance_to(self.vertices[3]))
            angles = []
            for i in range(4):
                angles.append(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1]))
            q2 = min(angles) / max(angles)
            return math.sqrt(q1 * q2)
        elif quality_type == 'edge_angle':
            q1, q2 = self.edge_angle_quality()
            return math.sqrt(q1 * q2)
        elif quality_type == 'taper':
            p0, p1, p2, p3 = self.vertices[0], self.vertices[-1], self.vertices[-2], self.vertices[-3]
            x1 = (p1 - p0) + (p2 - p3)
            x2 = (p2 - p1) + (p3 - p0)
            x12 = (p0 - p1) + (p2 - p3)
            return x12.length() / min(x1.length(), x2.length())
        elif quality_type == 's_jacobian':
            p0, p1, p2, p3 = self.vertices[0], self.vertices[-1], self.vertices[-2], self.vertices[-3]
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
            q1, _ = self.edge_angle_quality()
            angles = []
            for i in range(4):
                angles.append(math.fabs(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])))
            q2 = min(angles) / max(angles)
            return math.sqrt(q1 * q2)
        raise ValueError(f"Unknown quality type: {quality_type}")

    def area(self):
        lengths = self.length_4_segments()
        corner_1 = self.vertices[0].to_find_clockwise_angle(self.vertices[1], self.vertices[-1])
        corner_3 = self.vertices[2].to_find_clockwise_angle(self.vertices[3], self.vertices[1])

        return 0.5 * lengths[0] * lengths[1] * math.sin(corner_1) + \
               0.5 * lengths[2] * lengths[3] * math.sin(corner_3)

    def edge_angle_quality(self):
        length_of_edges = self.length_4_segments()
        area = self.area()
        if area <= 0:
            q1 = 0
        else:
            s = math.sqrt(area)
            product = 1
            for edge in length_of_edges:
                product *= math.pow(edge / s, 1 if s - edge > 0 else -1)
            q1 = math.pow(product, 1 / 4)

        angle_product = 1
        for i in range(4):
            angle_product *= 1 - (math.fabs(math.degrees(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])) - 90) / 90)
        if angle_product < 0:
            q2 = 0
        else:
            q2 = math.pow(angle_product, 1/4)

        return q1, q2
