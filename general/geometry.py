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

    def clockwise_angle(self, vertex1, vertex2):
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
            sa = round(math.sin(seg.point1.clockwise_angle(other.point1, seg.point2)), 4)
            sb = round(math.sin(seg.point1.clockwise_angle(other.point2, seg.point2)), 4)
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

    def deep_copy(self):
        copied = type(self)([v.copy() for v in self.vertices])
        copied.connect_vertices()
        return copied

    def connect_vertices(self):
        for i in range(len(self.vertices)):
            prev, curr = self.vertices[i - 1], self.vertices[i]
            if curr.has_segment_with_vertex(prev):
                continue
            seg = Segment(prev, curr)
            prev.assign_segment(seg)
            curr.assign_segment(seg)

    def all_segments(self):
        segts = []
        for vertex in self.vertices:
            for seg in vertex.segments or []:
                if seg not in segts and seg.point1 in self.vertices and seg.point2 in self.vertices:
                    segts.append(seg)
        return segts

    def get_neighbors(self, vertex, num_points=4):
        if num_points % 2 != 0:
            raise ValueError("The neighbor number is not even!")
        half = num_points // 2
        n = len(self.vertices)
        index = self.vertices.index(vertex)

        vertices = [self.vertices[(index + i) % n] for i in reversed(range(half + 1))]
        vertices += [self.vertices[index - i] for i in range(1, half + 1)]
        return vertices

    @staticmethod
    def sorted_by_distance(vertices, vertex):
        dists = [(v, vertex.distance_to(v)) for v in vertices if vertex is not v]
        return sorted(dists, key=lambda x: x[1])

    @staticmethod
    def get_closest_points(vertices, vertex, exclusion=None, max_dist=None):
        exclusion = exclusion or []
        selected = []
        for v, d in Polygon.sorted_by_distance(vertices, vertex):
            if d > max_dist:
                break
            if v not in exclusion:
                selected.append(v)
        return selected

    @staticmethod
    def get_points_within_angle(vertices, base_point, start_point, start_angle, end_angle):
        target_points = [v for v in vertices
            if start_angle < base_point.clockwise_angle(start_point, v) < end_angle]
        return target_points

    @staticmethod
    def coincides_with_any(vertex, points):
        return any(p.distance_to(vertex) < 0.001 for p in points)

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

    def find_same_point(self, point):
        return next((p for p in self.vertices if p.distance_to(point) < 0.001), None)

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
        return math.degrees(vertex.clockwise_angle(left_v, right_v))

    def poly_area(self):
        xy = np.array([[v.x, v.y] for v in self.vertices])
        return 0.5 * np.abs(np.dot(xy[:, 0],np.roll(xy[:, 1],1))-np.dot(xy[:, 1],np.roll(xy[:, 0],1)))


class Quad(Polygon):
    def __init__(self, vertices):
        super().__init__(vertices)
        self.segments = None

        self.max_degree = 0.99 * math.pi
        self.min_degree = 0.01 * math.pi

    def length_4_segments(self):
        return [self.vertices[i - 1].distance_to(self.vertices[i]) for i in range(len(self.vertices))]

    def corner_angles(self):
        return [self.vertices[i].clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])
                for i in range(4)]

    def is_valid(self, quality_method=0):
        if self.segments_crossed():
            return False

        if quality_method == 0:
            for degree in self.corner_angles():
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
        return [math.degrees(math.fabs(a - math.pi / 2)) for a in self.corner_angles()]

    def area(self):
        lengths = self.length_4_segments()
        corner_1 = self.vertices[0].clockwise_angle(self.vertices[1], self.vertices[-1])
        corner_3 = self.vertices[2].clockwise_angle(self.vertices[3], self.vertices[1])

        return 0.5 * lengths[0] * lengths[1] * math.sin(corner_1) + \
               0.5 * lengths[2] * lengths[3] * math.sin(corner_3)


def clip_angle(angle, max_angle):
    return min(angle, max_angle + math.pi / 2)


def transformation(points, dist, p0, p1):
    matrix = np.asarray(points, dtype=float).reshape(-1, 2) - p0
    matrix = np.divide(matrix, dist)

    d = p1 - p0
    theta = math.atan2(d[1], d[0])

    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [- np.sin(theta), np.cos(theta)]
    ])
    matrix = np.matmul(rotation_matrix, matrix.T).T

    return np.asarray(matrix).reshape(-1)


def detransformation(point, dist, p0, p1):
    d = p1 - p0
    theta = 2 * math.pi - math.atan2(d[1], d[0])
    original_point = np.empty(2)

    original_point[0] = np.cos(theta) * point[0] + np.sin(theta) * point[1]
    original_point[1] = -np.sin(theta) * point[0] + np.cos(theta) * point[1]

    original_point *= dist

    original_point[0] += p0[0]
    original_point[1] += p0[1]

    return original_point


def _circle_line_vertices(a, b, A, B, W, dist):
    if B == 0:
        x1 = x2 = W / A + a
        y1, y2 = b + math.sqrt(dist ** 2 - (W / A) ** 2), b - math.sqrt(dist ** 2 - (W / A) ** 2)
    elif A == 0:
        x1, x2 = a + math.sqrt(dist ** 2 - (W / B) ** 2), a - math.sqrt(dist ** 2 - (W / B) ** 2)
        y1 = y2 = W / B + b
    else:
        M = -A / B
        N = (W + A * a + B * b) / B
        lin = 2 * M * b - 2 * M * N + 2 * a
        disc = math.sqrt(math.fabs(lin ** 2 - 4 * (M ** 2 + 1) * ((N - b) ** 2 + a ** 2 - dist ** 2)))
        denom = 2 * (M ** 2 + 1)
        x1, x2 = (lin + disc) / denom, (lin - disc) / denom
        y1, y2 = M * x1 + N, M * x2 + N
    return Vertex(x1, y1), Vertex(x2, y2)


def middle_vertex(vertex, left_v, right_v, target_angle):
    m_v = (left_v + right_v) / 2
    A = right_v.x - left_v.x
    B = right_v.y - left_v.y
    D = left_v.distance_to(m_v) / math.tan(math.radians(target_angle / 2))

    V1, V2 = _circle_line_vertices(m_v.x, m_v.y, A, B, 0, D)
    return V1 if V1.distance_to(vertex) < V2.distance_to(vertex) else V2


def side_vertex(vertex, next_v, next_next_v, angle, dist):
    W = dist * next_v.distance_to(next_next_v) * math.cos(math.radians(angle))
    V1, V2 = _circle_line_vertices(next_v.x, next_v.y, next_next_v.x - next_v.x, next_next_v.y - next_v.y, W, dist)
    return V1 if V1.distance_to(vertex) < V2.distance_to(vertex) else V2


def indention_vertex(vertex, left_v, right_v, angle, dist):
    W = dist * vertex.distance_to(left_v) * math.cos(math.radians(angle))
    V1, V2 = _circle_line_vertices(vertex.x, vertex.y, left_v.x - vertex.x, left_v.y - vertex.y, W, dist)
    return V1 if V1.clockwise_angle(left_v, right_v) < V2.clockwise_angle(left_v, right_v) else V2


def stays_inside_ring(original, candidate, ring, left_v, right_v):
    for i in range(len(ring)):
        if left_v in [ring[i], ring[i - 1]] and right_v in [ring[i], ring[i - 1]]:
            continue
        if (candidate.clockwise_angle(ring[i], ring[i - 1]) < math.pi) != \
                (original.clockwise_angle(ring[i], ring[i - 1]) < math.pi):
            return False
    return True


def clockwise_vertices(inner_v, vertices):
    for i in range(1, len(vertices)):
        max_angle = -1
        flag = i
        for j in range(i, len(vertices)):
            current_angle = inner_v.clockwise_angle(vertices[j], vertices[i - 1])
            if current_angle > max_angle:
                max_angle = current_angle
                flag = j
        if flag != i:
            vertices[i], vertices[flag] = vertices[flag], vertices[i]

    final_vertices = []
    for i in range(len(vertices)):
        inter_v = [v for v in vertices[i].get_connected_vertices()
                   if v in vertices[i - 1].get_connected_vertices() and v is not inner_v]
        final_vertices.append(vertices[i - 1])
        if inter_v:
            final_vertices.append(inter_v[0])
    return final_vertices
