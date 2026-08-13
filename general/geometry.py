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

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        return Vertex(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Vertex(self.x / other, self.y / other)

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

    def is_neighbor(self, vertex):
        return bool(self.segments) and any(seg.has_vertex(vertex) for seg in self.segments)

    def get_neighbors(self):
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

    def get_common_neighbors(self, another_vertex):
        connected_v = self.get_neighbors()
        another_vertex_v = another_vertex.get_neighbors()
        return [v for v in connected_v if v in another_vertex_v]

    def sorted_by_distance(self, vertices):
        dists = [(v, self.distance_to(v)) for v in vertices if self is not v]
        return sorted(dists, key=lambda x: x[1])

    def get_closest_points(self, vertices, exclusion=None, max_dist=None):
        exclusion = exclusion or []
        selected = []
        for v, d in self.sorted_by_distance(vertices):
            if d > max_dist:
                break
            if v not in exclusion:
                selected.append(v)
        return selected

    def coincides_with_any(self, points):
        return any(p.distance_to(self) < 0.001 for p in points)

    @staticmethod
    def flatten(vertices):
        return [coord for vertex in vertices for coord in (vertex.x, vertex.y)]

    @staticmethod
    def rotate_counterclockwise(vertex, angle, origin=None):
        o = np.array([0.0, 0.0]) if origin is None else np.array([origin.x, origin.y])
        c, s = math.cos(angle), math.sin(angle)
        q = o + np.array([[c, -s], [s, c]]) @ (np.array([vertex.x, vertex.y]) - o)
        return Vertex(float(q[0]), float(q[1]))


class Segment:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def is_intersecting(self, another_segment):
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

    def angle(self):
        d = self.point2 - self.point1
        return math.atan2(d.y, d.x)

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

    @staticmethod
    def get_ray_segment(segment, another_segment, remote_dist):
        ray_point1 = (segment.point1 + another_segment.point1) / 2
        ray_point2 = (segment.point2 + another_segment.point2) / 2

        average_segment = Segment(ray_point1, ray_point2)
        return Segment.build_ray(average_segment, remote_dist)

    @staticmethod
    def build_ray(segment, remote_dist):
        theta = segment.angle()
        p1 = segment.point1.copy()
        p2 = Vertex(p1.x + remote_dist * math.cos(theta), p1.y + remote_dist * math.sin(theta))
        return Segment(p1, p2)


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
            if curr.is_neighbor(prev):
                continue
            seg = Segment(prev, curr)
            prev.assign_segment(seg)
            curr.assign_segment(seg)

    def own_segments(self):
        segts = []
        for vertex in self.vertices:
            for seg in vertex.segments or []:
                if seg not in segts and seg.point1 in self.vertices and seg.point2 in self.vertices:
                    segts.append(seg)
        return segts

    def n_sides(self, vertex):
        return sum(1 for seg in vertex.segments
                   if seg.point1 in self.vertices and seg.point2 in self.vertices)

    def get_neighbors(self, vertex, num_points=4):
        if num_points % 2 != 0:
            raise ValueError("The neighbor number is not even!")
        half = num_points // 2
        n = len(self.vertices)
        index = self.vertices.index(vertex)

        vertices = [self.vertices[(index + i) % n] for i in reversed(range(half + 1))]
        vertices += [self.vertices[index - i] for i in range(1, half + 1)]
        return vertices

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
            if not Segment(v_i, v_p).is_intersecting(ray_segment):
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

    def area(self):
        xy = np.array([[v.x, v.y] for v in self.vertices])
        return 0.5 * np.abs(np.dot(xy[:, 0],np.roll(xy[:, 1],1))-np.dot(xy[:, 1],np.roll(xy[:, 0],1)))

    def edge_lengths(self):
        return [self.vertices[i - 1].distance_to(self.vertices[i]) for i in range(len(self.vertices))]

    def corner_angles(self):
        n = len(self.vertices)
        return [self.vertices[i].clockwise_angle(self.vertices[(i + 1) % n], self.vertices[i - 1])
                for i in range(n)]

    def inner_angles(self):
        return [math.degrees(math.fabs(a - math.pi / 2)) for a in self.corner_angles()]

    def segments_crossed(self):
        n = len(self.vertices)
        edges = [Segment(self.vertices[i], self.vertices[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if edges[i].is_intersecting(edges[j]):
                    return True
        return False

    def get_centroid(self, diff=False):
        centroid = sum(self.vertices, Vertex(0, 0)) / len(self.vertices)
        if diff:
            _diff = sum(self.edge_lengths()) / len(self.vertices)
            centroid.x -= _diff / 3
            centroid.y -= _diff * 0.1
        return centroid


class Quad(Polygon):
    def __init__(self, vertices):
        if len(vertices) != 4:
            raise ValueError(f"A Quad requires exactly 4 vertices, got {len(vertices)}.")
        super().__init__(vertices)


class Angle:
    @staticmethod
    def clip_angle(angle, max_angle):
        return min(angle, max_angle + math.pi / 2)

    @staticmethod
    def get_points_within_angle(vertices, base_point, start_point, start_angle, end_angle):
        target_points = [v for v in vertices
            if start_angle < base_point.clockwise_angle(start_point, v) < end_angle]
        return target_points


class Lin_Alg:
    @staticmethod
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

    @staticmethod
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


def circle_line_intersection(a, b, A, B, W, dist):
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
        inter_v = [v for v in vertices[i].get_neighbors()
                   if v in vertices[i - 1].get_neighbors() and v is not inner_v]
        final_vertices.append(vertices[i - 1])
        if inter_v:
            final_vertices.append(inter_v[0])
    return final_vertices
