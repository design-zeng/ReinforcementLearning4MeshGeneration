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
        vertices = []
        for seg in self.segments or []:
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

    def get_closest_points(self, vertices, exclusion, max_dist):
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

    def distance(self, vertex):
        d = self.point2 - self.point1
        w = vertex - self.point1
        s = max(0.0, min(1.0, d.dot(w) / d.length() ** 2))
        return vertex.distance_to(self.point1 + d * s)

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
        segments = []
        for vertex in self.vertices:
            for seg in vertex.segments or []:
                if seg not in segments and seg.point1 in self.vertices and seg.point2 in self.vertices:
                    segments.append(seg)
        return segments

    def n_sides(self, vertex):
        return sum(1 for seg in vertex.segments
                   if seg.point1 in self.vertices and seg.point2 in self.vertices)

    def get_neighbors(self, vertex, num_points):
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
            current, previous = self.vertices[i], self.vertices[i - 1]
            orientation = round(current.y - previous.y, 4)
            if orientation == 0:
                continue
            if not Segment(current, previous).is_intersecting(ray_segment):
                continue
            if round(current.y - ray_y, 4) == 0:
                next_orientation = round(self.vertices[(i + 1) % n].y - current.y, 4)
                count += (next_orientation * orientation > 0 and orientation < 0)
            elif round(previous.y - ray_y, 4) == 0:
                previous_orientation = round(previous.y - self.vertices[i - 2].y, 4)
                count += (previous_orientation * orientation > 0 and orientation > 0)
            else:
                count += 1
        return count % 2 != 0

    def find_same_point(self, point):
        return next((p for p in self.vertices if p.distance_to(point) < 0.001), None)

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

    def is_self_intersecting(self):
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

    def is_valid(self):
        return not self.is_self_intersecting() and \
            all(0.01 * math.pi <= a <= 0.99 * math.pi for a in self.corner_angles())


class Mesh:
    def __init__(self, polygon):
        self.original_vertices = list(polygon.vertices)
        self.original_area = polygon.area()
        self.generated_quads = []

    def vertices(self):
        vertices = list(self.original_vertices)
        for quad in self.generated_quads:
            vertices.extend(v for v in quad.vertices if v not in vertices)
        return vertices

    def add_quad(self, quad):
        quad.connect_vertices()
        self.generated_quads.append(quad)

    def find_related_quads(self, vertex):
        return list({m for m in self.generated_quads if vertex in m.vertices})


def circle_line_intersection(a, b, A, B, W, dist):
    if B == 0:
        h = math.sqrt(dist ** 2 - (W / A) ** 2)
        x1 = x2 = W / A + a
        y1, y2 = b + h, b - h
    elif A == 0:
        h = math.sqrt(dist ** 2 - (W / B) ** 2)
        x1, x2 = a + h, a - h
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
        flag = max(range(i, len(vertices)),
                   key=lambda j: inner_v.clockwise_angle(vertices[j], vertices[i - 1]))
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
