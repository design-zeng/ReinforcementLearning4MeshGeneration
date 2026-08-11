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
