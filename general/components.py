import math

import numpy as np


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Point2D(self.x, self.y)

    def distance_to(self, point):
        return math.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"({self.x, self.y})"

    def __mul__(self, other):
        return Point2D(self.x * other, self.y * other)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)


class Vertex(Point2D):
    def __init__(self, x, y):
        super(Vertex, self).__init__(x, y)
        self.segments = None

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

    def to_find_clockwise_angle(self, point1, point2):
        v1 = point1 - self
        v2 = point2 - self

        theta = - math.atan2(v1.x * v2.y - v1.y * v2.x, v1.x * v2.x + v1.y * v2.y)

        return round(theta, 4) if math.copysign(1, theta) >= 0 else round(2 * math.pi + theta, 4)

    def has_segment_with_vertex(self, vertex):
        if not self.segments:
            return False
        for seg in self.segments:
            if seg.has_vertex(vertex):
                return True
        return False

    def get_connected_vertices(self):
        if not self.segments:
            return []

        vertices = []
        for seg in self.segments:
            if seg.point1 not in vertices:
                vertices.append(seg.point1)

            if seg.point2 not in vertices:
                vertices.append(seg.point2)

        vertices.remove(self)

        return vertices

    def copy(self):
        return Vertex(self.x, self.y)

    @staticmethod
    def rotate(point, angle, origin=None):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        if origin is None:
            ox, oy = 0, 0
        else:
            ox, oy = origin.x, origin.y
        px, py = point.x, point.y

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return Vertex(qx, qy)

    def get_common_vertex(self, another_vertex):
        connected_v = self.get_connected_vertices()
        another_vertex_v = another_vertex.get_connected_vertices()
        return [v for v in connected_v if v in another_vertex_v]


class Boundary2D:
    def __init__(self, vertices):
        self.vertices = vertices

    def copy(self):
        return Boundary2D([vertex for vertex in self.vertices])

    def deep_copy(self):
        points = [vertex.copy() for vertex in self.vertices]
        for i in range(len(points)):
            segmt = Segment(points[i - 1], points[i])
            points[i - 1].assign_segment(segmt)
            points[i].assign_segment(segmt)

        return Boundary2D(points)

    def all_segments(self):
        segts = []
        for vertex in self.vertices:
            if vertex.segments:
                for segt in vertex.segments:
                    if segt not in segts and (segt.point1 in self.vertices and segt.point2 in self.vertices):
                        segts.append(segt)
        return segts

    def sort_segments_by_length(self, reverse=False):
        segts = self.all_segments()
        sorted_segts = sorted([(seg, seg.length()) for seg in segts], key=lambda x: x[1], reverse=reverse)
        return sorted_segts

    @staticmethod
    def compute_dist(vertices, point):
        dists = []
        for vertex in vertices:
            if point is not vertex:
                dist = point.distance_to(vertex)
                dists.append((vertex, dist))
        # dists = sorted(dists, key=lambda x: x[1])
        return sorted(dists, key=lambda x: x[1])

    @staticmethod
    def get_closet_point(vertices, point, exclusion=None, S_T=None):
        points = Boundary2D.get_closet_points(vertices, point, exclusion, S_T)
        return points[0] if len(points) else None

    @staticmethod
    def get_closet_points(vertices, point, exclusion=None, S_T=None):
        dists = Boundary2D.compute_dist(vertices, point)
        points = []
        if exclusion:
            for v in dists:
                if v[0] not in exclusion and v[1] <= S_T:
                    points.append(v[0])
                if v[1] > S_T:
                    break
        else:
            for v in dists:
                if v[1] <= S_T:
                    points.append(v[0])
                else:
                    break
        return points

    @staticmethod
    def get_points_within_angle(vertices, base_point, start_point, start_angle, end_angle):
        target_points = [v for v in vertices
            if start_angle < base_point.to_find_clockwise_angle(start_point, v) < end_angle]
        return target_points

    def get_centriod(self):
        if not len(self.vertices):
            return
        centriod = Vertex(0, 0)
        for v in self.vertices:
            centriod += v
        centriod.x /= len(self.vertices)
        centriod.y /= len(self.vertices)
        return centriod

    def get_neighbors(self, point, num_points=4):
        half = int(num_points / 2)
        if num_points % 2 != 0:
            raise ValueError("The neighbor number is not even!")
        p_num = len(self.vertices)
        index = self.vertices.index(point)

        vertices = [self.vertices[(index + i) % p_num] for i in reversed(range(half + 1))]
        [vertices.append(self.vertices[index - i]) for i in range(1, half + 1)]
        return vertices

    def average_edge_length(self):
        _length = len(self.vertices)
        dist = 0
        for i in range(_length):
            dist += self.vertices[i].distance_to(self.vertices[i-1])
        return round(dist / _length, 4) if _length != 0 else 0

    def get_perimeter(self):
        perimeter = 0
        for i in range(1, len(self.vertices)):
            perimeter += self.vertices[i - 1].distance_to(self.vertices[i])
        return perimeter

    def compute_boundary_angle(self, vertex):
        if vertex in self.vertices:
            index = self.vertices.index(vertex)
            right_v = self.vertices[index - 1]
            left_v = self.vertices[(index + 1) % len(self.vertices)]
            angle = math.degrees(vertex.to_find_clockwise_angle(left_v, right_v))
            return angle

        return None

    def poly_area(self):
        xy = np.array([[v.x, v.y] for v in self.vertices])
        return 0.5 * np.abs(np.dot(xy[:, 0],np.roll(xy[:, 1],1))-np.dot(xy[:, 1],np.roll(xy[:, 0],1)))


def cross_product(v1, v2):
    return v1.x*v2.y - v2.x*v1.y


class Segment:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def straddle(self, another_segment):
        v1 = another_segment.point1 - self.point1
        v2 = another_segment.point2 - self.point1
        vm = self.point2 - self.point1

        # check if two segments are colinear
        if round(math.sin(self.point1.to_find_clockwise_angle(another_segment.point1, self.point2)), 4) == \
            round(math.sin(self.point1.to_find_clockwise_angle(another_segment.point2, self.point2)), 4) and \
                round(math.sin(self.point1.to_find_clockwise_angle(another_segment.point2, self.point2)), 4) == 0:
            l1 = self.length()
            l2 = another_segment.length()
            if l1 > l2:
                m = (self.point2 + self.point1) / 2
                if min(m.distance_to(another_segment.point2), m.distance_to(another_segment.point1)) <= l1/2:
                    return True
            else:
                m = (another_segment.point2 + another_segment.point1) / 2
                if min(m.distance_to(self.point2), m.distance_to(self.point1)) <= l2 / 2:
                    return True
            return False

        if cross_product(v1, vm) * cross_product(v2, vm) <= 0:
            return True
        else:
            return False

    def is_cross(self, another_segment):
        return self.straddle(another_segment) and another_segment.straddle(self)

    def seg_angle(self):
        theta = math.atan2(self.point2.y - self.point1.y, self.point2.x - self.point1.x)
        return theta

    @staticmethod
    def get_ray_segment(segment, another_segment, remote_dist):
        ray_point1 = Vertex((segment.point1 + another_segment.point1).x / 2,
                            (segment.point1 + another_segment.point1).y / 2)

        ray_point2 = Vertex((segment.point2 + another_segment.point2).x / 2,
                            (segment.point2 + another_segment.point2).y / 2)

        average_segment = Segment(ray_point1, ray_point2)
        return Segment.build_ray(average_segment, remote_dist)

    @staticmethod
    def build_ray(segment, remote_dist):
        average_segment = Segment(segment.point1.copy(), segment.point2.copy())
        theta = average_segment.seg_angle()

        average_segment.point2.x = average_segment.point1.x + remote_dist * math.cos(theta)
        average_segment.point2.y = average_segment.point1.y + remote_dist * math.sin(theta)

        return average_segment

    def __str__(self):
        return f"Segment({self.point1}, {self.point2})"

    def has_vertex(self, vertex):
        if vertex == self.point1 or vertex == self.point2:
            return True
        else:
            return False

    def perpendicular_point(self, point):
        a = self.point1.x
        b = self.point1.y
        A = self.point2.x - self.point1.x
        B = self.point2.y - self.point1.y
        s = (A * point.x + B * point.y - B * b - A * a) / (A ** 2 + B ** 2)
        target = Vertex(a+s*A, b+s*B)
        return target, point.distance_to(target), True if 0 <= s <= 1 else False

    def length(self):
        return self.point1.distance_to(self.point2)

    def intersection_vertex(self, another_seg):
        u = self.point2 - self.point1
        w = another_seg.point2 - another_seg.point1
        if w.y == 0:
            if u.y == 0:
                return None, None
            s = (another_seg.point1.y - self.point1.y) / u.y
            h = (self.point1.x - another_seg.point1.x + s * u.x) / w.x
        else:
            if w.x == 0:
                if u.x == 0:
                    return None, None
                s = (another_seg.point1.x - self.point1.x) / u.x
                h = (self.point1.y - another_seg.point1.y + s * u.y) / w.y
            else:
                s = ((self.point1.x - another_seg.point1.x) / w.x - (self.point1.y - another_seg.point1.y) / w.y) / (u.y/w.y - u.x/w.x)
                h = (self.point1.x - another_seg.point1.x + s * u.x) / w.x
        is_inside = True if 0 < s < 1 and 0 < h < 1 else False
        return is_inside, Vertex(self.point1.x + s * u.x, self.point1.y + s * u.y)

    def distance(self, another):
        dists = []
        if isinstance(another, Vertex):
            a = self.point1.x
            b = self.point1.y
            A = self.point2.x - self.point1.x
            B = self.point2.y - self.point1.y
            s = (A * another.x + B * another.y - B * b - A * a) / (A ** 2 + B ** 2)
            if 0 <= s <= 1:
                target = Vertex(a + s * A, b + s * B)
                return another.distance_to(target)
            elif s < 0:
                return another.distance_to(self.point1)
            else:
                return another.distance_to(self.point2)
        elif isinstance(another, Segment):
            dists.append(self.distance(another.point1))
            dists.append(self.distance(another.point2))
            dists.append(another.distance(self.point1))
            dists.append(another.distance(self.point2))
            return min(dists)
        else:
            raise ValueError('Not recognized object type!')


class Mesh:
    def __init__(self, vertices):
        self.vertices = vertices
        self.segments = None

        self.max_aspect_ratio = 5
        # self.min_aspect_ratio = 0.3
        # self.max_taper_ratio = 3
        # self.min_taper_ratio = 0.3
        self.max_degree = 0.99 * math.pi
        self.min_degree = 0.01 * math.pi

    def length_4_segments(self):
        return [Segment(self.vertices[i - 1], self.vertices[i]).length() for i in range(len(self.vertices))]

    def is_valid(self, quality_method=0):
        if self.segments_crossed():
            return False

        if quality_method == 0:
            for i in range(len(self.vertices)):
                degree = self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])
                if degree > self.max_degree or degree < self.min_degree:
                    return False
        elif quality_method == 1:
            if self.get_quality() < 0.3:
                return False
        elif quality_method == 2:
            if self.get_quality_2() < 0.8:
                return False
        elif quality_method == 3:
            if self.get_quality() < 0.3 or self.get_quality_2() < 0.8:
                return False
        elif quality_method == 4:
            if self.get_quality() < 0.2:
                return False
        elif quality_method == 5:
            for i in range(len(self.vertices)):

                degree = self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])
                # degree = self.vertices[i].to_find_clockwise_angle(self.vertices[i - 1], self.vertices[(i + 1) % 4])

                if degree > self.max_degree or degree < self.min_degree:
                    return False

            if self.get_quality() < 0.3:
                return False
        elif quality_method == 6:
            for i in range(len(self.vertices)):

                degree = self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])
                # degree = self.vertices[i].to_find_clockwise_angle(self.vertices[i - 1], self.vertices[(i + 1) % 4])

                if degree > self.max_degree or degree < self.min_degree:
                    return False

            q1, q2 = self.get_quality_3()
            if q1 * q2 < 0.5:
                return False
        elif quality_method == 7:
            if self.get_aspect_ratio() > self.max_aspect_ratio:
                return False
            for i in range(len(self.vertices)):

                degree = self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])
                # degree = self.vertices[i].to_find_clockwise_angle(self.vertices[i - 1], self.vertices[(i + 1) % 4])

                if degree > self.max_degree or degree < self.min_degree:
                    return False

            q1, q2 = self.get_quality_3()
            if q1 * q2 <= 0.2:
                return False
        return True

    def segments_crossed(self):

        seg1 = Segment(self.vertices[0], self.vertices[1])
        seg2 = Segment(self.vertices[2], self.vertices[3])
        if seg1.is_cross(seg2):
            return True

        seg1 = Segment(self.vertices[0], self.vertices[3])
        seg2 = Segment(self.vertices[1], self.vertices[2])
        return seg1.is_cross(seg2)

    def get_centriod(self, diff=False):
        ave_point = Vertex(0, 0)
        for v in self.vertices:
            ave_point += v

        _diff = sum([self.vertices[i].distance_to(self.vertices[i - 1]) for i in range(4)]) / 4

        ave_point.x = ave_point.x / len(self.vertices) - (_diff/3 if diff else 0)
        ave_point.y = ave_point.y / len(self.vertices) - (_diff*0.1 if diff else 0)

        return ave_point

    def connect_vertices(self):
        for i in range(len(self.vertices)):
            if not self.vertices[i].has_segment_with_vertex(self.vertices[i - 1]):
                new_seg = Segment(self.vertices[i], self.vertices[i - 1])
                self.vertices[i].assign_segment(new_seg)
                self.vertices[i - 1].assign_segment(new_seg)

    def get_aspect_ratio(self):
        dists = []
        for i in range(4):
            dists.append(self.vertices[i].distance_to(self.vertices[i - 1]))

        return max(dists) / min(dists) if min(dists) != 0 else 0.001

    def inner_angles(self):
        angles = []
        for i in range(4):
            angles.append(math.degrees(math.fabs(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1]) - math.pi / 2)))
            # angles.append(math.degrees(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])))
        return angles

    def get_ave_error_angle(self):
        angles = []
        for i in range(4):
            angles.append(math.fabs(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1]) - math.pi / 2))
        return max(angles)

    def get_quality(self, type='default'):
        if type == 'default':
            aspect_ratio = self.get_aspect_ratio()
            ave_error_angle = self.get_ave_error_angle()
            return 1 / (aspect_ratio + ave_error_angle)
        elif type =='stretch':
            return math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]), self.vertices[1].distance_to(self.vertices[3]))
        elif type == 'robust':
            q1 = math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]), self.vertices[1].distance_to(self.vertices[3]))
            # segts = self.length_4_segments()
            # q1 = min(segts) / max(segts)
            angles = []
            for i in range(4):
                angles.append(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1]))
            q2 = min(angles) / max(angles)
            return math.sqrt(q1 * q2) #(q1 + q2) /2
        elif type == 'taper':
            p0, p1, p2, p3 = self.vertices[0], self.vertices[-1], self.vertices[-2], self.vertices[-3]
            x1 = (p1 - p0) + (p2 - p3)
            x2 = (p2 - p1) + (p3 - p0)
            x12 = (p0 - p1) + (p2 - p3)
            return x12.length() / min(x1.length(), x2.length())
        elif type == 's_jacobian':
            p0, p1, p2, p3 = self.vertices[0], self.vertices[-1], self.vertices[-2], self.vertices[-3]
            l0, l1, l2, l3 = p1-p0, p2-p1, p3-p2, p0-p3
            a3 = cross_product(l2, l3)
            a2 = cross_product(l1, l2)
            a1 = cross_product(l0, l1)
            a0 = cross_product(l3, l0)
            return min([a0 / (l0.length() * l3.length()),
                     a1 / (l0.length() * l1.length()),
                     a2 / (l1.length() * l2.length()),
                     a3 / (l2.length() * l3.length())])
        elif type == 'strong':
            q1, _ = self.get_quality_3()
            # q1 = math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]), self.vertices[1].distance_to(self.vertices[3]))
            angles = []
            for i in range(4):
                angles.append(math.fabs(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])))
            q2 = min(angles) / max(angles)
            # angle_product = 1
            # for i in range(4):
            #     angle_product *= 1 - (math.fabs(math.degrees(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])) - 90) / 90)
            # if angle_product < 0:
            #     q2 = 0
            # else:
            #     q2 = math.pow(angle_product, 1 / 4)
            # q2 = angle_product

            return math.sqrt(q1 * q2)
        elif type == 'area':
            q1, q2 = self.get_quality_3()
            return q1 * q2
        raise ValueError(f"Unknown quality type: {type}")

    def compute_area(self):
        length_of_edges = [self.vertices[i].distance_to(self.vertices[i - 1]) for i in range(4)]

        corner_1 = self.vertices[0].to_find_clockwise_angle(self.vertices[1], self.vertices[-1])
        corner_3 = self.vertices[2].to_find_clockwise_angle(self.vertices[3], self.vertices[1])

        area = 0.5 * length_of_edges[0] * length_of_edges[1] * math.sin(corner_1) + \
               0.5 * length_of_edges[2] * length_of_edges[3] * math.sin(corner_3)

        # s = sum(length_of_edges) / 2
        # product = 1
        # for edge in length_of_edges:
        #     product *= s - edge
        # area = math.sqrt(product)

        return area, length_of_edges

    def get_quality_3(self):
        area, length_of_edges = self.compute_area()
        product = 1
        if area <= 0:
            q1 = 0
        else:
            for edge in length_of_edges:
                product *= math.pow(edge / math.sqrt(area), 1 if math.sqrt(area) - edge > 0 else -1)
            q1 = math.pow(product, 1 / 4)

        angle_product = 1
        for i in range(4):
            angle_product *= 1 - (math.fabs(math.degrees(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])) - 90) / 90)
        if angle_product < 0:
            q2 = 0
        else:
            q2 = math.pow(angle_product, 1/4)

        return q1, q2

    def get_quality_2(self):
        area, length_of_edges = self.compute_area()
        circumstance = sum(length_of_edges)
        return area/((circumstance/4) * (circumstance/4))

    @staticmethod
    def estimate_4th_vertex(origin_p, left_p, right_p, factor=0.5, suggest_dist=None):
        distance = (origin_p.distance_to(left_p) + origin_p.distance_to(right_p)) * factor

        if suggest_dist is not None:
            distance = min(distance, 0.6 * suggest_dist)

        s = Segment.get_ray_segment(Segment(origin_p, left_p), Segment(origin_p, right_p), distance)
        return s.point2