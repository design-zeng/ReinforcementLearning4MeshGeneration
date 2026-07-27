import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Point2D(self.x, self.y)

    def distance_to(self, point):
        return math.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def show(self, style='b.'):
        plt.plot(self.x, self.y, style)
        plt.gca().set_aspect('equal', adjustable='box')

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

#
# A = Point2D(0, 1)
# B = Point2D(2, 5.1)
# C = Point2D(1, 3)
# D = Point2D(0.99, 1.2)
# AB = Segment(A, B)
# CD = Segment(C, D)
# r = AB.cross(CD)
# print()


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

    def find_angle(self, another_vertex):
        theta = np.math.atan2(another_vertex.y - self.y, another_vertex.x - self.x)
        return theta

    def angle(self):
        theta = np.math.atan2(self.y, self.x)
        return theta

    def to_find_angle(self, point1, point2):
        theta1 = math.atan2(point1.y - self.y, point1.x - self.x)

        theta2 = math.atan2(point2.y - self.y, point2.x - self.x)

        diff = math.fabs(theta1 - theta2)
        theta = min(diff, math.fabs(math.pi - diff))

        return theta

    def to_find_clockwise_angle(self, point1, point2):
        v1 = point1 - self
        v2 = point2 - self

        # theta = - math.asin((vector_a.x * vector_b.y - vector_a.y * vector_b.x) /
        #                   (self.distance_to(point1) * self.distance_to(point2)))

        theta = - math.atan2(v1.x * v2.y - v1.y * v2.x, v1.x * v2.x + v1.y * v2.y)

        return round(theta, 4) if math.copysign(1, theta) >= 0 else round(2 * math.pi + theta, 4)

    def has_segment_with_vertex(self, vertex):
        if not self.segments:
            return False
        for seg in self.segments:
            if seg.has_vertex(vertex):
                return True
        return False

    @staticmethod
    def get_random_vertex():
        random.seed()
        return Vertex(random.randint(-10, 10), random.randint(-10, 10))

    def get_connected_vertices(self):
        vertices = []
        for seg in self.segments:
            if seg.point1 not in vertices:
                vertices.append(seg.point1)

            if seg.point2 not in vertices:
                vertices.append(seg.point2)
        vertices.remove(self)
        return vertices

    def equal(self, another_point):
        if self.x == another_point.x and self.y == another_point.y:
            return True
        else:
            return False

    def copy(self):
        return Vertex(self.x, self.y)

    def sampling_between_endpoints(self, end_vertex, size, is_even=False):
        dist = self.distance_to(end_vertex)
        if is_even:
            finals = np.arange(0, dist, dist / (size))
        else:
            samples = np.random.uniform(0, dist, size * 2)
            finals = sorted(np.random.choice(samples, size, replace=False))
        angle = self.find_angle(end_vertex)
        return [Vertex(self.x + s * math.cos(angle), self.y + s * math.sin(angle)) for s in finals]

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

    def get_mid_vertex(self, another_vertex):
        return Vertex(self.x + another_vertex.x, self.y + another_vertex.y) / 2

    def perpendicular_vector(self, another_vertex):
        u = (self.x - another_vertex.x, self.y - another_vertex.y)
        if self.y - another_vertex.y == 0:
            return (0, 1)
        else:
            v = (1, (self.x - another_vertex.x) / (another_vertex.y - self.y))
            return v

    def get_mid_perpendicular_line(self, another_vertex):
        v = self.perpendicular_vector(another_vertex)
        m_vertex = self.get_mid_vertex(another_vertex)
        b = m_vertex.y - v[1] * m_vertex.x
        return v[1], b

    def get_perpendicular_vertex(self, another_vertex):
        m_v = self.get_mid_vertex(another_vertex)
        v = self.perpendicular_vector(another_vertex)
        dist = self.distance_to(another_vertex) / 2
        if v[0] == 0:
            s = dist
        else:
            s = dist / math.sqrt(1+ v[1]**2)
        return Vertex(m_v.x + s, m_v.y + s * v[1]), Vertex(m_v.x - s, m_v.y - s * v[1])

# v0 = Vertex(0, 0)
# v1 = Vertex(0, 3)
# v2 = Vertex(-3, 0)
# print(math.degrees(v0.to_find_clockwise_angle(v1, v2)))

# samples = v1.sampling_between_endpoints(end_vertex=v2, size=5, is_even=True)
# for i in samples:
#     i.show()
# plt.show()
# print()
# print(Vertex(1, 1).angle())

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


    def show(self, style='b.-', linewidth=1, markersize=6, show=True):
        segts = self.all_segments()
        for segt in segts:
            if segt.point1 not in self.vertices or segt.point2 not in self.vertices:
                continue
            segt.show(style=style, linewidth=linewidth, markersize=markersize)
        plt.gca().set_aspect('equal', adjustable='box')
        if show:
            plt.show()

    def plot(self, style='b.-', linewidth=2, markersize=10):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        segts = self.all_segments()
        x, y = [], []
        for segt in segts:
            if segt.point1 not in self.vertices or segt.point2 not in self.vertices:
                continue
            segt.show(style=style, linewidth=linewidth, markersize=markersize)
            x.extend([segt.point1.x, segt.point2.x])
            y.extend([segt.point1.y, segt.point2.y])

        ax.set_frame_on(False)
        # ax.margins(0.05)
        plt.gca().set_xlim([min(x) - 0.1, max(x) + 0.1])
        plt.gca().set_ylim([min(y) - 0.1, max(y) + 0.1])
        plt.xticks([])
        plt.yticks([])

    def savefig(self, name, title="", style="k.-", dpi=600):
        sns.set_context('paper')
        fig = plt.figure()
        # fig = plt.figure(figsize=(550 / dpi, 480 / dpi), dpi=dpi)

        ax = fig.add_subplot(111)
        ax.set_title(title)
        segts = self.all_segments()
        x, y = [], []
        for segt in segts:
            if segt.point1 not in self.vertices or segt.point2 not in self.vertices:
                continue
            segt.show(style=style, linewidth=2, markersize=10)
            x.extend([segt.point1.x, segt.point2.x])
            y.extend([segt.point1.y, segt.point2.y])

        ax.set_frame_on(False)
        # ax.margins(0.05)
        plt.gca().set_xlim([min(x) - 0.1, max(x) + 0.1])
        plt.gca().set_ylim([min(y) - 0.1, max(y) + 0.1])
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.gca().set_axis_off()
        # plt.gca().set_aspect('equal')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=-0, hspace=0, wspace=0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # plt.show()
        plt.savefig(name, dpi=dpi)
        plt.close('all')

    def save_intermediate_boundary_fig(self, name, boundary_vs, title="", style="k.-", dpi=300, r_vertices=None):
        # sns.set_context('paper')
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        segts = self.all_segments()
        x, y = [], []
        for segt in segts:
            if segt.point1 not in self.vertices or segt.point2 not in self.vertices:
                continue
            x.extend([segt.point1.x, segt.point2.x])
            y.extend([segt.point1.y, segt.point2.y])

            if r_vertices is not None:
                if segt.point1 in r_vertices and segt.point2 in r_vertices:
                    segt.show(style="b.-", linewidth=2, markersize=10)
                else:
                    segt.show(style=style, linewidth=2, markersize=10)

                if segt.point1 in boundary_vs and segt.point2 in boundary_vs:
                    segt.show(style="r.-", linewidth=2, markersize=10)
            else:
                if segt.point1 in boundary_vs and segt.point2 in boundary_vs:
                    segt.show(style="r.-", linewidth=2, markersize=10)
                else:
                    segt.show(style=style, linewidth=2, markersize=10)

        ax.set_frame_on(False)
        # ax.margins(0.05)
        plt.gca().set_xlim([min(x) - 0.1, max(x) + 0.1])
        plt.gca().set_ylim([min(y) - 0.1, max(y) + 0.1])
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(name, dpi=dpi)
        plt.close('all')

    def save_vertices_into_fig(self, name, boundary_vs, title="", style="k.-", dpi=300):
        sns.set_context('paper')
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_subplot(111)
        ax.set_title(title)
        segts = self.all_segments()
        x, y = [], []
        for segt in segts:
            if segt.point1 not in self.vertices or segt.point2 not in self.vertices:
                continue

            if segt.point1 in boundary_vs and segt.point2 in boundary_vs:
                segt.show(style=style, linewidth=1, markersize=6)
                x.extend([segt.point1.x, segt.point2.x])
                y.extend([segt.point1.y, segt.point2.y])

        ax.set_frame_on(False)
        # ax.margins(0.05)
        plt.gca().set_xlim([min(x) - 0.1, max(x) + 0.1])
        plt.gca().set_ylim([min(y) - 0.1, max(y) + 0.1])
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.gca().set_axis_off()
        # plt.gca().set_aspect('equal')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=-0, hspace=0, wspace=0)
        # plt.show()
        plt.savefig(name, dpi=dpi)
        plt.close('all')

    @staticmethod
    def compute_dist(vertices, point):
        dists = []
        for vertex in vertices:
            if point is not vertex:
                dist = point.distance_to(vertex)
                dists.append((vertex, dist))
        # dists = sorted(dists, key=lambda x: x[1])
        return sorted(dists, key=lambda x: x[1])

    def get_remotest_point(self, point):
        dists = self.compute_dist(self.vertices, point)
        return dists[-1][0], dists[-1][1] + 1

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
    def get_points_within_two_edges(vertices, base_point, start_point, end_point):
        target_angle = base_point.to_find_clockwise_angle(start_point, end_point)
        return Boundary2D.get_points_within_angle(vertices, base_point, start_point, 0, target_angle)

    @staticmethod
    def get_points_within_angle(vertices, base_point, start_point, start_angle, end_angle):
        target_points = [v for v in vertices if start_angle < base_point.to_find_clockwise_angle(start_point, v)
                         < end_angle]
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

    def cal_segts_len(self):
        lens = [s.__len__() for s in self.all_segments()]
        return sorted(lens)

    def average_edge_length(self):
        _length = len(self.vertices)
        dist = 0
        for i in range(_length):
            dist += self.vertices[i].distance_to(self.vertices[i-1])
        return round(dist / _length, 4) if _length != 0 else 0

    def segmts_diff_ratio(self):
        res = self.cal_segts_len()
        r1 = min((res[0], res[1])) / max((res[0], res[1]))
        r2 = min((res[-2], res[-1])) / max((res[-2], res[-1]))
        r3 = res[0] / res[1]
        min_r = min((r1, r2, r3))
        max_r = max((r1, r2, r3))
        return min_r, max_r

    def get_perimeter(self):
        perimeter = 0
        for i in range(1, len(self.vertices)):
            perimeter += self.vertices[i - 1].distance_to(self.vertices[i])
        return perimeter

    def get_boundary_quality(self):
        pass

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
        vm_2 = another_segment.point2 - another_segment.point1

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

        # if one end point of a segment is on the another segment

        # theta1 = self.angle_from_point_2_segment(self.point1, another_segment)
        # theta2 = self.angle_from_point_2_segment(self.point2, another_segment)
        #
        # if round(theta1, 4) == 180 or round(theta2, 4) == 180:
        #     return True
        # elif round(theta1, 4) == 0 and round(theta2, 4) == 0:
        #     return False

        if self.straddle(another_segment) and another_segment.straddle(self):
            return True
        else:
            return False


    def angle_from_point_2_segment(self, point, segment):
        v1 = segment.point1 - point
        v2 = segment.point2 - point
        if v1.length() != 0 and v2.length() != 0:
            theta = math.acos(round((v1.x * v2.x + v1.y * v2.y) / (v1.length() * v2.length()), 6))
            return math.degrees(theta)
        else:
            return 0

    def cross(self, another_segment):
        if self.is_cross(another_segment):
            print("Two segments crossed")
        else:
            print("Two segments are not crossed!")

        plt.plot([self.point1.x, self.point2.x], [self.point1.y, self.point2.y], 'b-')
        plt.plot([another_segment.point1.x, another_segment.point2.x],
                 [another_segment.point1.y, another_segment.point2.y], 'r-')
        plt.show()

    def show(self, style='b.-', linewidth=2, markersize=0.1):
        # sns.set()
        # plt.plot([self.point1.x, self.point2.x], [self.point1.y, self.point2.y], style,
        #                   linewidth=linewidth, markersize=markersize)
        # ax = sns.lineplot([self.point1.x, self.point2.x], [self.point1.y, self.point2.y],
        #                   linewidth=linewidth, markersize=markersize, marker='.')
        plt.plot([self.point1.x, self.point2.x], [self.point1.y, self.point2.y], style,
                 linewidth=linewidth, markersize=markersize)
        plt.gca().set_aspect('equal', adjustable='box')

    @staticmethod
    def angle(segment, another_segment):

        theta1 = segment.seg_angle()
        theta2 = another_segment.seg_angle()

        # delta_y = ((segment.point2 - segment.point1) - (another_segment.point2 - another_segment.point1)).y
        # delta_x = ((segment.point2 - segment.point1) - (another_segment.point2 - another_segment.point1)).x
        # theta = np.math.atan2(delta_y, delta_x)

        # print(math.degrees(theta1), math.degrees(theta2))
        diff = math.fabs(theta1 - theta2)
        theta = min(diff, math.fabs(math.pi-diff))

        return theta

    def seg_angle(self):
        theta = math.atan2(self.point2.y - self.point1.y,
                            self.point2.x - self.point1.x)
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

        # print(average_segment, remote_dist, math.degrees(theta), )

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

    def __len__(self):
        return self.point1.distance_to(self.point2)

    def is_inner_perpendicular(self, point):
        a = self.point1.x
        b = self.point1.y
        A = self.point2.x - self.point1.x
        B = self.point2.y - self.point1.y
        s = (A*point.x + B*point.y -B*b - A*a)/(A**2 + B**2)
        if 0 <= s <= 1:
            return True
        return False

    def perpendicular_point(self, point):
        a = self.point1.x
        b = self.point1.y
        A = self.point2.x - self.point1.x
        B = self.point2.y - self.point1.y
        s = (A * point.x + B * point.y - B * b - A * a) / (A ** 2 + B ** 2)
        target = Vertex(a+s*A, b+s*B)
        return target, point.distance_to(target), True if 0 <= s <= 1 else False

    def vector(self):
        return (self.point2.x - self.point1.x, self.point2.y - self.point1.y)

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
                s = ((self.point1.x - another_seg.point1.x) / w.x - (self.point1.y - another_seg.point1.y) / w.y) / \
                    (u.y/w.y - u.x/w.x)
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


# S1 = Segment(Vertex(1, 1), Vertex(3, 3))
# S2 = Segment(Vertex(0, 0), Vertex(4, 4))
# vertex = Vertex(-1, 5.5)
# print(S1.distance(S2))
# print(S1.is_inner_perpendicular(vertex))

# S2.show()
# plt.show()
# S3 = Segment(Vertex(0, 1), Vertex(2, 0))
# print(S1.is_cross(S2))
# print(S2.is_cross(S3))
# s1 = Segment(Vertex(-1, -1), Vertex(1, 1))
# s2 = Segment(Vertex(2, 1), Vertex(3, 1))
# is_inside, v = s1.intersection_vertex(s2)
# print(is_inside, v)

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

    def get_all_segments(self):
        return [Segment(self.vertices[i - 1], self.vertices[i]) for i in range(len(self.vertices))]

    def length_4_segments(self):
        return [Segment(self.vertices[i - 1], self.vertices[i]).length() for i in range(len(self.vertices))]

    def is_valid(self, quality_method=0):
        if self.segments_crossed():
            return False

        if quality_method == 0:
            # if self.get_aspect_ratio() > self.max_aspect_ratio:
            #     return False

            for i in range(len(self.vertices)):

                # if self.vertices[i].segments:
                #     if len(self.vertices[i].get_connected_vertices()) >= 5:
                #         return False

                degree = self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])

                # d = math.degrees(degree)
                if degree > self.max_degree or degree < self.min_degree:
                    # print(f"This mesh's corner degree is {d}")
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
                    # print(f"This mesh's corner degree is {d}")
                    return False

            if self.get_quality() < 0.3:
                return False

        elif quality_method == 6:
            for i in range(len(self.vertices)):

                degree = self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4], self.vertices[i - 1])
                # degree = self.vertices[i].to_find_clockwise_angle(self.vertices[i - 1], self.vertices[(i + 1) % 4])

                if degree > self.max_degree or degree < self.min_degree:
                    # print(f"This mesh's corner degree is {d}")
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
                    # print(f"This mesh's corner degree is {d}")
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
        if seg1.is_cross(seg2):
            return True
        else:
            return False

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
            angles.append(math.degrees(math.fabs(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4],
                                                                             self.vertices[i - 1]) - math.pi / 2)))
            # angles.append(math.degrees(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4],
            #                                                                               self.vertices[i - 1])))
        return angles

    def get_ave_error_angle(self):
        angles = []
        for i in range(4):
            angles.append(math.fabs(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4],
                                                                             self.vertices[i - 1]) -
                          math.pi / 2))
        return max(angles)

    def get_quality(self, type='default'):
        if type == 'default':
            aspect_ratio = self.get_aspect_ratio()
            ave_error_angle = self.get_ave_error_angle()
            # print("aspect:", aspect_ratio)
            # print("angle error:", ave_error_angle)
            return 1 / (aspect_ratio + ave_error_angle)
        elif type =='stretch':
            return math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]),
                                                               self.vertices[1].distance_to(self.vertices[3]))
        elif type == 'robust':
            q1 = math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]),
                                                               self.vertices[1].distance_to(self.vertices[3]))
            # segts = self.length_4_segments()
            # q1 = min(segts) / max(segts)
            angles = []
            for i in range(4):
                angles.append(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4],
                                        self.vertices[i - 1]))
            q2 = min(angles) / max(angles)
            # print(f"Edge: {q1}, Angle: {q2}")
            return math.sqrt(q1 * q2) #(q1 + q2) /2
        elif type == 'taper':
            p0, p1, p2, p3 = self.vertices[0], self.vertices[-1], self.vertices[-2], self.vertices[-3]
            x1 = (p1 - p0) + (p2 - p3)
            x2 = (p2 - p1) + (p3 - p0)
            x12 = (p0 - p1) + (p2 - p3)
            return x12.length() / min(x1.length(), x2.length())
        elif type == 's_jacobian':
            p0, p1, p2, p3 = self.vertices[0], self.vertices[-1], self.vertices[-2], self.vertices[-3]
            p01 = (p0 + p1) / 2
            p12 = (p2 + p1) / 2
            p23 = (p2 + p3) / 2
            p30 = (p0 + p3) / 2
            p00 = (p0 + p1 + p2 + p3) / 4
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
            # q1 = math.sqrt(2) * min(self.length_4_segments()) / max(self.vertices[0].distance_to(self.vertices[2]),
            #                                                         self.vertices[1].distance_to(self.vertices[3]))
            angles = []
            for i in range(4):
                angles.append(math.fabs(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4],
                                                                       self.vertices[i - 1])))
            q2 = min(angles) / max(angles)
            # angle_product = 1
            # for i in range(4):
            #     angle_product *= 1 - (
            #                 math.fabs(math.degrees(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4],
            #                                                                                 self.vertices[
            #                                                                                     i - 1])) - 90) / 90)
            # print(angle_product)
            # if angle_product < 0:
            #     q2 = 0
            # else:
            #     q2 = math.pow(angle_product, 1 / 4)
            # q2 = angle_product

            # print(f"Edge: {q1}, Angle: {q2}")
            return math.sqrt(q1 * q2)
        elif type == 'area':
            q1, q2 = self.get_quality_3()
            return q1 * q2

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
            angle_product *= 1 - (math.fabs(math.degrees(self.vertices[i].to_find_clockwise_angle(self.vertices[(i + 1) % 4],
                                                                                 self.vertices[i - 1])) - 90) / 90)
        # print(angle_product)
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
        distance = (origin_p.distance_to(left_p) + \
                   origin_p.distance_to(right_p)) * factor

        if suggest_dist is not None:
            distance = min(distance, 0.6 * suggest_dist)

        s = Segment.get_ray_segment(Segment(origin_p, left_p),
                                    Segment(origin_p, right_p),
                                    distance)
        return s.point2

    def show(self, quality = 3):
        for i in range(len(self.vertices)):
            segt = Segment(self.vertices[i], self.vertices[i - 1])
            segt.show(style='k.-', linewidth=1, markersize=8)
        plt.gca().set_aspect('equal', adjustable='box')
        if quality != 0:
            center = self.get_centriod()
            q1, q2 = self.get_quality_3()
            _quality = round(math.sqrt(q1 * q2), 2)
            plt.text(center.x*0.8, center.y * 0.8, _quality, fontsize=15)
        # plt.xlim(-0.3,1.5)
        # plt.ylim(-0.3, 1.5)
        # plt.show()
        plt.gca().set_frame_on(False)
        plt.xticks([])
        plt.yticks([])

    @staticmethod
    def test():
        v1 = Vertex(1 + math.cos(math.radians(40)), math.sin(math.radians(40)))
        a = 15
        vs = [
            Vertex(0.9 * math.cos(math.radians(a)), 0.9 * math.sin(math.radians(a))),
            Vertex(math.cos(math.radians(a)), math.sin(math.radians(a))),
            Vertex(1.2 * math.cos(math.radians(a)), 1.2 * math.sin(math.radians(a))),
              Vertex(1.45 * math.cos(math.radians(a)), 1.45 * math.sin(math.radians(a))),
              Vertex(1.68 * math.cos(math.radians(a)), 1.68 * math.sin(math.radians(a))),
              Vertex(1.88 * math.cos(math.radians(a)), 1.88 * math.sin(math.radians(a))),
            Vertex(1 + math.cos(math.radians(2 * a)), math.sin(math.radians(2 * a))),
              Vertex(3 * math.cos(math.radians(a)), 3 * math.sin(math.radians(a)))
        ]
        print(v1.distance_to(Vertex(0, 0)))
        for v in vs:
            m = Mesh([Vertex(0, 0), Vertex(math.cos(math.radians(2 * a)), math.sin(math.radians(2 * a))),
                      v, Vertex(1, 0)])
            q1 = m.get_quality(type='area')
            print(f"area: {q1}, default: {m.get_quality()}, robust: {m.get_quality(type='robust')}, "
                  f"strong: {m.get_quality(type='strong')}")

        a = 60
        vs = [
              Vertex(0.9*math.cos(math.radians(a)), 0.9*math.sin(math.radians(a))),
              Vertex(math.cos(math.radians(a)), math.sin(math.radians(a))),
              Vertex(1.2 * math.cos(math.radians(a)), 1.2 * math.sin(math.radians(a))),
              Vertex(1.45 * math.cos(math.radians(a)), 1.45 * math.sin(math.radians(a))),
              Vertex(1.68 * math.cos(math.radians(a)), 1.68 * math.sin(math.radians(a))),
              Vertex(1.88 * math.cos(math.radians(a)), 1.88 * math.sin(math.radians(a))),
              # Vertex(1 + math.cos(math.radians(2 * a)), math.sin(math.radians(2 * a))),
              Vertex(3 * math.cos(math.radians(a)), 3 * math.sin(math.radians(a)))]
        print(v1.distance_to(Vertex(0, 0)))
        for v in vs:
            m = Mesh([Vertex(0, 0), Vertex(math.cos(math.radians(2 * a)), math.sin(math.radians(2 * a))),
                      v, Vertex(1, 0)])
            q1 = m.get_quality(type='area')
            print(f"area: {q1}, default: {m.get_quality()}, robust: {m.get_quality(type='robust')}, "
                  f"strong: {m.get_quality(type='strong')}")

# Mesh.test()

class PointEnvironment(object):

    def __init__(self, reference_point, boundary, neighbor_num=4, radius_num=3,
                 average_edge_length=1, area_ratio=1, radius=6, static=False):
        self.reference_point = reference_point
        self.neighbors = None
        self.radius_neighbors = None
        self.base_length = None
        self.available_radius = None
        self.state = None
        self.boundary = boundary
        self.neighbor_num = neighbor_num
        self.radius_num = radius_num
        self.average_edge_length =average_edge_length
        self.theta = 0
        self.radius = radius
        self.state_vertices = [None for i in range(self.neighbor_num + self.radius_num)]
        # self.get_available_radius()
        self.area_ratio = area_ratio
        self.static = static
        self.get_state(type=2)

    def get_neighbors(self, boundary):
        # half = int(neighbor_num / 2)
        # if neighbor_num % 2 != 0:
        #     raise ValueError("The neighbor number is not even!")
        p_num = len(boundary.vertices)
        index = boundary.vertices.index(self.reference_point)
        vertices = boundary.get_neighbors(self.reference_point, num_points=self.neighbor_num)
        self.neighbors = vertices
        self.base_length = round(sum([vertices[i].distance_to(vertices[i-1])
                                      for i in range(1, len(vertices))]) / self.neighbor_num, 4)


    def get_closest_radius_neighbors(self, boundary, base_point, start_point, end_point, radius):
        def radius_neighbors_with_angle(start_angle, end_angle):
            closet_neighbor = boundary.get_closet_point(
                boundary.get_points_within_angle(boundary.vertices,
                                                      base_point,
                                                      start_point,
                                                      start_angle,
                                                      end_angle
                                                      ),
                base_point, exclusion=self.neighbors,
                S_T=self.base_length)

            if not closet_neighbor:
                _angle = base_point. \
                    to_find_clockwise_angle(start_point,
                                            Vertex(base_point.x + 1, base_point.y))

                return base_point + Vertex(self.base_length * math.cos(_angle - (start_angle + end_angle) / 2),
                                           self.base_length * math.sin((_angle - (start_angle + end_angle) / 2)))
            return closet_neighbor

        angle = base_point.to_find_clockwise_angle(start_point, end_point)
        left_neighbor = radius_neighbors_with_angle(0.01, angle / 3)
        middle_neighbor = radius_neighbors_with_angle(angle / 3, 2 * angle / 3)
        right_neighbor = radius_neighbors_with_angle(2 * angle / 3, angle * 0.99)
        return [left_neighbor, middle_neighbor, right_neighbor]

    def get_closest_radius_neighbors_2(self, boundary, base_point, start_point, end_point):
        angle = base_point.to_find_clockwise_angle(start_point, end_point)
        vertices = boundary.get_points_within_angle(boundary.vertices,
                                                      base_point,
                                                      start_point,
                                                      0.02,
                                                        angle-0.02
                                                      )
        _vs = [v for v in vertices if v not in self.neighbors]
        if len(_vs) == 0:
            _index = boundary.vertices.index(self.reference_point)
            closest_point = boundary.vertices[_index-2]
        else:
            closest_point = Boundary2D.compute_dist(_vs, base_point)[0][0]
        index = boundary.vertices.index(closest_point)
        return [boundary.vertices[(index + 1) % len(boundary.vertices)], closest_point, boundary.vertices[index-1]]

    def get_closet_neighbor(self, boundary, index=0):
        half = int(len(self.neighbors) / 2)
        if index == 0:
            self.radius_neighbors = self.get_closest_radius_neighbors(boundary, self.reference_point, self.neighbors[half + 1],
                                                     self.neighbors[half - 1], self.radius)
        elif index == 1:
            self.radius_neighbors = self.get_closest_radius_neighbors_2(boundary, self.reference_point, self.neighbors[half + 1],
                                                     self.neighbors[half - 1])
        elif index == 2:
            pass
        else:
            pass

            # def get_available_radius(self):
    #     if len(self.radius_neighbors):
    #         all_distance = [self.reference_point.distance_to(v) / 2 for v in self.radius_neighbors]
    #         self.available_radius = round(min(all_distance) / self.base_length, 1)
    #     else:
    #         self.available_radius = self.radius

    def get_state(self, type=0):
        if type == 0:
            self.get_neighbors(self.boundary)
            # self.get_radius_neighbors(boundary)
            self.get_closet_neighbor(self.boundary, index=0)

            self.state = self.points_as_array(self.neighbors + self.radius_neighbors)
        elif type == 1:
            self.get_neighbors(self.boundary)
            angle = self.neighbors[1].to_find_clockwise_angle(self.neighbors[-1],
                                                              self.neighbors[0])
            d_r = self.neighbors[1].distance_to(self.neighbors[-1]) / self.base_length
            d_l = self.neighbors[1].distance_to(self.neighbors[0]) / self.base_length

            self.get_closet_neighbor(self.boundary, index=1)
            r_angle = self.reference_point.to_find_clockwise_angle(self.radius_neighbors[1],
                                                              self.neighbors[0])
            r_d = self.neighbors[1].distance_to(self.radius_neighbors[1]) / self.base_length
            self.state = [angle, d_l, d_r, r_angle, r_d]

        elif type == 2:
            self.get_neighbors(self.boundary)
            self.state = self.get_radius_points().flatten()
        else:
            pass

    def clip_angle(self, angle, max_angle):
        # normalize the angle
        # if 1.5 * math.pi > angle > max_angle + math.pi / 2:
        #     return max_angle + math.pi / 2
        # elif 1.5 * math.pi <= angle:
        #     return angle - 2 * math.pi
        # else:
        return min(angle, max_angle + math.pi / 2)

    def get_radius_points(self):
        r_points = np.full([self.radius_num + self.neighbor_num, 2], 1, dtype=np.float32)
        index = self.boundary.vertices.index(self.reference_point)
        right_p = self.boundary.vertices[index - 1]
        # rr_p = self.boundary.vertices[index - 2]
        left_p = self.boundary.vertices[(index + 1) % len(self.boundary.vertices)]
        # ll_p = self.boundary.vertices[(index + 2) % len(self.boundary.vertices)]
        target_length = self.base_length * self.radius

        theta = self.reference_point.to_find_clockwise_angle(left_p, right_p)
        self.theta = theta

        for i in range(self.neighbor_num // 2):
            if i == 0:
                # r_points[i] = [(self.reference_point.distance_to(right_p) / self.radius) / self.base_length,
                #                self.base_length / self.average_edge_length]
                # the second value is the absolute distance between current base length and the mean length of the boundary
                if not self.static:
                    r_points[i] = [(self.reference_point.distance_to(right_p) / self.radius) / self.base_length,
                                   self.area_ratio] #self.area_ratio
                else:
                    r_points[i] = [(self.reference_point.distance_to(right_p) / self.radius) / self.base_length,
                                   0]  # self.area_ratio
                r_points[self.radius_num + self.neighbor_num - i - 1] = [
                    (self.reference_point.distance_to(left_p) / self.radius) / self.base_length,
                    theta]
                self.state_vertices[i] = right_p
                self.state_vertices[self.radius_num + self.neighbor_num - i - 1] = left_p

            else:
                _angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[index - i - 1], right_p)
                r_points[i] = [(self.reference_point.distance_to(self.boundary.vertices[index - i - 1]) / self.radius) / self.base_length,
                               _angle if _angle < math.pi else max(_angle, 1.5 * math.pi) - 2 * math.pi]
                _angle = self.reference_point.to_find_clockwise_angle(
                    self.boundary.vertices[(index + 1 + i) % len(self.boundary.vertices)], right_p)
                r_points[self.radius_num + self.neighbor_num - i - 1] = [(self.reference_point.distance_to(
                    self.boundary.vertices[(index + i + 1) % len(self.boundary.vertices)]) / self.radius) / self.base_length,
                             min(_angle, theta + math.pi / 2)]
                self.state_vertices[i] = self.boundary.vertices[index - i - 1]
                self.state_vertices[self.radius_num + self.neighbor_num - i - 1] = \
                    self.boundary.vertices[(index + i + 1) % len(self.boundary.vertices)]


        rotation_angle = self.reference_point.to_find_clockwise_angle(right_p, self.reference_point + Vertex(1, 0))
        # initial angle for all the middle vertices
        angles = [i * theta / (2 * self.radius_num) for i in range(1, 2 * self.radius_num, 2)]
        # p_s = []
        for i, a in enumerate(angles):
            r_points[self.neighbor_num // 2 + i][1] = self.clip_angle(a, theta)
        #     p_s.append(self.reference_point + Vertex.rotate(Vertex(target_length * math.cos(a),
        #                                                  target_length * math.sin(a)), rotation_angle))
        p_s = self.reference_point + Vertex.rotate(Vertex(target_length * math.cos(theta / 2),
                                                         target_length * math.sin(theta / 2)), rotation_angle)
        shortest_edge = [1, 0] # dist, id

        for i in range(index - 1, index - len(self.boundary.vertices), -1):
            d = self.reference_point.distance_to(self.boundary.vertices[i])
            if self.boundary.vertices[i] in [right_p, left_p]:
                continue
            else:
                angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i], right_p)
                l_angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i + 1], right_p)

            if angle == 0:
                continue
            k = int(angle / (theta / self.radius_num))
            # h = int(l_angle / (theta / self.radius_num))
            if k < self.radius_num and d < target_length:
                if r_points[k + self.neighbor_num // 2][0] > (d / self.radius) / self.base_length:
                    r_points[k + self.neighbor_num // 2][0] = (d / self.radius) / self.base_length
                    r_points[k + self.neighbor_num // 2][1] = self.clip_angle(angle, theta)
                    self.state_vertices[k + self.neighbor_num // 2] = self.boundary.vertices[i]

            # Check intersected segments
            seg = Segment(self.boundary.vertices[i], self.boundary.vertices[i + 1])
            # for c_k in p_s:
            ll = Segment(self.reference_point, p_s)
            flag, vv = ll.intersection_vertex(seg)
            if flag is not None:
                if flag:
                    _d = self.reference_point.distance_to(vv)
                    if shortest_edge[0] > (_d / self.radius) / self.base_length:
                        shortest_edge[0] = (_d / self.radius) / self.base_length
                        shortest_edge[1] = i

        if shortest_edge[0] != 1 and shortest_edge[0] < r_points[(self.radius_num + self.neighbor_num) // 2][0]:
            _i = shortest_edge[1]

            for i in range(self.radius_num):
                r_points[self.neighbor_num // 2 + i] = [(self.reference_point.distance_to(
                    self.boundary.vertices[i - self.radius_num // 2 + _i]) / self.radius) / self.base_length,
                                 self.reference_point.to_find_clockwise_angle(
                                     self.boundary.vertices[i - self.radius_num // 2 + _i],
                                     right_p)]
                self.state_vertices[self.neighbor_num // 2 + i] = \
                    self.boundary.vertices[i - self.radius_num // 2 + _i]

        self.state_vertices.insert(0, self.reference_point)
        return np.asarray([[round(v[0], 4), round(v[1], 4)] for v in r_points])

    def get_radius_points_old(self):
        r_points = np.full([self.radius_num + self.neighbor_num, 2], 1, dtype=np.float32)
        index = self.boundary.vertices.index(self.reference_point)
        right_p = self.boundary.vertices[index - 1]
        left_p = self.boundary.vertices[(index + 1) % len(self.boundary.vertices)]
        target_length = self.base_length * self.radius

        theta = self.reference_point.to_find_clockwise_angle(left_p, right_p)
        self.theta = theta

        for i in range(self.neighbor_num // 2):
            if i == 0:
                # the second value is the absolute distance between current base length and the mean length of the boundary
                r_points[i] = [(self.reference_point.distance_to(right_p) / self.radius) / self.base_length,
                               self.area_ratio]  # self.area_ratio
                r_points[self.radius_num + self.neighbor_num - i - 1] = [
                    (self.reference_point.distance_to(left_p) / self.radius) / self.base_length,
                    theta]
            else:
                _angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[index - i - 1], right_p)
                r_points[i] = [(self.reference_point.distance_to(
                    self.boundary.vertices[index - i - 1]) / self.radius) / self.base_length,
                               _angle if _angle < math.pi else max(_angle, 1.5 * math.pi) - 2 * math.pi]
                _angle = self.reference_point.to_find_clockwise_angle(
                    self.boundary.vertices[(index + 1 + i) % len(self.boundary.vertices)], right_p)
                r_points[self.radius_num + self.neighbor_num - i - 1] = [(self.reference_point.distance_to(
                    self.boundary.vertices[
                        (index + i + 1) % len(self.boundary.vertices)]) / self.radius) / self.base_length,
                                                                         min(_angle, theta + math.pi / 2)]

        # rr_angle = self.reference_point.to_find_clockwise_angle(rr_p, right_p)
        # r_points[1] = [(self.reference_point.distance_to(rr_p) / self.radius) / self.base_length,
        #                rr_angle]
        # r_points[self.radius_num + self.neighbor_num - 2] = [(self.reference_point.distance_to(ll_p) / self.radius) / self.base_length,
        #                self.reference_point.to_find_clockwise_angle(ll_p, right_p)]

        rotation_angle = self.reference_point.to_find_clockwise_angle(right_p, self.reference_point + Vertex(1, 0))
        # initial angle for all the middle vertices
        angles = [i * theta / (2 * self.radius_num) for i in range(1, 2 * self.radius_num, 2)]
        p_s = []
        for i, a in enumerate(angles):
            r_points[self.neighbor_num // 2 + i][1] = self.clip_angle(a, theta)
            p_s.append(self.reference_point + Vertex.rotate(Vertex(target_length * math.cos(a),
                                                                   target_length * math.sin(a)), rotation_angle))

        # r_points[2][1], r_points[3][1], r_points[4][1] = [self.clip_angle(a, theta) for a in angles]

        shortest_edge = [1, 0]  # dist, id

        for i in range(index - 1, index - len(self.boundary.vertices), -1):
            d = self.reference_point.distance_to(self.boundary.vertices[i])
            if self.boundary.vertices[i] in [right_p, left_p]:
                continue
            else:
                angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i], right_p)
                l_angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i + 1], right_p)

            if angle == 0:
                continue
            k = int(angle / (theta / self.radius_num))
            h = int(l_angle / (theta / self.radius_num))
            if k < self.radius_num and d < target_length:
                if r_points[k + self.neighbor_num // 2][0] > (d / self.radius) / self.base_length:
                    r_points[k + self.neighbor_num // 2][0] = (d / self.radius) / self.base_length
                    r_points[k + self.neighbor_num // 2][1] = self.clip_angle(angle, theta)

            # if d < target_length and 0 < angle < theta:
            #     if r_points[3][0] > (d / self.radius) / self.base_length:
            #         r_points[3][0] = (d / self.radius) / self.base_length
            #         r_points[3][1] = self.clip_angle(angle, theta)
            #         r_points[2][0] = (self.reference_point.distance_to(self.boundary.vertices[i + 1])
            #                          / self.radius) / self.base_length
            #         r_points[2][1] = self.clip_angle(self.reference_point.to_find_clockwise_angle(
            #             self.boundary.vertices[i + 1], right_p), theta)
            #         r_points[4][0] = (self.reference_point.distance_to(self.boundary.vertices[i - 1])
            #                          / self.radius) / self.base_length
            #         r_points[4][1] = self.clip_angle(self.reference_point.to_find_clockwise_angle(
            #             self.boundary.vertices[i - 1], right_p), theta)
            else:
                seg = Segment(self.boundary.vertices[i], self.boundary.vertices[i + 1])
                for c_k in p_s:
                    ll = Segment(self.reference_point, c_k)
                    flag, vv = ll.intersection_vertex(seg)
                    if flag is not None:
                        if flag:
                            _d = self.reference_point.distance_to(vv)
                            if shortest_edge[0] > (_d / self.radius) / self.base_length:
                                shortest_edge[0] = (_d / self.radius) / self.base_length
                                shortest_edge[1] = i
        change = True
        for i in range(self.radius_num):
            if r_points[self.neighbor_num // 2 + i][0] != 1:
                change = False
        if change and shortest_edge[0] != 1:
            _i = shortest_edge[1]

            for i in range(self.radius_num):
                r_points[self.neighbor_num // 2 + i] = [(self.reference_point.distance_to(
                    self.boundary.vertices[i - self.radius_num // 2 + _i]) / self.radius) / self.base_length,
                                                        self.reference_point.to_find_clockwise_angle(
                                                            self.boundary.vertices[i - self.radius_num // 2 + _i],
                                                            right_p)]

        # if d > target_length:
        #     continue
        # if right_p is self.boundary.vertices[i] or left_p is self.boundary.vertices[i]:
        #     continue
        # else:
        #     angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i], right_p)
        #     l_angle = self.reference_point.to_find_clockwise_angle(self.boundary.vertices[i + 1], right_p)
        #
        # if angle == 0:
        #     continue
        # k = int(angle / (theta / 3))
        # h = int(l_angle / (theta / 3))
        # if k < 3:
        #     if r_points[k + 2][0] > (d / self.radius) / self.base_length or r_points[k + 2][0] == 1:
        #         r_points[k + 2][0] = (d / self.radius) / self.base_length
        #         r_points[k + 2][1] = self.clip_angle(angle)
        #
        # if angle < math.pi:
        #     if k != h:
        #         cross_k = 2 if k >= 3 else k
        #         seg = Segment(self.boundary.vertices[i], self.boundary.vertices[i + 1])
        #         for c_k in range(cross_k + 1):
        #             for j in [0, 1]:
        #                 ll = Segment(self.reference_point, p_s[c_k + j])
        #                 flag, vv = ll.intersection_vertex(seg)
        #                 if flag is not None:
        #                     if flag:
        #                         _d = self.reference_point.distance_to(vv)
        #                         if r_points[c_k + 2][0] > (_d / self.radius) / \
        #                                 self.base_length or r_points[c_k + 2][0] == 1:
        #                             r_points[c_k + 2][0] = (_d / self.radius) / self.base_length
        #                             r_points[c_k + 2][1] = self.clip_angle(angles[c_k + j])

        return np.asarray([[round(v[0], 4), round(v[1], 4)] for v in r_points])

    def points_as_array(self, points):
        flated_points = []
        for point in points:
            flated_points.append(point.x)
            flated_points.append(point.y)
        return np.asarray(flated_points)

    def segmts_diff_ratio(self):
        res = list(sorted([self.neighbors[i - 1].distance_to(self.neighbors[i]) for i in range(1, len(self.neighbors))]))
        r1 = min((res[0], res[1])) / max((res[0], res[1]))
        r2 = min((res[-2], res[-1])) / max((res[-2], res[-1]))
        r3 = res[0] / res[1]
        min_r = min((r1, r2, r3))
        max_r = max((r1, r2, r3))
        return min_r, max_r

    # def get_aspect_ratio(self):
    #     if self.neighbors:
    #         dists = [self.neighbors[i - 1].distance_to(self.neighbors[i]) for i in range(1, len(self.neighbors))]
    #         return min(dists) / max(dists)


def solve_quadratic_equation(a, b, c):
    drt = b * b - 4 * a * c
    if a == 0:
        if b != 0:
            return -c / b
    else:
        if drt == 0:
            return -b / 2 / a, -b / 2 / a
        else:
            if drt > 0:
                x1 = -b + math.sqrt(drt) / 2 / a;
                x2 = -b - math.sqrt(drt) / 2 / a;
                return x1, x2