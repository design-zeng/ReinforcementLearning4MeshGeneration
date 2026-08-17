import math

import numpy as np

from general.config import M_ANGLE
from general.geometry import Segment


def reward(outcome, boundary, mesh, area_bounds):
    if not outcome['absorbed']:
        n = len(mesh.generated_quads)
        return -1 / n if n else -1

    outcome['measurement'] = measurement(boundary, outcome['quad'], outcome['new_vertex'], area_bounds)
    outcome['last'] = len(boundary.vertices) <= 5
    return outcome['measurement'] + 10 if outcome['last'] else outcome['measurement']


def measurement(boundary, quad, new_vertex, area_bounds):
    element = QuadQuality.element(quad)
    remaining_boundary = BoundaryQuality.one_new_vertex(boundary, new_vertex) if new_vertex is not None \
        else BoundaryQuality.no_new_vertex(boundary, quad)
    return element + (remaining_boundary - 1) + density(quad.area(), area_bounds)


def density(area, area_bounds):
    area_min, area_max = area_bounds
    if area_min <= area < area_max:
        return (area - area_max) / (area_max - area_min)
    if area < area_min:
        return -1
    return 0


def can_add_quad(boundary, quad, reference_point):
    return quad.is_valid() and \
        not _intersects_boundary(boundary, quad, reference_point)


def _intersects_boundary(boundary, quad, reference_point):
    max_dist = max(reference_point.distance_to(v) for v in quad.vertices if v is not reference_point)
    neighbouring = [v for v in boundary.vertices
                    if reference_point.distance_to(v) < max_dist and v not in quad.vertices]

    idx = quad.vertices.index(reference_point)
    checking_segs = [Segment(quad.vertices[idx - 1], quad.vertices[idx - 2]),
                     Segment(quad.vertices[idx - 2], quad.vertices[idx - 3])]
    n = len(boundary.vertices)
    for v in neighbouring:
        index = boundary.vertices.index(v)
        for u in (boundary.vertices[index - 1], boundary.vertices[(index + 1) % n]):
            if u not in quad.vertices and \
                    any(seg.is_intersecting(Segment(v, u)) for seg in checking_segs):
                return True
    return False


class QuadQuality:
    @staticmethod
    def element(quad):
        return math.sqrt(QuadQuality.stretch(quad) * QuadQuality.angle_ratio(quad))

    @staticmethod
    def stretch(quad):
        longest_diagonal = max(quad.vertices[0].distance_to(quad.vertices[2]),
                               quad.vertices[1].distance_to(quad.vertices[3]))
        return math.sqrt(2) * min(quad.edge_lengths()) / longest_diagonal

    @staticmethod
    def angle_ratio(quad):
        angles = quad.corner_angles()
        return min(angles) / max(angles)

    @staticmethod
    def edge_angle(quad):
        edge_quality, angle_quality = QuadQuality._edge_and_angle(quad)
        return math.sqrt(edge_quality * angle_quality)

    @staticmethod
    def edge_angle_ratio(quad):
        edge_quality, _ = QuadQuality._edge_and_angle(quad)
        angles = [math.fabs(a) for a in quad.corner_angles()]
        return math.sqrt(edge_quality * (min(angles) / max(angles)))

    @staticmethod
    def taper(quad):
        p0, p1, p2, p3 = quad.vertices[0], quad.vertices[-1], quad.vertices[-2], quad.vertices[-3]
        x1 = (p1 - p0) + (p2 - p3)
        x2 = (p2 - p1) + (p3 - p0)
        x12 = (p0 - p1) + (p2 - p3)
        return x12.length() / min(x1.length(), x2.length())

    @staticmethod
    def scaled_jacobian(quad):
        p0, p1, p2, p3 = quad.vertices[0], quad.vertices[-1], quad.vertices[-2], quad.vertices[-3]
        l0, l1, l2, l3 = p1-p0, p2-p1, p3-p2, p0-p3
        return min([l3.cross(l0) / (l0.length() * l3.length()),
                    l0.cross(l1) / (l0.length() * l1.length()),
                    l1.cross(l2) / (l1.length() * l2.length()),
                    l2.cross(l3) / (l2.length() * l3.length())])

    @staticmethod
    def transition(mesh, quad):
        near = QuadQuality._near_quads(mesh, quad)
        if not near:
            return 1
        s_i = quad.area()
        ratios = []
        for m in near:
            s_j = m.area()
            ratios.append((s_i / s_j) ** ((s_j - s_i) / math.fabs(s_i - s_j)) if s_j != s_i else 1)
        return float(np.average(np.asarray(ratios)))

    @staticmethod
    def _edge_and_angle(quad):
        area = quad.area()
        if area <= 0:
            edge_quality = 0
        else:
            side = math.sqrt(area)
            edge_quality = math.pow(math.prod(math.pow(edge / side, 1 if side - edge > 0 else -1)
                                              for edge in quad.edge_lengths()), 1 / 4)

        angle_product = math.prod(1 - (math.fabs(math.degrees(a) - 90) / 90)
                                  for a in quad.corner_angles())
        angle_quality = 0 if angle_product < 0 else math.pow(angle_product, 1/4)
        return edge_quality, angle_quality

    @staticmethod
    def _near_quads(mesh, quad):
        near = []
        for v in quad.vertices:
            for m in mesh.generated_quads:
                if m is not quad and v in m.vertices:
                    near.append(m)
        return set(near)


class BoundaryQuality:
    @staticmethod
    def no_new_vertex(boundary, quad):
        v = boundary.vertices
        n = len(v)
        front_vertices = [vertex for vertex in quad.vertices if vertex in v]
        if not front_vertices:
            return 1

        generated_angles = []
        for vertex in front_vertices:
            index = v.index(vertex)
            angle = vertex.clockwise_angle(v[(index + 1) % n], v[index - 1])
            if angle < M_ANGLE:
                generated_angles.append(angle)
        index = min(v.index(front_vertices[0]), v.index(front_vertices[1]))

        target_len = front_vertices[0].distance_to(front_vertices[1])
        ring = [(index + i) % n for i in range(-2, 4)]
        mean_dist = sum(v[ring[i]].distance_to(v[ring[i + 1]]) for i in range(len(ring) - 1)) / (len(ring) - 1)

        smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)
        angle_quality = 3 * min(generated_angles) / math.pi if generated_angles else 1
        return math.pow(angle_quality * smoothness, 1 / 2)

    @staticmethod
    def one_new_vertex(boundary, new_vertex):
        v = boundary.vertices
        n = len(v)
        index = v.index(new_vertex)

        generated_angles = [a for i in (1, -1)
                            if (a := v[(index + i) % n].clockwise_angle(v[(index + i + 1) % n], v[index + i - 1])) < M_ANGLE]
        angle_quality = 3 * min(generated_angles) / math.pi if generated_angles else 1

        dist = new_vertex.distance_to(v[(index + 1) % n]) + new_vertex.distance_to(v[index - 1])
        excluded = {v[index], v[(index + 1) % n], v[(index + 2) % n], v[index - 1], v[index - 2]}
        close_vs = []
        for i, vv in enumerate(v):
            if vv not in excluded and new_vertex.distance_to(vv) < dist and i - 1 not in close_vs:
                close_vs.append(i)
        dists = [Segment(v[(i + 1) % n], v[i]).distance(new_vertex) for i in close_vs]

        target_len = dist / 2
        ring = [(index + i) % n for i in range(-2, 3)]
        mean_dist = sum(v[ring[i]].distance_to(v[ring[i + 1]]) for i in range(len(ring) - 1)) / (len(ring) - 1)
        smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)

        dist_quality = min(dists) / (0.5 * dist) if dists and min(dists) < 0.5 * dist else 1
        return math.pow(smoothness * angle_quality * dist_quality, 1 / 3)
