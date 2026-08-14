import math

import numpy as np

from general.geometry import Vertex, Segment


def reference_state(boundary, reference_point, neighbor_num, radius_num, area_ratio, radius, static):
    neighbors = boundary.get_neighbors(reference_point, num_points=neighbor_num)
    base_length = round(sum(neighbors[i].distance_to(neighbors[i - 1])
                            for i in range(1, len(neighbors))) / neighbor_num, 4)
    r_points = _radius_points(boundary, reference_point, neighbor_num, radius_num,
                             area_ratio, radius, static, base_length)
    return {'reference_point': reference_point, 'neighbors': neighbors,
            'base_length': base_length, 'state': r_points.flatten()}

def _radius_points(boundary, rp, neighbor_num, radius_num, area_ratio, radius, static, base_length):
    verts = boundary.vertices
    n = len(verts)
    half = neighbor_num // 2
    rp_index = verts.index(rp)
    right_p = verts[rp_index - 1]
    left_p = verts[(rp_index + 1) % n]
    front_angle = rp.clockwise_angle(left_p, right_p)
    target_length = base_length * radius

    r_points = np.full([radius_num + neighbor_num, 2], 1, dtype=np.float32)

    def fan_slot(i):
        return half + i

    def left_slot(i):
        return radius_num + neighbor_num - i - 1

    def norm_dist(p):
        return (rp.distance_to(p) / radius) / base_length

    def angle_from_right(p):
        return rp.clockwise_angle(p, right_p)

    for i in range(half):
        if i == 0:
            r_points[i] = [norm_dist(right_p), area_ratio if not static else 0]
            r_points[left_slot(0)] = [norm_dist(left_p), front_angle]
        else:
            right_neighbor = verts[rp_index - i - 1]
            a = angle_from_right(right_neighbor)
            r_points[i] = [norm_dist(right_neighbor),
                           a if a < math.pi else max(a, 1.5 * math.pi) - 2 * math.pi]
            left_neighbor = verts[(rp_index + 1 + i) % n]
            a = angle_from_right(left_neighbor)
            r_points[left_slot(i)] = [norm_dist(left_neighbor), min(a, front_angle + math.pi / 2)]

    for i in range(radius_num):
        ray_angle = (2 * i + 1) * front_angle / (2 * radius_num)
        r_points[fan_slot(i)][1] = _clip_angle(ray_angle, front_angle)

    rotation_angle = rp.clockwise_angle(right_p, rp + Vertex(1, 0))
    bisector_tip = rp + Vertex.rotate_counterclockwise(
        Vertex(target_length * math.cos(front_angle / 2), target_length * math.sin(front_angle / 2)), rotation_angle)
    nearest_crossing = [1, 0]

    def pull_fan_ray_to(v, a):
        sector = int(a / (front_angle / radius_num))
        if sector < radius_num and rp.distance_to(v) < target_length:
            d = norm_dist(v)
            if r_points[fan_slot(sector)][0] > d:
                r_points[fan_slot(sector)] = [d, _clip_angle(a, front_angle)]

    def track_bisector_crossing(i):
        hit, point = Segment(rp, bisector_tip).intersection_vertex(Segment(verts[i], verts[i + 1]))
        if hit:
            d = norm_dist(point)
            if nearest_crossing[0] > d:
                nearest_crossing[0] = d
                nearest_crossing[1] = i

    for i in range(rp_index - 1, rp_index - n, -1):
        v = verts[i]
        if v in [right_p, left_p]:
            continue
        a = angle_from_right(v)
        if a == 0:
            continue
        pull_fan_ray_to(v, a)
        track_bisector_crossing(i)

    if nearest_crossing[0] != 1 and nearest_crossing[0] < r_points[(radius_num + neighbor_num) // 2][0]:
        edge_index = nearest_crossing[1]
        for i in range(radius_num):
            v = verts[i - radius_num // 2 + edge_index]
            r_points[fan_slot(i)] = [norm_dist(v), angle_from_right(v)]

    return np.asarray([[round(v[0], 4), round(v[1], 4)] for v in r_points])


def _clip_angle(angle, max_angle):
    return min(angle, max_angle + math.pi / 2)


def golden_ratio_weighted_average_angle(boundary, vertex, index=None):
    if index is None:
        index = boundary.vertices.index(vertex)
    n = len(boundary.vertices)
    half = boundary.num_ref_neighbor // 2
    golden = 0.618
    weights = [golden, 1 - golden] if half == 2 else [2 / boundary.num_ref_neighbor] * half

    weighted_angle = 0
    for i in range(half):
        angle = vertex.clockwise_angle(
            boundary.vertices[(index + 1 + i) % n], boundary.vertices[index - 1 - i])
        if i == 0 and (angle >= boundary.maximum_reference_angle or angle == 0):
            return None
        weighted_angle += angle * weights[i]
    return math.degrees(weighted_angle)


def reference_candidates(boundary, target_angle=0):
    candidates = [(vertex, score) for i, vertex in enumerate(boundary.vertices)
                  if (score := golden_ratio_weighted_average_angle(boundary, vertex, i)) is not None]
    return sorted(candidates, key=lambda x: math.fabs(x[1] - target_angle))


def next_reference_point(candidates, not_valid_points=None):
    if not candidates:
        return None
    if not not_valid_points:
        return candidates[0][0]
    for vertex, _ in candidates:
        if not vertex.coincides_with_any(not_valid_points):
            return vertex
    return None


def updated_candidates(candidates, boundary, retired, rescored):
    candidates = [entry for entry in candidates if entry[0] not in retired]
    for vertex in rescored:
        score = golden_ratio_weighted_average_angle(boundary, vertex)
        if score is None:
            continue
        i = next((i for i, (_, other) in enumerate(candidates) if score <= other),
                 len(candidates))
        candidates.insert(i, (vertex, score))
    return candidates
