import math

import numpy as np

from general.config import NEIGHBOR_NUM, FAN_NUM, BETA
from general.geometry import Vertex, Segment
from general.boundary import Boundary


def recognize(boundary, candidates, view, area_ratio, static, not_valid_points=None):
    candidates[:] = [entry for entry in candidates if entry[0] not in boundary.retired]
    for vertex in boundary.rescored:
        score = Boundary.weighted_reference_angle(boundary, vertex)
        if score is None:
            continue
        i = next((i for i, (_, other) in enumerate(candidates) if score <= other),
                 len(candidates))
        candidates.insert(i, (vertex, score))
    boundary.retired = []
    boundary.rescored = []

    view['reference_point'] = None
    for vertex, _ in candidates:
        if not not_valid_points or not vertex.coincides_with_any(not_valid_points):
            view['reference_point'] = vertex
            break
    if view['reference_point'] is None:
        return
    view['neighbors'] = boundary.get_neighbors(view['reference_point'], num_points=NEIGHBOR_NUM)
    view['base_length'] = round(sum(view['neighbors'][i].distance_to(view['neighbors'][i - 1])
                                    for i in range(1, len(view['neighbors']))) / NEIGHBOR_NUM, 4)
    view['state'] = _partial_observation(boundary, view['reference_point'], area_ratio, static, view['base_length']).flatten()

def _partial_observation(boundary, v0, area_ratio, static, base_length):
    verts = boundary.vertices
    n = len(verts)
    half = NEIGHBOR_NUM // 2
    v0_index = verts.index(v0)
    v_r1 = verts[v0_index - 1]
    v_l1 = verts[(v0_index + 1) % n]
    fan_angle = v0.clockwise_angle(v_l1, v_r1)
    fan_radius = base_length * BETA

    state_points = np.full([FAN_NUM + NEIGHBOR_NUM, 2], 1, dtype=np.float32)

    def fan_slot(i):
        return half + i

    def left_slot(i):
        return FAN_NUM + NEIGHBOR_NUM - i - 1

    def norm_dist(p):
        return (v0.distance_to(p) / BETA) / base_length

    def angle_from_right(p):
        return v0.clockwise_angle(p, v_r1)

    for i in range(half):
        if i == 0:
            state_points[i] = [norm_dist(v_r1), area_ratio if not static else 0]
            state_points[left_slot(0)] = [norm_dist(v_l1), fan_angle]
        else:
            right_neighbor = verts[v0_index - i - 1]
            a = angle_from_right(right_neighbor)
            state_points[i] = [norm_dist(right_neighbor),
                           a if a < math.pi else max(a, 1.5 * math.pi) - 2 * math.pi]
            left_neighbor = verts[(v0_index + 1 + i) % n]
            a = angle_from_right(left_neighbor)
            state_points[left_slot(i)] = [norm_dist(left_neighbor), min(a, fan_angle + math.pi / 2)]

    for i in range(FAN_NUM):
        ray_angle = (2 * i + 1) * fan_angle / (2 * FAN_NUM)
        state_points[fan_slot(i)][1] = _clip_angle(ray_angle, fan_angle)

    rotation_angle = v0.clockwise_angle(v_r1, v0 + Vertex(1, 0))
    bisector_tip = v0 + Vertex.rotate_counterclockwise(
        Vertex(fan_radius * math.cos(fan_angle / 2), fan_radius * math.sin(fan_angle / 2)), rotation_angle)
    nearest_crossing = [1, 0]

    def pull_fan_ray_to(v, a):
        sector = int(a / (fan_angle / FAN_NUM))
        if sector < FAN_NUM and v0.distance_to(v) < fan_radius:
            d = norm_dist(v)
            if state_points[fan_slot(sector)][0] > d:
                state_points[fan_slot(sector)] = [d, _clip_angle(a, fan_angle)]

    def track_bisector_crossing(i):
        hit, point = Segment(v0, bisector_tip).intersection_vertex(Segment(verts[i], verts[i + 1]))
        if hit:
            d = norm_dist(point)
            if nearest_crossing[0] > d:
                nearest_crossing[0] = d
                nearest_crossing[1] = i

    for i in range(v0_index - 1, v0_index - n, -1):
        v = verts[i]
        if v in [v_r1, v_l1]:
            continue
        a = angle_from_right(v)
        if a == 0:
            continue
        pull_fan_ray_to(v, a)
        track_bisector_crossing(i)

    if nearest_crossing[0] != 1 and nearest_crossing[0] < state_points[(FAN_NUM + NEIGHBOR_NUM) // 2][0]:
        edge_index = nearest_crossing[1]
        for i in range(FAN_NUM):
            v = verts[i - FAN_NUM // 2 + edge_index]
            state_points[fan_slot(i)] = [norm_dist(v), angle_from_right(v)]

    return np.asarray([[round(v[0], 4), round(v[1], 4)] for v in state_points])


def _clip_angle(angle, max_angle):
    return min(angle, max_angle + math.pi / 2)
