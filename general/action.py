import math

import numpy as np

from general.config import NEIGHBOR_NUM, RADIUS
from general.geometry import Vertex, Quad


def action_point(ref_state, point):
    state = Vertex.flatten(ref_state['neighbors'])
    v1 = np.asarray([state[NEIGHBOR_NUM], state[NEIGHBOR_NUM + 1]], dtype=float)
    v2 = np.asarray([state[NEIGHBOR_NUM + 2], state[NEIGHBOR_NUM + 3]], dtype=float)
    decoded = from_local_frame(point, ref_state['base_length'], v1, v2)
    return Vertex(round(decoded[0], 4), round(decoded[1], 4))


def polar_action_point(ref_state, polar):
    state = Vertex.flatten(ref_state['neighbors'])
    v1 = np.asarray([state[NEIGHBOR_NUM], state[NEIGHBOR_NUM + 1]], dtype=float)
    v2 = np.asarray([state[NEIGHBOR_NUM + 2], state[NEIGHBOR_NUM + 3]], dtype=float)
    x = ref_state['base_length'] * RADIUS * polar[0] * math.cos(polar[1])
    y = ref_state['base_length'] * RADIUS * polar[0] * math.sin(polar[1])
    decoded = from_local_frame([round(x, 6), round(y, 6)], 1, v1, v2)
    return Vertex(round(decoded[0], 4), round(decoded[1], 4))


def rule_quad(boundary, rule, index, new_point=None):
    v = boundary.vertices
    n = len(v)
    if rule == -1:
        return Quad([v[index - 1], v[index], v[(index + 1) % n], v[(index + 2) % n]])
    if rule == 1:
        return Quad([v[index - 2], v[index - 1], v[index], v[(index + 1) % n]])
    return Quad([new_point, v[index - 1], v[index], v[(index + 1) % n]])


def from_local_frame(point, dist, p0, p1):
    d = p1 - p0
    theta = 2 * math.pi - math.atan2(d[1], d[0])
    return np.array([(np.cos(theta) * point[0] + np.sin(theta) * point[1]) * dist + p0[0],
                     (-np.sin(theta) * point[0] + np.cos(theta) * point[1]) * dist + p0[1]])
