import math

import numpy as np

from general.geometry import Vertex


def action_point(ref_state, point, neighbor_num):
    state = Vertex.flatten(ref_state['neighbors'])
    v1 = np.asarray([state[neighbor_num], state[neighbor_num + 1]], dtype=float)
    v2 = np.asarray([state[neighbor_num + 2], state[neighbor_num + 3]], dtype=float)
    decoded = from_local_frame(point, ref_state['base_length'], v1, v2)
    return Vertex(round(decoded[0], 4), round(decoded[1], 4))


def from_local_frame(point, dist, p0, p1):
    d = p1 - p0
    theta = 2 * math.pi - math.atan2(d[1], d[0])
    return np.array([(np.cos(theta) * point[0] + np.sin(theta) * point[1]) * dist + p0[0],
                     (-np.sin(theta) * point[0] + np.cos(theta) * point[1]) * dist + p0[1]])
