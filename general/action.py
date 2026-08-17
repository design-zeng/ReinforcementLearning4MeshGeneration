import math

import numpy as np

from general.config import NEIGHBOR_NUM, ALPHA
from general.geometry import Vertex, Quad, Mesh
from general.boundary import Boundary
from general.quality import can_add_quad


def act(view, action, boundary, mesh, outcome):
    state = Vertex.flatten(view['neighbors'])
    v1 = np.asarray([state[NEIGHBOR_NUM], state[NEIGHBOR_NUM + 1]], dtype=float)
    v2 = np.asarray([state[NEIGHBOR_NUM + 2], state[NEIGHBOR_NUM + 3]], dtype=float)
    decoded = from_local_frame([round(action[1], 4), round(action[2], 4)], view['base_length'], v1, v2)
    new_point = Vertex(round(decoded[0], 4), round(decoded[1], 4))

    quad = Quad()
    new_vertex = None
    if action[0] <= -0.5:
        rule = -1
        rule_quad(boundary, -1, view['reference_point'], quad)
    elif action[0] >= 0.5:
        rule = 1
        rule_quad(boundary, 1, view['reference_point'], quad)
    else:
        rule = 0
        if boundary.contains_point(new_point):
            if boundary.find_same_point(new_point):
                rule_quad(boundary, -1, view['reference_point'], quad)
            else:
                rule_quad(boundary, 0, view['reference_point'], quad, new_point)
                new_vertex = new_point

    outcome['rule'] = rule
    outcome['quad'] = quad
    outcome['new_vertex'] = new_vertex
    outcome['absorbed'] = bool(quad.vertices) and can_add_quad(boundary, quad, view['reference_point'])
    if outcome['absorbed']:
        Mesh.add_quad(quad, mesh)
        Boundary.absorb(quad, boundary)


def polar_act(view, action, rule_type, boundary, mesh, outcome):
    state = Vertex.flatten(view['neighbors'])
    v1 = np.asarray([state[NEIGHBOR_NUM], state[NEIGHBOR_NUM + 1]], dtype=float)
    v2 = np.asarray([state[NEIGHBOR_NUM + 2], state[NEIGHBOR_NUM + 3]], dtype=float)
    x = view['base_length'] * ALPHA * action[0] * math.cos(action[1])
    y = view['base_length'] * ALPHA * action[0] * math.sin(action[1])
    decoded = from_local_frame([round(x, 6), round(y, 6)], 1, v1, v2)
    new_point = Vertex(round(decoded[0], 4), round(decoded[1], 4))

    quad = Quad()
    if rule_type <= 0.3:
        rule_quad(boundary, -1, view['reference_point'], quad)
    elif rule_type >= 0.7:
        rule_quad(boundary, 1, view['reference_point'], quad)
    elif boundary.contains_point(new_point):
        rule_quad(boundary, 0, view['reference_point'], quad, new_point)

    outcome['absorbed'] = bool(quad.vertices) and can_add_quad(boundary, quad, view['reference_point'])
    if outcome['absorbed']:
        Mesh.add_quad(quad, mesh)
        Boundary.absorb(quad, boundary)


def rule_quad(boundary, rule, reference_point, quad, new_point=None):
    v = boundary.vertices
    n = len(v)
    index = v.index(reference_point)
    if rule == -1:
        quad.vertices[:] = [v[index - 1], v[index], v[(index + 1) % n], v[(index + 2) % n]]
    elif rule == 1:
        quad.vertices[:] = [v[index - 2], v[index - 1], v[index], v[(index + 1) % n]]
    else:
        quad.vertices[:] = [new_point, v[index - 1], v[index], v[(index + 1) % n]]


def from_local_frame(point, d, p0, p_r1):
    reference_direction = p_r1 - p0
    theta = 2 * math.pi - math.atan2(reference_direction[1], reference_direction[0])
    return np.array([(np.cos(theta) * point[0] + np.sin(theta) * point[1]) * d + p0[0],
                     (-np.sin(theta) * point[0] + np.cos(theta) * point[1]) * d + p0[1]])


def act_last(boundary, mesh):
    Mesh.add_quad(Quad(boundary.vertices), mesh)
