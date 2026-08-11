import math

import numpy as np


def clip_angle(angle, max_angle):
    return min(angle, max_angle + math.pi / 2)


def _circle_line_x(lin, M, N, a, b, r):
    disc = math.sqrt(math.fabs(lin ** 2 - 4 * (M ** 2 + 1) * ((N - b) ** 2 + a ** 2 - r ** 2)))
    denom = 2 * (M ** 2 + 1)
    return (lin + disc) / denom, (lin - disc) / denom


def transformation(arra, dist, p0, p1):
    matrix = np.asarray(arra, dtype=float).reshape(-1, 2) - p0
    matrix = np.divide(matrix, dist)

    d = p1 - p0
    theta = math.atan2(d[1], d[0])

    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [- np.sin(theta), np.cos(theta)]
    ])
    matrix = np.matmul(rotation_matrix, matrix.T).T

    return np.asarray(matrix).reshape(-1)


def detransformation(point, dist, p0, p1):
    d = p1 - p0
    theta = 2 * math.pi - math.atan2(d[1], d[0])
    original_point = np.empty(2)

    original_point[0] = np.cos(theta) * point[0] + np.sin(theta) * point[1]
    original_point[1] = -np.sin(theta) * point[0] + np.cos(theta) * point[1]

    original_point *= dist

    original_point[0] += p0[0]
    original_point[1] += p0[1]

    return original_point
