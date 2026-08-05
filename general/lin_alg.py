import math

import numpy as np


def transformation(arra, dist, p0, p1):
    matrix = np.asarray(arra, dtype=float).reshape(-1, 2) - p0
    matrix = np.divide(matrix, dist)

    theta = math.atan2((p1-p0)[1], (p1-p0)[0])

    rotation_matrix = np.asmatrix([
        [np.cos(theta), np.sin(theta)],
        [- np.sin(theta), np.cos(theta)]
    ])
    matrix = np.matmul(rotation_matrix, matrix.T).T

    return np.asarray(matrix).reshape(-1)


def detransformation(point, dist, p0, p1):
    theta = 2 * math.pi - math.atan2((p1 - p0)[1], (p1 - p0)[0])
    original_point = np.empty(2)

    original_point[0] = np.cos(theta) * point[0] + np.sin(theta) * point[1]
    original_point[1] = -np.sin(theta) * point[0] + np.cos(theta) * point[1]

    original_point *= dist

    original_point[0] += p0[0]
    original_point[1] += p0[1]

    return original_point
