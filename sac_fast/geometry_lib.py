import math

import numpy as np
import numpy.typing as npt


def segment_intersect(a1: npt.NDArray[np.floating], a2: npt.NDArray[np.floating], b1: npt.NDArray[np.floating], b2: npt.NDArray[np.floating]) -> bool:
    ux, uy = b1[0] - b2[0], b1[1] - b2[1]
    c1 = ux * (a1[1] - b2[1]) - uy * (a1[0] - b2[0])
    c2 = ux * (a2[1] - b2[1]) - uy * (a2[0] - b2[0])

    u2x, u2y = a1[0] - a2[0], a1[1] - a2[1]
    c3 = u2x * (b1[1] - a2[1]) - u2y * (b1[0] - a2[0])
    c4 = u2x * (b2[1] - a2[1]) - u2y * (b2[0] - a2[0])

    # collinear cross products land on rounding noise rather than exact 0,
    # so compare against a tolerance scaled to each pair's own magnitudes
    tol_a = 1e-9 * math.hypot(ux, uy) * _reach(a1, a2, b2)
    tol_b = 1e-9 * math.hypot(u2x, u2y) * _reach(b1, b2, a2)

    s1 = _sign(c1, tol_a)
    s2 = _sign(c2, tol_a)
    s3 = _sign(c3, tol_b)
    s4 = _sign(c4, tol_b)

    return s1 != s2 and s3 != s4


def _sign(value: float, tolerance: float) -> float:
    return 1.0 if value > tolerance else (-1.0 if value < -tolerance else 0.0)


def _reach(p1: npt.NDArray[np.floating], p2: npt.NDArray[np.floating], origin: npt.NDArray[np.floating]) -> float:
    return max(math.hypot(p1[0] - origin[0], p1[1] - origin[1]),
               math.hypot(p2[0] - origin[0], p2[1] - origin[1]))


def rotation_matrix(angle: float) -> npt.NDArray[np.floating]:
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])


def vector_angle(vector: npt.NDArray[np.floating]) -> float:
    return math.atan2(vector[1], vector[0]) % (2 * math.pi)


def polar_to_Cartesian(angle: float, mag: float):
    return np.array([mag * math.cos(angle), mag * math.sin(angle)])
