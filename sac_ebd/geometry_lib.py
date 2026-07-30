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

    s1 = 1.0 if c1 > 0 else (-1.0 if c1 < 0 else 0.0)
    s2 = 1.0 if c2 > 0 else (-1.0 if c2 < 0 else 0.0)
    s3 = 1.0 if c3 > 0 else (-1.0 if c3 < 0 else 0.0)
    s4 = 1.0 if c4 > 0 else (-1.0 if c4 < 0 else 0.0)

    return s1 != s2 and s3 != s4

def rotation_matrix(angle: float) -> npt.NDArray[np.floating]:
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])

def vector_angle(vector: npt.NDArray[np.floating]) -> float:
    return math.atan2(vector[1], vector[0]) % (2 * math.pi)

def polar_to_Cartesian(angle: float, mag: float):
    return np.array([mag * math.cos(angle), mag * math.sin(angle)])
