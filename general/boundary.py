import math

from general.config import NUM_REF_NEIGHBOR, MAXIMUM_REFERENCE_ANGLE, TARGET_ANGLE, KAPPA, UPSILON
from general.geometry import Polygon


class Boundary(Polygon):
    def __init__(self, vertices):
        super().__init__(vertices)
        self.retired = []
        self.rescored = []

    def estimate_area_bounds(self):
        lengths = sorted(seg.length() for seg in self.own_segments())
        mean_length = sum(lengths) / len(lengths)
        e_max = min(lengths[-2], 2 * mean_length)
        e_min = min(mean_length / math.sqrt(2), lengths[1])
        return UPSILON * e_min ** 2, UPSILON * ((e_max + (KAPPA - 1) * e_min) / KAPPA) ** 2

    @staticmethod
    def absorb(quad, boundary):
        new_vertices = [v for v in quad.vertices if v not in boundary.vertices]
        half = NUM_REF_NEIGHBOR // 2

        if len(new_vertices) == 1:
            new_v = new_vertices[0]
            replaced = quad.vertices[quad.vertices.index(new_v) - 2]
            idx = boundary.vertices.index(replaced)
            boundary.vertices.insert(idx, new_v)
            boundary.vertices.remove(replaced)

            rescored = [v for i in range(half)
                        for v in (boundary.vertices[(idx + i + 1) % len(boundary.vertices)],
                                  boundary.vertices[idx - i - 1])]
            boundary.retired = rescored + [replaced]
            boundary.rescored = rescored

        elif len(new_vertices) == 0:
            removable = [v for v in quad.vertices if boundary.n_sides(v) < 3]
            for v in removable:
                boundary.vertices.remove(v)

            idx = max(boundary.vertices.index(v) for v in quad.vertices if v not in removable)
            rescored = [v for i in range(half)
                        for v in (boundary.vertices[(idx + i) % len(boundary.vertices)],
                                  boundary.vertices[idx - i - 1])]
            boundary.retired = removable + rescored
            boundary.rescored = rescored

    @staticmethod
    def weighted_reference_angle(boundary, vertex, index=None):
        if index is None:
            index = boundary.vertices.index(vertex)
        n = len(boundary.vertices)
        half = NUM_REF_NEIGHBOR // 2
        golden = 0.618
        weights = [golden, 1 - golden] if half == 2 else [2 / NUM_REF_NEIGHBOR] * half

        weighted_angle = 0
        for i in range(half):
            angle = vertex.clockwise_angle(
                boundary.vertices[(index + 1 + i) % n], boundary.vertices[index - 1 - i])
            if i == 0 and (angle >= MAXIMUM_REFERENCE_ANGLE or angle == 0):
                return None
            weighted_angle += angle * weights[i]
        return math.degrees(weighted_angle)

    @staticmethod
    def score_candidates(boundary, candidates):
        candidates[:] = sorted(
            ((vertex, score) for i, vertex in enumerate(boundary.vertices)
             if (score := Boundary.weighted_reference_angle(boundary, vertex, i)) is not None),
            key=lambda x: math.fabs(x[1] - TARGET_ANGLE))
