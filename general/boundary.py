import math

from general.config import NUM_REF_NEIGHBOR
from general.geometry import Polygon


class Boundary(Polygon):
    def __init__(self, vertices):
        super().__init__(vertices)

    def estimate_area_range(self):
        lengths = sorted(seg.length() for seg in self.own_segments())
        L = sum(lengths) / len(lengths)
        max_L = min(lengths[-2], 2 * L)
        min_L = min(L / math.sqrt(2), lengths[1])
        return min_L ** 2, ((max_L + 3 * min_L) / 4) ** 2

    def update_boundary(self, quad):
        new_vertices = [v for v in quad.vertices if v not in self.vertices]
        half = NUM_REF_NEIGHBOR // 2

        if len(new_vertices) == 1:
            new_v = new_vertices[0]
            replaced = quad.vertices[quad.vertices.index(new_v) - 2]
            idx = self.vertices.index(replaced)
            self.vertices.insert(idx, new_v)
            self.vertices.remove(replaced)

            rescored = [v for i in range(half)
                        for v in (self.vertices[(idx + i + 1) % len(self.vertices)],
                                  self.vertices[idx - i - 1])]
            return rescored + [replaced], rescored

        elif len(new_vertices) == 0:
            removable = [v for v in quad.vertices if self.n_sides(v) < 3]
            for v in removable:
                self.vertices.remove(v)

            idx = max(self.vertices.index(v) for v in quad.vertices if v not in removable)
            rescored = [v for i in range(half)
                        for v in (self.vertices[(idx + i) % len(self.vertices)],
                                  self.vertices[idx - i - 1])]
            return removable + rescored, rescored
