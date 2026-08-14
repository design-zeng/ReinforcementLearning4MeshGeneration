import math

from general.geometry import Segment, Polygon, Quad


class Boundary(Polygon):
    def __init__(self, vertices, num_ref_neighbor=4, maximum_reference_angle=math.pi * 0.972):
        super().__init__(vertices)
        self.num_ref_neighbor = num_ref_neighbor
        self.maximum_reference_angle = maximum_reference_angle

    def estimate_area_range(self):
        lengths = sorted(seg.length() for seg in self.own_segments())
        L = sum(lengths) / len(lengths)
        max_L = min(lengths[-2], 2 * L)
        min_L = min(L / math.sqrt(2), lengths[1])
        return min_L ** 2, ((max_L + 3 * min_L) / 4) ** 2

    def rule_quad(self, rule, index, new_point=None):
        v = self.vertices
        n = len(v)
        if rule == -1:
            return Quad([v[index - 1], v[index], v[(index + 1) % n], v[(index + 2) % n]])
        if rule == 1:
            return Quad([v[index - 2], v[index - 1], v[index], v[(index + 1) % n]])
        return Quad([new_point, v[index - 1], v[index], v[(index + 1) % n]])

    def check_intersection_with_boundary(self, quad, reference_point):
        max_dist = max(reference_point.distance_to(v) for v in quad.vertices if v is not reference_point)
        neighbouring = [v for v in self.vertices
                        if reference_point.distance_to(v) < max_dist and v not in quad.vertices]

        idx = quad.vertices.index(reference_point)
        checking_segs = [Segment(quad.vertices[idx - 1], quad.vertices[idx - 2]),
                         Segment(quad.vertices[idx - 2], quad.vertices[idx - 3])]
        n = len(self.vertices)
        for v in neighbouring:
            index = self.vertices.index(v)
            for u in (self.vertices[index - 1], self.vertices[(index + 1) % n]):
                if u not in quad.vertices and \
                        any(seg.is_intersecting(Segment(v, u)) for seg in checking_segs):
                    return True
        return False

    def update_boundary(self, quad):
        new_vertices = [v for v in quad.vertices if v not in self.vertices]
        half = self.num_ref_neighbor // 2

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
