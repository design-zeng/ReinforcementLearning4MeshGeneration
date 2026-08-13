import math

from general.geometry import Segment, Polygon, Quad


class Boundary(Polygon):
    def __init__(self, vertices, num_ref_neighbor=4, maximum_reference_angle=math.pi * 0.972):
        super().__init__(vertices)
        self.num_ref_neighbor = num_ref_neighbor
        self.maximum_reference_angle = maximum_reference_angle
        self.candidate_vertices = None

    def reference_angle_score(self, vertex, index=None):
        if index is None:
            index = self.vertices.index(vertex)
        n = len(self.vertices)
        half = self.num_ref_neighbor // 2
        if half == 2:
            lam = 0.618
            weights = [lam, 1 - lam]
        else:
            weights = [2 / self.num_ref_neighbor] * half

        sum_angle = 0
        for i in range(half):
            clockwise_angle = vertex.clockwise_angle(
                self.vertices[(index + 1 + i) % n], self.vertices[index - 1 - i])
            if i == 0 and (clockwise_angle >= self.maximum_reference_angle or clockwise_angle == 0):
                return None
            sum_angle += clockwise_angle * weights[i]
        return math.degrees(sum_angle)

    def find_reference_candidates(self, target_angle):
        candidates = [(vertex, ad) for i, vertex in enumerate(self.vertices)
                      if (ad := self.reference_angle_score(vertex, i)) is not None]
        self.candidate_vertices = sorted(candidates, key=lambda x: math.fabs(x[1] - target_angle))

    def find_reference_point(self, not_valid_points=None, target_angle=0):
        if self.candidate_vertices is None:
            self.find_reference_candidates(target_angle)
        if not self.candidate_vertices:
            return None
        if not not_valid_points:
            return self.candidate_vertices[0][0]
        for vertex, _ in self.candidate_vertices:
            if not vertex.coincides_with_any(not_valid_points):
                return vertex
        return None

    def add_reference_candidates(self, points):
        for vertex in points:
            angle_dist = self.reference_angle_score(vertex)
            if angle_dist is None:
                continue
            i = 0
            while i < len(self.candidate_vertices):
                if angle_dist <= self.candidate_vertices[i][1]:
                    self.candidate_vertices.insert(i, (vertex, angle_dist))
                    break
                i += 1
            else:
                self.candidate_vertices.append((vertex, angle_dist))

    def remove_reference_candidates(self, points):
        for p in points:
            i = 0
            while i < len(self.candidate_vertices):
                if self.candidate_vertices[i][0] == p:
                    del self.candidate_vertices[i]
                else:
                    i += 1

    def rule_element(self, rule, index, new_point=None):
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
            prev, nxt = self.vertices[index - 1], self.vertices[(index + 1) % n]
            for seg in checking_segs:
                if prev not in quad.vertices and seg.is_intersecting(Segment(v, prev)):
                    return True
                if nxt not in quad.vertices and seg.is_intersecting(Segment(v, nxt)):
                    return True
        return False

    def update_boundary(self, reference_point, quad, mesh_boundary):
        new_vertices = [v for v in quad.vertices if v not in self.vertices]
        half = self.num_ref_neighbor // 2

        if len(new_vertices) == 1:
            new_v = new_vertices[0]
            replaced = quad.vertices[quad.vertices.index(new_v) - 2]
            idx = self.vertices.index(replaced)
            self.vertices.insert(idx, new_v)
            self.vertices.remove(replaced)
            if new_v not in mesh_boundary.vertices:
                mesh_boundary.vertices.append(new_v)

            ref_neighbors = []
            for i in range(half):
                ref_neighbors += [self.vertices[(idx + i + 1) % len(self.vertices)],
                                  self.vertices[idx - i - 1]]
            self.remove_reference_candidates(ref_neighbors + [replaced])
            self.add_reference_candidates(ref_neighbors)

        elif len(new_vertices) == 0:
            removable = [v for v in quad.vertices if self.n_sides(v) < 3]
            for v in removable:
                self.vertices.remove(v)

            idx = max(self.vertices.index(v) for v in quad.vertices if v not in removable)
            ref_neighbors = []
            for i in range(half):
                ref_neighbors += [self.vertices[(idx + i) % len(self.vertices)],
                                  self.vertices[idx - i - 1]]
            self.remove_reference_candidates(removable + ref_neighbors)
            self.add_reference_candidates(ref_neighbors)
