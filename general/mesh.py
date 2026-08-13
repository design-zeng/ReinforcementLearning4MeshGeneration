from general.quad_quality import is_valid_quad, quad_quality
from general.boundary_quality import compute_element_boundary_quality


class Mesh:
    def __init__(self, boundary):
        self.boundary = boundary
        self.all_vertices = boundary.vertices
        self.original_vertices = list(boundary.vertices)
        self.average_edge_length = self.boundary.average_edge_length()

        self.original_boundary = boundary.deep_copy()
        self.original_area = boundary.area()
        self.generated_quads = []

    def reset(self):
        self.boundary = self.original_boundary.deep_copy()
        self.original_vertices = list(self.boundary.vertices)
        self.generated_quads = []

    def can_commit_quad(self, boundary, quad, reference_point):
        return is_valid_quad(quad, 0) and \
            not boundary.check_intersection_with_boundary(quad, reference_point)

    def commit_quad(self, boundary, quad, reference_point):
        quad.connect_vertices()
        self.generated_quads.append(quad)
        boundary.update_boundary(reference_point, quad, self.boundary)

    def find_related_quads(self, vertex):
        return list({m for m in self.generated_quads if vertex in m.vertices})

    def get_quality(self, boundary, element, index=0):
        if index == 1:
            return quad_quality(element, 'edge_angle')
        elif index == 2:
            b_reward = compute_element_boundary_quality(boundary, element)
            e_reward = quad_quality(element, 'robust')
            return e_reward + 1 * (b_reward - 1)
        elif index == 4:
            return quad_quality(element, 'robust')
        elif index == 5:
            return quad_quality(element, 'strong')
        raise ValueError(f"Unknown quality index: {index}")
