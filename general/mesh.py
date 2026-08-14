class Mesh:
    def __init__(self, boundary):
        self.original_vertices = list(boundary.vertices)
        self.original_area = boundary.area()
        self.generated_quads = []

    def vertices(self):
        vertices = list(self.original_vertices)
        for quad in self.generated_quads:
            vertices.extend(v for v in quad.vertices if v not in vertices)
        return vertices

    def can_commit_quad(self, boundary, quad, reference_point):
        return quad.is_valid() and \
            not boundary.check_intersection_with_boundary(quad, reference_point)

    def commit_quad(self, boundary, quad):
        quad.connect_vertices()
        self.generated_quads.append(quad)
        return boundary.update_boundary(quad)

    def find_related_quads(self, vertex):
        return list({m for m in self.generated_quads if vertex in m.vertices})
