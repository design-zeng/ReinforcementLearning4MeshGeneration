import unittest

from general.components import *
from general.mesh import MeshGeneration as mg

class MeshTest(unittest.TestCase):
    def setUp(self):
        points = [Vertex(0, 1), Vertex(0, 2), Vertex(0, 3),
                  Vertex(1, 3), Vertex(2, 3), Vertex(3, 3),
                  Vertex(3, 2), Vertex(3, 1), Vertex(3, 0),
                  Vertex(2, 0), Vertex(1, 0), Vertex(0, 0)]

        for i in range(len(points)):
            segmt = Segment(points[i - 1], points[i])
            points[i - 1].assign_segment(segmt)
            points[i].assign_segment(segmt)

        self.boundary = Boundary2D(points)

    def test_calculate_crossing_vertices(self):
        ray_segment = Segment(Vertex(-1, -1), Vertex(4, 4))
        points = mg.calculate_crossing_vertices(self.boundary.vertices, ray_segment)
        self.assertEqual(points, 2)

    def test_calculate_crossing_vertices_2(self):
        ray_segment = Segment(Vertex(0, 0), Vertex(4, 4))
        points = mg.calculate_crossing_vertices_2(self.boundary.vertices, ray_segment)
        self.assertEqual(points, 2)


    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()