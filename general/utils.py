import json

from general.components import Vertex
from general.boundary import Boundary


def read_polygon(filename):
    with open(filename, 'r') as fr:
        vertices = json.loads(fr.readline())
    points = [Vertex(p[0]/100, p[1]/100) for p in vertices]
    boundary = Boundary(points)
    boundary.connect_vertices()
    return boundary
