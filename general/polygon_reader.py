import json

from general.components import *
from general.mesh import connect_vertices


def read_polygon(filename):
    with open(filename, 'r') as fr:
        vertices = json.loads(fr.readline())
    points = [Vertex(p[0]/100, p[1]/100) for p in vertices]
    connect_vertices(points)
    env = Boundary2D(points)
    return env