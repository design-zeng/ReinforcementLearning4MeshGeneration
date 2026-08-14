import json

from general.geometry import Vertex
from general.boundary import Boundary


def read_polygon(filename):
    with open(filename, 'r') as fr:
        vertices = json.loads(fr.readline())
    points = [Vertex(p[0]/100, p[1]/100) for p in vertices]
    boundary = Boundary(points)
    boundary.connect_vertices()
    return boundary


def write_inp(mesh, filename):
    if len(mesh.generated_quads) == 0:
        return

    nodes = mesh.vertices()

    with open(filename, 'w') as fw:
        fw.write("*NODE, NSET=ALLNODES\n")
        for idx, node in enumerate(nodes):
            fw.write(f"{idx+1}, {node.x}, {node.y}" + "\n")

        i = 0
        for i in range(1, len(mesh.original_vertices)):
            fw.write(f'*ELEMENT, TYPE=B21, ELSET=EB{i}\n {i+1}, {nodes.index(mesh.original_vertices[i-1]) + 1}, {nodes.index(mesh.original_vertices[i]) + 1}\n')
        fw.write(f'*ELEMENT, TYPE=S4R, ELSET=EB{i+1} \n')
        for idx, ele in enumerate(mesh.generated_quads):
            fw.write(f"{idx+1}, {nodes.index(ele.vertices[0]) + 1}, "
                     f"{nodes.index(ele.vertices[1]) + 1}, "
                     f"{nodes.index(ele.vertices[2]) + 1}, "
                     f"{nodes.index(ele.vertices[3]) + 1}" + "\n")
