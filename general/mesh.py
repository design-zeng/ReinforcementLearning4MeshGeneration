from general.smoothing import SmoothingMixin
from general.sample_extraction import SampleExtractionMixin


class Mesher(SmoothingMixin, SampleExtractionMixin):
    def __init__(self, boundary):
        self.boundary = boundary
        self.generated_quads = []
        self.updated_boundary = boundary.copy()
        self.all_vertices = boundary.vertices
        self.original_vertices = list(boundary.vertices)
        self.average_edge_length = self.boundary.average_edge_length()

    def find_related_meshes(self, vertex):
        return list({m for m in self.generated_quads if vertex in m.vertices})

    def get_quality(self, element, index=0):
        if index == 1:
            return element.get_quality(quality_type='edge_angle')
        elif index == 2:
            b_reward = self.updated_boundary.compute_ele_boundary_quality(element)
            e_reward = element.get_quality(quality_type='robust')
            return e_reward + 1 * (b_reward - 1)
        elif index == 4:
            return element.get_quality(quality_type='robust')
        elif index == 5:
            return element.get_quality(quality_type='strong')
        raise ValueError(f"Unknown quality index: {index}")

    def write_generated_elements_2_file(self, filename, format='inp'):
        if len(self.generated_quads) == 0:
            print("There are no elements generated!")
            return

        nodes = list(self.original_vertices)  # boundary nodes (referenced by the B21 edge elements)
        for ele in self.generated_quads:
            nodes.extend([n for n in ele.vertices if n not in nodes])

        with open(filename, 'w') as fw:
            fw.write("*NODE, NSET=ALLNODES\n")
            for id, node in enumerate(nodes):
                fw.write(f"{id+1}, {node.x}, {node.y}" + "\n")

            i = 0
            for i in range(1, len(self.original_vertices)):
                fw.write(f'*ELEMENT, TYPE=B21, ELSET=EB{i}\n {i+1}, {nodes.index(self.original_vertices[i-1]) + 1}, {nodes.index(self.original_vertices[i]) + 1}\n')
            fw.write(f'*ELEMENT, TYPE=S4R, ELSET=EB{i+1} \n')
            for id, ele in enumerate(self.generated_quads):
                fw.write(f"{id+1}, {nodes.index(ele.vertices[0]) + 1}, "
                         f"{nodes.index(ele.vertices[1]) + 1}, "
                         f"{nodes.index(ele.vertices[2]) + 1}, "
                         f"{nodes.index(ele.vertices[3]) + 1}" + "\n")
        print("Document writing is finished!")
