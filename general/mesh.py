import math

from general.components import *
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

    def compute_boundary_quality(self, add_v):
        v = self.updated_boundary.vertices
        n = len(v)
        index = v.index(add_v)
        angles = []
        # product = 1
        for i in [1, -1]:
            angle = v[(index + i) % n].to_find_clockwise_angle(
                v[(index + i + 1) % n],
                v[index + i - 1])
            if angle < math.pi / 3:
                angles.append(angle)
                # product *= 3 * angle / math.pi
        # return math.pow(product, 1 / 2)
        q1 = 3 * min(angles) / math.pi if len(angles) else 1
        # q1 = product

        close_vs = []
        dist = add_v.distance_to(v[(index + 1) % n]) + add_v.distance_to(v[index - 1])
        for i, vv in enumerate(v):
            if vv in [v[index],
                      v[(index + 1) % n],
                      v[(index + 2) % n],
                      v[index - 1],
                      v[index - 2]]:
                continue
            if add_v.distance_to(vv) < dist:
                if i - 1 in close_vs:
                    continue
                close_vs.append(i)
        dists = []
        for i in close_vs:
            seg = Segment(v[(i + 1) % n], v[i])
            dists.append(seg.distance(add_v))

        target_len = dist / 2

        _dists = [(index + i) % n for i in range(-2, 3)]
        mean_dist = sum([v[_dists[i]].distance_to(
            v[_dists[i + 1]]) for i in range(len(_dists) - 1)]) / (len(_dists) - 1)

        smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)

        if len(dists):
            m_d = min(dists)
            q2 = m_d / (0.5 * dist) if m_d < 0.5 * dist else 1
        else:
            q2 = 1

        pow = 1/3
        return math.pow(smoothness * q1 * q2, pow)

    def compute_ele_boundary_quality(self, element):
        v = self.updated_boundary.vertices
        n = len(v)
        new_vs = [x for x in element.vertices
                  if len(x.get_connected_vertices()) == 2 and x in v]

        if len(new_vs):
            return self.compute_boundary_quality(new_vs[0]) # * self.compute_boundary_narrowness(new_vs[0])
        else:
            target_vs = [x for x in element.vertices if x in v]

            if not len(target_vs):
                print("No enough vertices to compute boundary quality!")
                return 1

            angles, dists = [], []
            # product = 1
            for i, x in enumerate(target_vs):
                index = v.index(x)
                angle = x.to_find_clockwise_angle(
                    v[(index + 1) % n],
                    v[index - 1])
                if angle < math.pi / 3:
                    angles.append(angle)
                    # product *= 2 * angle / math.pi
            # return math.pow(product, 1 / len(target_vs))
            index_1, index_r = v.index(target_vs[0]), v.index(target_vs[1])
            index = index_1 if index_1 < index_r else index_r

            target_len = target_vs[0].distance_to(target_vs[1])

            dists = [(index + i) % n for i in range(-2, 4)]
            mean_dist = sum([
                v[dists[i]].distance_to(v[dists[i+1]])
                for i in range(len(dists) - 1)
            ]) / (len(dists) - 1)

            smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)
            angle_quality = 3 * min(angles) / math.pi if len(angles) else 1
            pow = 1/2
            return math.pow(angle_quality * smoothness, pow)

    def find_related_meshes(self, vertex):
        return list({m for m in self.generated_quads if vertex in m.vertices})

    def estimate_area_range(self):
        # e_min/e_max robustified to 2nd shortest/longest edge, capped at 2*mean, not raw extremes
        lengths = [l[1] for l in self.boundary.sort_segments_by_length()]
        L = sum(lengths) / len(lengths)
        max_L = min(lengths[-2], 2 * L)
        min_L = min(L / math.sqrt(2), lengths[1])
        return min_L, (max_L + 3 * min_L) / 4

    def get_quality(self, element, index=0):
        if index == 1:
            return element.get_quality(quality_type='edge_angle')
        elif index == 2:
            b_reward = self.compute_ele_boundary_quality(element)
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
