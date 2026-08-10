import math
from typing import Any

from general.components import *
from general.smoothing import SmoothingMixin
from general.sample_extraction import SampleExtractionMixin


class MeshGeneration(SmoothingMixin, SampleExtractionMixin):
    def __init__(self, boundary):
        self.boundary = boundary
        self.generated_meshes = []
        self.updated_boundary = boundary.copy()
        self.all_vertices = boundary.vertices
        self.original_vertices = list(boundary.vertices)
        self.candidate_vertices: Any = None
        self.test_candidate_vertices: Any = []
        self.maximum_reference_angle = math.pi * 0.972
        self.num_ref_neighbor = 4
        self.rp_index = 0
        self.average_edge_length = self.boundary.average_edge_length()

    @staticmethod
    def calculate_crossing_segments(closed_curve_vertices, ray_segment):
        '''
        http://geomalgorithms.com/a03-_inclusion.html#:~:text=Inclusion%20of%20a%20Point%20in%20a%20Polygon&text=%2D%20which%20counts%20the%20number%20of,%22even%2Dodd%22%20test.
        :param closed_curve_vertices:
        :param ray_segment:
        :return:
        '''
        count = 0
        n = len(closed_curve_vertices)
        ray_y = ray_segment.point2.y
        for i in range(n):
            v_i, v_p = closed_curve_vertices[i], closed_curve_vertices[i - 1]
            orientation = round(v_i.y - v_p.y, 4)
            if orientation == 0:
                continue
            if not Segment(v_i, v_p).is_cross(ray_segment):
                continue
            if round(v_i.y - ray_y, 4) == 0:
                no = round(closed_curve_vertices[(i + 1) % n].y - v_i.y, 4)
                count += (no * orientation > 0 and orientation < 0)
            elif round(v_p.y - ray_y, 4) == 0:
                po = round(v_p.y - closed_curve_vertices[i - 2].y, 4)
                count += (po * orientation > 0 and orientation > 0)
            else:
                count += 1
        return count

    def count_crossing_segments(self, ray_segment):
        return self.calculate_crossing_segments(self.updated_boundary.vertices, ray_segment)

    def is_inside(self, ray_segment):
        return self.count_crossing_segments(ray_segment) % 2 != 0

    def remove_reference_candidates(self, points):
        if isinstance(points, list):
            for p in points:
                i = 0
                while i < len(self.candidate_vertices):
                    if self.candidate_vertices[i][0] == p:
                        del self.candidate_vertices[i]
                    else:
                        i += 1
                i = 0
                while i < len(self.test_candidate_vertices):
                    if self.test_candidate_vertices[i][0] == p:
                        del self.test_candidate_vertices[i]
                    else:
                        i += 1
        else:
            raise ValueError("Points should be a list!")

    def add_reference_candidates(self, points):
        if isinstance(points, list):
            # v_angles = []
            for v in points:
                angle_dist = self.check_boundary_point(v)
                if angle_dist is not None:
                    i = 0
                    while i < len(self.candidate_vertices):
                        if angle_dist <= self.candidate_vertices[i][1]:
                            self.candidate_vertices.insert(i, (v, angle_dist))
                            break
                        # if angle_dist[1] <= self.candidate_vertices[i][2]:
                        #     self.candidate_vertices.insert(i, (v, angle_dist[0], angle_dist[1]))
                        #     break
                        i += 1
                    else:
                        self.candidate_vertices.append((v, angle_dist))
                    # v_angles.append([v, angle, dist])
            # self.test_candidate_vertices.extend(sorted(v_angles, key=lambda x: x[1]))
        else:
            raise ValueError("Points should be a list!")

    def check_boundary_point(self, vertex, index=None):
        if index is None:
            index = self.updated_boundary.vertices.index(vertex)

        if self.num_ref_neighbor % 2 != 0:
            raise ValueError('Wrong number of reference neighbors')
        sum_angle = 0
        if self.num_ref_neighbor // 2 == 2:
            lam = 0.618
            weights = [lam, 1-lam]
        else:
            weights = [2/self.num_ref_neighbor for i in range(self.num_ref_neighbor // 2)]

        for i in range(self.num_ref_neighbor // 2):
            clockwise_angle = vertex.to_find_clockwise_angle(
                self.updated_boundary.vertices[(index + 1 + i) % len(self.updated_boundary.vertices)],
                self.updated_boundary.vertices[index - 1 - i]
            )
            if i == 0:
                if clockwise_angle >= self.maximum_reference_angle or clockwise_angle == 0:  # 175
                    return
            sum_angle += clockwise_angle * weights[i]

        return math.degrees(sum_angle)

    def find_reference_candidates(self, target_angle):
        candidate_vertices = []
        for i, vertex in enumerate(self.updated_boundary.vertices):
            angle_dist = self.check_boundary_point(vertex, i)
            if angle_dist is not None:
                candidate_vertices.append((vertex, angle_dist))
        self.candidate_vertices = sorted(candidate_vertices, key=lambda x: math.fabs(x[1] - target_angle))

    def find_reference_point(self, not_valid_points=None, target_angle=0):
        if self.candidate_vertices is None:
            self.find_reference_candidates(target_angle)

        assert self.candidate_vertices is not None
        if len(self.candidate_vertices):
            if not_valid_points:
                for v in self.candidate_vertices:
                    if not self.is_vertex_inside_list(v[0], not_valid_points):
                        return v[0]
            else:
                return self.candidate_vertices[0][0]

    def compute_boundary_quality(self, add_v):
        index = self.updated_boundary.vertices.index(add_v)
        angles = []
        # product = 1
        for i in [1, -1]:
            angle = self.updated_boundary.vertices[(index + i) % len(self.updated_boundary.vertices)].to_find_clockwise_angle(
                self.updated_boundary.vertices[(index + i + 1) % len(self.updated_boundary.vertices)],
                self.updated_boundary.vertices[index + i - 1])
            if angle < math.pi / 3:
                angles.append(angle)
                # product *= 3 * angle / math.pi
        # return math.pow(product, 1 / 2)
        q1 = 3 * min(angles) / math.pi if len(angles) else 1
        # q1 = product

        close_vs = []
        dist = add_v.distance_to(self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)]) + add_v.distance_to(self.updated_boundary.vertices[index - 1])
        for i, v in enumerate(self.updated_boundary.vertices):
            if v in [self.updated_boundary.vertices[index],
                     self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                     self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                     self.updated_boundary.vertices[index - 1],
                     self.updated_boundary.vertices[index - 2]]:
                continue
            if add_v.distance_to(v) < dist:
                if i - 1 in close_vs:
                    continue
                close_vs.append(i)
        dists = []
        for i in close_vs:
            seg = Segment(self.updated_boundary.vertices[(i + 1) % len(self.updated_boundary.vertices)],
                          self.updated_boundary.vertices[i])
            dists.append(seg.distance(add_v))

        target_len = dist / 2

        _dists = [(index + i) % len(self.updated_boundary.vertices) for i in range(-2, 3)]
        mean_dist = sum([self.updated_boundary.vertices[_dists[i]].distance_to(
            self.updated_boundary.vertices[_dists[i + 1]]) for i in range(len(_dists) - 1)]) / (len(_dists) - 1)

        smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)

        if len(dists):
            m_d = min(dists)
            q2 = m_d / (0.5 * dist) if m_d < 0.5 * dist else 1
        else:
            q2 = 1

        pow = 1/3
        return math.pow(smoothness * q1 * q2, pow)

    def compute_ele_boundary_quality(self, element):
        new_vs = [v for v in element.vertices
                  if len(v.get_connected_vertices()) == 2 and v in self.updated_boundary.vertices]

        if len(new_vs):
            return self.compute_boundary_quality(new_vs[0]) # * self.compute_boundary_narrowness(new_vs[0])
        else:
            target_vs = [v for v in element.vertices if v in self.updated_boundary.vertices]

            if not len(target_vs):
                print("No enough vertices to compute boundary quality!")
                return 1

            angles, dists = [], []
            # product = 1
            for i, v in enumerate(target_vs):
                index = self.updated_boundary.vertices.index(v)
                angle = v.to_find_clockwise_angle(
                    self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[index - 1])
                if angle < math.pi / 3:
                    angles.append(angle)
                    # product *= 2 * angle / math.pi
            # return math.pow(product, 1 / len(target_vs))
            index_1, index_r = self.updated_boundary.vertices.index(target_vs[0]), \
                               self.updated_boundary.vertices.index(target_vs[1])
            index = index_1 if index_1 < index_r else index_r

            target_len = target_vs[0].distance_to(target_vs[1])

            dists = [(index + i) % len(self.updated_boundary.vertices) for i in range(-2, 4)]
            mean_dist = sum([
                self.updated_boundary.vertices[dists[i]].distance_to(
                    self.updated_boundary.vertices[dists[i+1]]
                )
                for i in range(len(dists) - 1)
            ]) / (len(dists) - 1)

            smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)
            angle_quality = 3 * min(angles) / math.pi if len(angles) else 1
            pow = 1/2
            return math.pow(angle_quality * smoothness, pow)

    def is_vertex_inside_list(self, vertex, points):
        return any(p.distance_to(vertex) < 0.001 for p in points)

    def count_segts_in_boundary(self, vertex):
        count = 0
        for seg in vertex.segments:
            if seg.point1 in self.updated_boundary.vertices and seg.point2 in self.updated_boundary.vertices:
                count += 1
        return count

    def get_neighbors(self, reference_point, num_points=4):
        return self.updated_boundary.get_neighbors(reference_point, num_points=num_points)

    def check_intersection_with_boundary(self, mesh, reference_point):
        max_dist = max([reference_point.distance_to(v) for v in mesh.vertices if v is not reference_point])
        neighboring_vertices = [v for v in self.updated_boundary.vertices
                                if reference_point.distance_to(v) < max_dist and v not in mesh.vertices]

        _index = mesh.vertices.index(reference_point)
        checking_sesg = [Segment(mesh.vertices[_index - 1], mesh.vertices[_index - 2]),
                         Segment(mesh.vertices[_index - 2], mesh.vertices[_index - 3])]
        for v in neighboring_vertices:
            index = self.updated_boundary.vertices.index(v)
            for c_g in checking_sesg:
                if self.updated_boundary.vertices[index - 1] not in mesh.vertices:
                    if c_g.is_cross(Segment(v, self.updated_boundary.vertices[index - 1])):
                        return True

                if self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)] not in mesh.vertices:
                    if c_g.is_cross(Segment(v, self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)])):
                        return True

        return False

    def validate_mesh(self, mesh, quality_method=0):
        return mesh.is_valid(quality_method)

    def is_point_inside_area(self, vertex):
        remote_dist = 10000
        ray_segment = Segment(vertex, Vertex(remote_dist, vertex.y))
        return self.is_inside(ray_segment)

    def find_related_meshes(self, vertex):
        return list({m for m in self.generated_meshes if vertex in m.vertices})

    def update_boundary(self, reference_point, mesh):
        new_vertices = []
        for v in mesh.vertices:
            if v not in self.updated_boundary.vertices:
                new_vertices.append(v)

        if len(new_vertices) == 1:
            id = self.updated_boundary.vertices.index(mesh.vertices[mesh.vertices.index(new_vertices[0]) - 2])
            self.updated_boundary.vertices.insert(id, new_vertices[0])
            self.remove_point(mesh.vertices[mesh.vertices.index(new_vertices[0]) - 2])
            if not new_vertices[0] in self.boundary.vertices:
                self.boundary.vertices.append(new_vertices[0])
            ref_neighbors = []
            for i in range(self.num_ref_neighbor // 2):
                ref_neighbors.extend([
                    self.updated_boundary.vertices[(id + i + 1) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[id - i - 1]
                ])
            self.remove_reference_candidates(ref_neighbors + [mesh.vertices[mesh.vertices.index(new_vertices[0]) - 2]])
            self.add_reference_candidates(ref_neighbors)

            self.rp_index += 1

        elif len(new_vertices) == 0:
            removable_vertices = []
            for v in mesh.vertices:
                if self.count_segts_in_boundary(v) < 3:
                    removable_vertices.append(v)
            for v in removable_vertices:
                self.updated_boundary.vertices.remove(v)

            id = max([self.updated_boundary.vertices.index(v) for v in mesh.vertices
                      if v not in removable_vertices])
            ref_neighbors = []
            for i in range(self.num_ref_neighbor // 2):
                ref_neighbors.extend([
                    self.updated_boundary.vertices[(id + i) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[id - i - 1]
                ])

            self.remove_reference_candidates(removable_vertices + ref_neighbors)
            self.add_reference_candidates(ref_neighbors)

            self.rp_index = max([self.updated_boundary.vertices.index(v) for v in mesh.vertices
                                 if v not in removable_vertices])

    def estimate_area_range(self):
        lengths = [l[1] for l in self.boundary.sort_segments_by_length()]

        # min_L = lengths[0]
        # return min_L ** 2, 1.2 * min_L ** 2
        L = sum(lengths) / len(lengths)
        max_L = min(lengths[-2], 2 * L)
        min_L = min(L / math.sqrt(2), lengths[1])
        # if len(self.candidate_vertices):
        #     angle = self.candidate_vertices[0][1]
        # else:
        #     raise ValueError('Empty angle in the candidate vertices!')
        # return min_L ** 2, ((max_L + 3 * min_L) / 4) ** 2, L
        return min_L, (max_L + 3 * min_L) / 4

    def remove_point(self, point):
        self.updated_boundary.vertices.remove(point)

    def compute_element_quality(self, element):
        q1, q2 = element.get_quality_3()
        return math.pow(q1 * q2, 1/2)

    def get_quality(self, element, index=0):
        if index == 1:
            return self.compute_element_quality(element)
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
        if len(self.generated_meshes) == 0:
            print("There are no elements generated!")
            return

        nodes = list(self.original_vertices)  # boundary nodes (referenced by the B21 edge elements)
        for ele in self.generated_meshes:
            nodes.extend([n for n in ele.vertices if n not in nodes])

        with open(filename, 'w') as fw:
            fw.write("*NODE, NSET=ALLNODES\n")
            for id, node in enumerate(nodes):
                fw.write(f"{id+1}, {node.x}, {node.y}" + "\n")

            i = 0
            for i in range(1, len(self.original_vertices)):
                fw.write(f'*ELEMENT, TYPE=B21, ELSET=EB{i}\n {i+1}, {nodes.index(self.original_vertices[i-1]) + 1}, {nodes.index(self.original_vertices[i]) + 1}\n')
            fw.write(f'*ELEMENT, TYPE=S4R, ELSET=EB{i+1} \n')
            for id, ele in enumerate(self.generated_meshes):
                fw.write(f"{id+1}, {nodes.index(ele.vertices[0]) + 1}, "
                         f"{nodes.index(ele.vertices[1]) + 1}, "
                         f"{nodes.index(ele.vertices[2]) + 1}, "
                         f"{nodes.index(ele.vertices[3]) + 1}" + "\n")
        print("Document writing is finished!")


def connect_vertices(points):
    for i in range(len(points)):
        segmt = Segment(points[i - 1], points[i])
        points[i - 1].assign_segment(segmt)
        points[i].assign_segment(segmt)
