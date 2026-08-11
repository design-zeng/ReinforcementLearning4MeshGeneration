import json
import math
import itertools

from general.components import Vertex, Segment
from general.boundary import Boundary


class Mesh:
    """The mesh generated so far: the accumulated quads plus the node ring
    (``boundary``) and the evolving front (``updated_boundary``). Owns the
    mesh-level relaxation smoothing, element quality, file export, and the
    offline sample extraction used to train the imitation network."""

    def __init__(self, boundary):
        self.boundary = boundary
        self.updated_boundary = boundary.copy()
        self.all_vertices = boundary.vertices
        self.original_vertices = list(boundary.vertices)
        self.average_edge_length = self.boundary.average_edge_length()

        self.generated_quads = []

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

    # --- mesh relaxation (smoothing) ---

    def smooth_pave(self, vertices, current_boundary_vertices, lr_1=None, lr_2=None, iteration=400, interior=False):
        # self.smooth_current_boundary(current_boundary_vertices, lr_1=lr_1, lr_2=lr_2, iteration=iteration)
        if not interior:
            self.updated_boundary.smooth_front(self.original_vertices)
        self.smooth_fixed_vertices([v for v in vertices if v not in current_boundary_vertices], iteration)
        self.updated_boundary.find_reference_candidates(target_angle=0)

    @staticmethod
    def estimate_4th_vertex(origin, left, right, factor=0.5, suggest_dist=None):
        distance = (origin.distance_to(left) + origin.distance_to(right)) * factor
        if suggest_dist is not None:
            distance = min(distance, 0.6 * suggest_dist)
        s = Segment.get_ray_segment(Segment(origin, left), Segment(origin, right), distance)
        return s.point2

    def smooth_fixed_vertices(self, vertices, iteration):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        #  modified after running, needs to be checked later on
        while diffs > 0.001 and i_iteration < iteration:
            # while diffs > 0.001:
            i_iteration += 1
            new_sum_coordinates = 0
            for vertex in vertices:
                if vertex in self.original_vertices:
                    continue
                x = 0
                y = 0
                count = 0
                connected_vertices = vertex.get_connected_vertices()

                for connect_v in connected_vertices:
                    x += connect_v.x + vertex.x
                    y += connect_v.y + vertex.y
                    count += 1
                if count == 0:
                    continue
                vertex.x = x / (2 * count)
                vertex.y = y / (2 * count)

                new_sum_coordinates += vertex.x + vertex.y

            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates
        print(f"Smoothing fixed vertices,Iteration numbers: {i_iteration}, the diff of smoothing is {diffs}!")

    def smooth(self, vertices, lr_1=0.999, lr_2=0.999, iteration=400):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        #  modified after running, needs to be checked later on
        while diffs > 0.001 and i_iteration < iteration:
            # while diffs > 0.001:
            i_iteration += 1
            new_sum_coordinates = 0
            for vertex in vertices:
                if vertex in self.original_vertices:
                    continue
                x = 0
                y = 0
                count = 0
                connected_vertices = vertex.get_connected_vertices()
                near_meshes = self.find_related_meshes(vertex)
                if len(near_meshes) == 1:
                    lr = lr_1
                    if len(connected_vertices) == 2:
                        origins = connected_vertices[0].get_common_vertex(connected_vertices[1])
                        origin = [v for v in origins if v is not vertex]
                        origin = Boundary.compute_dist(origin, vertex)[0][0]

                        p_dist = Boundary.compute_dist(
                            [
                                v for v in self.updated_boundary.vertices
                                if v not in connected_vertices and v is not vertex
                            ],
                            origin
                        )
                        if not len(p_dist):
                            continue
                        cloest_p, dist = p_dist[0]

                        estimate_vertex = self.estimate_4th_vertex(origin, connected_vertices[0], connected_vertices[1], suggest_dist=dist)

                        vertex.x = estimate_vertex.x
                        vertex.y = estimate_vertex.y


                    else:
                        for connect_v in connected_vertices:
                            vertex.x = lr * vertex.x + (1 - lr) * connect_v.x
                            vertex.y = lr * vertex.y + (1 - lr) * connect_v.y

                elif len(near_meshes) == 2:
                    update_boundary_vertices = []
                    inside_vertices = []
                    for v in connected_vertices:
                        if v in self.updated_boundary.vertices:
                            update_boundary_vertices.append(v)
                        else:
                            inside_vertices.append(v)

                    if len(inside_vertices) == 1 and len(update_boundary_vertices) == 2:
                        inside_vertex = inside_vertices[0]

                        origins = inside_vertex.get_common_vertex(update_boundary_vertices[0])
                        common_v1s = [v for v in origins if v is not vertex]
                        common_v1s = Boundary.compute_dist(common_v1s, vertex)[0]

                        origins = inside_vertex.get_common_vertex(update_boundary_vertices[1])
                        common_v2s = [v for v in origins if v is not vertex]
                        common_v2s = Boundary.compute_dist(common_v2s, vertex)[0]

                        common_v1 = common_v1s[0]
                        common_v2 = common_v2s[0]

                        estimate_vertex_1 = self.estimate_4th_vertex(common_v1, update_boundary_vertices[0], inside_vertex, factor=0.7)

                        estimate_vertex_2 = self.estimate_4th_vertex(common_v2, update_boundary_vertices[1], inside_vertex, factor=0.7)

                        vertex.x = (estimate_vertex_1.x + estimate_vertex_2.x) / 2
                        vertex.y = (estimate_vertex_1.y + estimate_vertex_2.y) / 2

                    else:
                        lr = lr_2
                        for connect_v in connected_vertices:
                            vertex.x = lr * vertex.x + (1 - lr) * connect_v.x
                            vertex.y = lr * vertex.y + (1 - lr) * connect_v.y
                else:
                    for connect_v in connected_vertices:
                        x += connect_v.x + vertex.x
                        y += connect_v.y + vertex.y
                        count += 1
                    if count == 0:
                        continue
                    vertex.x = x / (2 * count)
                    vertex.y = y / (2 * count)
            for vertex in vertices:
                new_sum_coordinates += vertex.x + vertex.y
            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates
        print(f"Iteration numbers: {i_iteration}, the diff of smoothing is {diffs}!")

        self.updated_boundary.find_reference_candidates(target_angle=0)

    # --- offline sample extraction (training data for the imitation network) ---

    @staticmethod
    def get_nodes(root, exclusion, layer, path, paths, N):
        if root is None:
            return

        if len(path) < N:
            path.append(root)
        else:
            path[-layer-1] = root
        if layer == 0:
            paths.append(list(path))
            return
        else:
            nodes = [v for v in root.get_connected_vertices() if v not in exclusion and v not in path[:N-layer]]
            for i in range(len(nodes)):
                Mesh.get_nodes(nodes[i], exclusion, layer-1, path, paths, N)

    def extract_samples_2(self, quads, n_neighbor, n_radius, radius, index=1, quality_threshold=0.7):
        all_samples, outputs, types = [], [], []
        for id, element in enumerate(quads):
            print(f"Extracting element {id} out of {len(quads)}")
            if self.get_quality(element, index=index) >= quality_threshold:
                for i in range(4):
                    rp = element.vertices[i]
                    l_p = element.vertices[(i + 1) % 4]
                    r_p = element.vertices[(i - 1)]
                    target = element.vertices[(i - 2)]
                    l_p_path, l_p_paths, exclusion = [], [], [rp, r_p]
                    self.get_nodes(l_p, exclusion, n_neighbor - 1, l_p_path, l_p_paths, n_neighbor)

                    r_p_path, r_p_paths, exclusion = [], [], [rp, l_p]
                    self.get_nodes(r_p, exclusion, n_neighbor - 1, r_p_path, r_p_paths, n_neighbor)

                    radius_neighbors = self.get_radius_neighbors(rp, l_p, r_p, [rp, l_p, r_p, target], radius=radius, N=n_radius)

                    samples = list(itertools.product(
                        [
                            _path for _path in r_p_paths
                            if len(_path) == n_neighbor
                        ],
                        radius_neighbors,
                        [
                            _path for _path in l_p_paths
                            if len(_path) == n_neighbor
                        ]
                    ))

                    for rr, mm, ll in samples:
                        if target in rr and target in ll:
                            continue
                        _sample = []
                        base_length = (rp.distance_to(rr[0]) +
                            sum([rr[j].distance_to(rr[j-1]) for j in range(1, len(rr))]) +
                            sum([ll[j].distance_to(ll[j-1]) for j in range(1, len(ll))]) +
                            rp.distance_to(ll[0])) / (2 * n_neighbor)

                        def encode(p):
                            return [rp.distance_to(p) / (base_length * radius),
                                    rp.to_find_clockwise_angle(p, r_p) % round(2 * math.pi, 4)]

                        for p in rr:
                            _sample.extend(encode(p))
                        for p in mm:
                            _sample.extend(encode(p))
                        for p in reversed(ll):
                            _sample.extend(encode(p))
                        _target = encode(target)

                        if target in rr:
                            types.append([1])
                        elif target in ll:
                            types.append([0])
                        else:
                            types.append([0.5])
                        all_samples.append(_sample)
                        outputs.append(_target)
        print("Done!")
        return all_samples, types, outputs

    def get_radius_neighbors(self, base_point, start_point, end_point, exclusion, radius, N=3):
        def radius_neighbors_with_angle(start_angle, end_angle):
            base_length = radius * (0.5 * base_point.distance_to(start_point) + 0.5 * base_point.distance_to(end_point))
            closest_neighbors = self.boundary.get_closest_points(
                self.boundary.get_points_within_angle(
                    self.boundary.vertices, base_point, start_point, start_angle, end_angle
                ),
                base_point,
                exclusion=exclusion,
                S_T=base_length
            )

            _angle = base_point.to_find_clockwise_angle(start_point, Vertex(base_point.x + 1, base_point.y))

            closest_neighbors.append(base_point +
                                    Vertex(base_length * math.cos(_angle - (start_angle + end_angle) / 2),
                                           base_length * math.sin((_angle - (start_angle + end_angle) / 2))))
            return closest_neighbors

        angle = base_point.to_find_clockwise_angle(start_point, end_point)
        angles = [i * angle / N for i in range(N+1)]
        neighbors = [radius_neighbors_with_angle(angles[i-1], angles[i]) for i in range(1, N+1)]
        # left_neighbors = radius_neighbors_with_angle(0.01, angle / 3)
        # middle_neighbors = radius_neighbors_with_angle(angle / 3, 2 * angle / 3)
        # right_neighbors = radius_neighbors_with_angle(2 * angle / 3, angle * 0.99)
        all_combinations = list(itertools.product(*reversed(neighbors)))
        return all_combinations

    @staticmethod
    def save_samples(file_name, res, _type=1):
        if _type == 1:
            res['samples'] = [Vertex.points_as_array(s) for s in res['samples']]
            res['outputs'] = [Vertex.points_as_array(s) for s in res['outputs']]
        with open(file_name, 'w') as fw:
            json.dump(res, fw)
