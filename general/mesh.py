import json
import math
import itertools

from general.geometry import Vertex, Segment


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
            b_reward = boundary.compute_element_boundary_quality(element)
            e_reward = quad_quality(element, 'robust')
            return e_reward + 1 * (b_reward - 1)
        elif index == 4:
            return quad_quality(element, 'robust')
        elif index == 5:
            return quad_quality(element, 'strong')
        raise ValueError(f"Unknown quality index: {index}")

    def write_elements_to_file(self, filename, format='inp'):
        if len(self.generated_quads) == 0:
            return

        nodes = list(self.original_vertices)  # boundary nodes (referenced by the B21 edge elements)
        for ele in self.generated_quads:
            nodes.extend([n for n in ele.vertices if n not in nodes])

        with open(filename, 'w') as fw:
            fw.write("*NODE, NSET=ALLNODES\n")
            for idx, node in enumerate(nodes):
                fw.write(f"{idx+1}, {node.x}, {node.y}" + "\n")

            i = 0
            for i in range(1, len(self.original_vertices)):
                fw.write(f'*ELEMENT, TYPE=B21, ELSET=EB{i}\n {i+1}, {nodes.index(self.original_vertices[i-1]) + 1}, {nodes.index(self.original_vertices[i]) + 1}\n')
            fw.write(f'*ELEMENT, TYPE=S4R, ELSET=EB{i+1} \n')
            for idx, ele in enumerate(self.generated_quads):
                fw.write(f"{idx+1}, {nodes.index(ele.vertices[0]) + 1}, "
                         f"{nodes.index(ele.vertices[1]) + 1}, "
                         f"{nodes.index(ele.vertices[2]) + 1}, "
                         f"{nodes.index(ele.vertices[3]) + 1}" + "\n")

    # --- mesh relaxation (smoothing) ---

    def smooth_pave(self, boundary, vertices, current_boundary_vertices, lr_1=None, lr_2=None, iteration=400, interior=False):
        # self.smooth_current_boundary(current_boundary_vertices, lr_1=lr_1, lr_2=lr_2, iteration=iteration)
        if not interior:
            boundary.smooth_front(self.original_vertices)
        self.smooth_fixed_vertices([v for v in vertices if v not in current_boundary_vertices], iteration)
        boundary.find_reference_candidates(target_angle=0)

    def smooth_fixed_vertices(self, vertices, iteration):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        while diffs > 0.001 and i_iteration < iteration:
            i_iteration += 1
            new_sum_coordinates = 0
            for vertex in vertices:
                if vertex in self.original_vertices:
                    continue
                connected = vertex.get_neighbors()
                if not connected:
                    continue
                count = len(connected)
                vertex.x = sum(c.x + vertex.x for c in connected) / (2 * count)
                vertex.y = sum(c.y + vertex.y for c in connected) / (2 * count)
                new_sum_coordinates += vertex.x + vertex.y
            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates

    def smooth(self, boundary, vertices, lr_1=0.999, lr_2=0.999, iteration=400):
        sum_coordinates = 0
        diffs = 100
        i_iteration = 0
        while diffs > 0.001 and i_iteration < iteration:
            i_iteration += 1
            for vertex in vertices:
                if vertex not in self.original_vertices:
                    self.smooth_vertex(boundary, vertex, lr_1, lr_2)
            new_sum_coordinates = sum(v.x + v.y for v in vertices)
            diffs = math.fabs(new_sum_coordinates - sum_coordinates)
            sum_coordinates = new_sum_coordinates

        boundary.find_reference_candidates(target_angle=0)

    def smooth_vertex(self, boundary, vertex, lr_1, lr_2):
        connected = vertex.get_neighbors()
        n_meshes = len(self.find_related_quads(vertex))

        if n_meshes == 1 and len(connected) == 2:
            origins = [v for v in connected[0].get_common_neighbors(connected[1]) if v is not vertex]
            origin = vertex.sorted_by_distance(origins)[0][0]
            p_dist = origin.sorted_by_distance(
                [v for v in boundary.vertices
                 if v not in connected and v is not vertex])
            if not p_dist:
                return
            estimate = self.estimate_4th_vertex(origin, connected[0], connected[1], suggest_dist=p_dist[0][1])
            vertex.x, vertex.y = estimate.x, estimate.y
            return

        if n_meshes == 2:
            on_front = [v for v in connected if v in boundary.vertices]
            interior = [v for v in connected if v not in boundary.vertices]
            if len(interior) == 1 and len(on_front) == 2:
                inside = interior[0]
                origins1 = [v for v in inside.get_common_neighbors(on_front[0]) if v is not vertex]
                origins2 = [v for v in inside.get_common_neighbors(on_front[1]) if v is not vertex]
                e1 = self.estimate_4th_vertex(vertex.sorted_by_distance(origins1)[0][0], on_front[0], inside, factor=0.7)
                e2 = self.estimate_4th_vertex(vertex.sorted_by_distance(origins2)[0][0], on_front[1], inside, factor=0.7)
                vertex.x = (e1.x + e2.x) / 2
                vertex.y = (e1.y + e2.y) / 2
                return

        if n_meshes not in (1, 2):
            x = y = count = 0
            for c in connected:
                x += c.x + vertex.x
                y += c.y + vertex.y
                count += 1
            if count == 0:
                return
            vertex.x = x / (2 * count)
            vertex.y = y / (2 * count)
            return

        lr = lr_1 if n_meshes == 1 else lr_2
        for c in connected:
            vertex.x = lr * vertex.x + (1 - lr) * c.x
            vertex.y = lr * vertex.y + (1 - lr) * c.y

    # --- offline sample extraction (training data for the imitation network) ---

    def extract_samples(self, quads, n_neighbor, n_radius, radius, index=1, quality_threshold=0.7):
        all_samples, outputs, types = [], [], []
        for idx, element in enumerate(quads):
            if self.get_quality(self.boundary, element, index=index) >= quality_threshold:
                for i in range(4):
                    rp = element.vertices[i]
                    l_p = element.vertices[(i + 1) % 4]
                    r_p = element.vertices[(i - 1)]
                    target = element.vertices[(i - 2)]
                    l_p_path, l_p_paths, exclusion = [], [], [rp, r_p]
                    self.collect_neighbor_paths(l_p, exclusion, n_neighbor - 1, l_p_path, l_p_paths, n_neighbor)

                    r_p_path, r_p_paths, exclusion = [], [], [rp, l_p]
                    self.collect_neighbor_paths(r_p, exclusion, n_neighbor - 1, r_p_path, r_p_paths, n_neighbor)

                    radius_neighbors = self.boundary.get_radius_neighbors(rp, l_p, r_p, [rp, l_p, r_p, target], radius=radius, N=n_radius)

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
                                    rp.clockwise_angle(p, r_p) % round(2 * math.pi, 4)]

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
        return all_samples, types, outputs

    @staticmethod
    def estimate_4th_vertex(origin, left, right, factor=0.5, suggest_dist=None):
        distance = (origin.distance_to(left) + origin.distance_to(right)) * factor
        if suggest_dist is not None:
            distance = min(distance, 0.6 * suggest_dist)
        s = Segment.get_ray_segment(Segment(origin, left), Segment(origin, right), distance)
        return s.point2

    @staticmethod
    def collect_neighbor_paths(root, exclusion, layer, path, paths, N):
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
            nodes = [v for v in root.get_neighbors() if v not in exclusion and v not in path[:N-layer]]
            for i in range(len(nodes)):
                Mesh.collect_neighbor_paths(nodes[i], exclusion, layer-1, path, paths, N)

    @staticmethod
    def save_samples(file_name, res, _type=1):
        if _type == 1:
            res['samples'] = [Vertex.flatten(s) for s in res['samples']]
            res['outputs'] = [Vertex.flatten(s) for s in res['outputs']]
        with open(file_name, 'w') as fw:
            json.dump(res, fw)


def is_valid_quad(quad, quality_method=0):
    if quad.segments_crossed():
        return False
    if quality_method == 0:
        max_degree, min_degree = 0.99 * math.pi, 0.01 * math.pi
        for degree in quad.corner_angles():
            if degree > max_degree or degree < min_degree:
                return False
    return True


def edge_angle_quality(quad):
    length_of_edges = quad.edge_lengths()
    area = quad.area()
    if area <= 0:
        q1 = 0
    else:
        s = math.sqrt(area)
        product = 1
        for edge in length_of_edges:
            product *= math.pow(edge / s, 1 if s - edge > 0 else -1)
        q1 = math.pow(product, 1 / 4)

    angle_product = 1
    for a in quad.corner_angles():
        angle_product *= 1 - (math.fabs(math.degrees(a) - 90) / 90)
    if angle_product < 0:
        q2 = 0
    else:
        q2 = math.pow(angle_product, 1/4)

    return q1, q2


def quad_quality(quad, quality_type='robust'):
    if quality_type == 'stretch':
        return math.sqrt(2) * min(quad.edge_lengths()) / max(quad.vertices[0].distance_to(quad.vertices[2]), quad.vertices[1].distance_to(quad.vertices[3]))
    elif quality_type == 'robust':
        q1 = quad_quality(quad, 'stretch')
        angles = quad.corner_angles()
        q2 = min(angles) / max(angles)
        return math.sqrt(q1 * q2)
    elif quality_type == 'edge_angle':
        q1, q2 = edge_angle_quality(quad)
        return math.sqrt(q1 * q2)
    elif quality_type == 'taper':
        p0, p1, p2, p3 = quad.vertices[0], quad.vertices[-1], quad.vertices[-2], quad.vertices[-3]
        x1 = (p1 - p0) + (p2 - p3)
        x2 = (p2 - p1) + (p3 - p0)
        x12 = (p0 - p1) + (p2 - p3)
        return x12.length() / min(x1.length(), x2.length())
    elif quality_type == 's_jacobian':
        p0, p1, p2, p3 = quad.vertices[0], quad.vertices[-1], quad.vertices[-2], quad.vertices[-3]
        l0, l1, l2, l3 = p1-p0, p2-p1, p3-p2, p0-p3
        a3 = l2.cross(l3)
        a2 = l1.cross(l2)
        a1 = l0.cross(l1)
        a0 = l3.cross(l0)
        return min([a0 / (l0.length() * l3.length()),
                 a1 / (l0.length() * l1.length()),
                 a2 / (l1.length() * l2.length()),
                 a3 / (l2.length() * l3.length())])
    elif quality_type == 'strong':
        q1, _ = edge_angle_quality(quad)
        angles = [math.fabs(a) for a in quad.corner_angles()]
        q2 = min(angles) / max(angles)
        return math.sqrt(q1 * q2)
    raise ValueError(f"Unknown quality type: {quality_type}")
