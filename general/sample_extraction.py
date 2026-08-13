import json
import math
import itertools

from general.geometry import Vertex, Angle


def extract_samples(mesh, quads, n_neighbor, n_radius, radius, index=1, quality_threshold=0.7):
    all_samples, outputs, types = [], [], []
    for idx, element in enumerate(quads):
        if mesh.get_quality(mesh.boundary, element, index=index) >= quality_threshold:
            for i in range(4):
                rp = element.vertices[i]
                l_p = element.vertices[(i + 1) % 4]
                r_p = element.vertices[(i - 1)]
                target = element.vertices[(i - 2)]
                l_p_path, l_p_paths, exclusion = [], [], [rp, r_p]
                collect_neighbor_paths(l_p, exclusion, n_neighbor - 1, l_p_path, l_p_paths, n_neighbor)

                r_p_path, r_p_paths, exclusion = [], [], [rp, l_p]
                collect_neighbor_paths(r_p, exclusion, n_neighbor - 1, r_p_path, r_p_paths, n_neighbor)

                radius_neighbors = get_radius_neighbors(mesh.boundary, rp, l_p, r_p, [rp, l_p, r_p, target], radius=radius, N=n_radius)

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
            collect_neighbor_paths(nodes[i], exclusion, layer-1, path, paths, N)


def get_radius_neighbors(boundary, base_point, start_point, end_point, exclusion, radius, N=3):
    def radius_neighbors_with_angle(start_angle, end_angle):
        base_length = radius * (0.5 * base_point.distance_to(start_point) + 0.5 * base_point.distance_to(end_point))
        closest_neighbors = base_point.get_closest_points(
            Angle.get_points_within_angle(
                boundary.vertices, base_point, start_point, start_angle, end_angle
            ),
            exclusion=exclusion,
            max_dist=base_length
        )

        _angle = base_point.clockwise_angle(start_point, Vertex(base_point.x + 1, base_point.y))

        closest_neighbors.append(base_point +
                                Vertex(base_length * math.cos(_angle - (start_angle + end_angle) / 2),
                                       base_length * math.sin((_angle - (start_angle + end_angle) / 2))))
        return closest_neighbors

    angle = base_point.clockwise_angle(start_point, end_point)
    angles = [i * angle / N for i in range(N+1)]
    neighbors = [radius_neighbors_with_angle(angles[i-1], angles[i]) for i in range(1, N+1)]
    all_combinations = list(itertools.product(*reversed(neighbors)))
    return all_combinations


def write_samples(file_name, res, _type=1):
    if _type == 1:
        res['samples'] = [Vertex.flatten(s) for s in res['samples']]
        res['outputs'] = [Vertex.flatten(s) for s in res['outputs']]
    with open(file_name, 'w') as fw:
        json.dump(res, fw)


def write_inp(mesh, filename, format='inp'):
    if len(mesh.generated_quads) == 0:
        return

    nodes = list(mesh.original_vertices)  # boundary nodes (referenced by the B21 edge elements)
    for ele in mesh.generated_quads:
        nodes.extend([n for n in ele.vertices if n not in nodes])

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
