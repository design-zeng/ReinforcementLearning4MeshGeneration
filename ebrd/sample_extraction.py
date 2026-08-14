import json
import math
import itertools

from general.geometry import Vertex
from general.quality import edge_angle_quality


def extract_samples(mesh, quads, n_neighbor, n_radius, radius, metric=edge_angle_quality, quality_threshold=0.7):
    all_samples, outputs, types = [], [], []
    for element in quads:
        if metric(element) < quality_threshold:
            continue
        for i in range(4):
            reference = element.vertices[i]
            left = element.vertices[(i + 1) % 4]
            right = element.vertices[(i - 1)]
            target = element.vertices[(i - 2)]

            left_paths = []
            _collect_neighbor_paths(left, [reference, right], n_neighbor - 1, [], left_paths, n_neighbor)
            right_paths = []
            _collect_neighbor_paths(right, [reference, left], n_neighbor - 1, [], right_paths, n_neighbor)
            radius_neighbors = _get_radius_neighbors(mesh.vertices(), reference, left, right,
                                                     [reference, left, right, target], radius=radius, N=n_radius)

            combinations = itertools.product(
                [path for path in right_paths if len(path) == n_neighbor],
                radius_neighbors,
                [path for path in left_paths if len(path) == n_neighbor])

            for right_path, middle_points, left_path in combinations:
                if target in right_path and target in left_path:
                    continue
                base_length = (reference.distance_to(right_path[0]) +
                    sum([right_path[j].distance_to(right_path[j-1]) for j in range(1, len(right_path))]) +
                    sum([left_path[j].distance_to(left_path[j-1]) for j in range(1, len(left_path))]) +
                    reference.distance_to(left_path[0])) / (2 * n_neighbor)

                def encode(p):
                    return [reference.distance_to(p) / (base_length * radius),
                            reference.clockwise_angle(p, right) % round(2 * math.pi, 4)]

                sample = []
                for p in right_path:
                    sample.extend(encode(p))
                for p in middle_points:
                    sample.extend(encode(p))
                for p in reversed(left_path):
                    sample.extend(encode(p))

                types.append([1] if target in right_path else [0] if target in left_path else [0.5])
                all_samples.append(sample)
                outputs.append(encode(target))
    return all_samples, types, outputs


def _collect_neighbor_paths(root, exclusion, layer, path, paths, N):
    if root is None:
        return

    if len(path) < N:
        path.append(root)
    else:
        path[-layer-1] = root
    if layer == 0:
        paths.append(list(path))
        return
    for node in root.get_neighbors():
        if node not in exclusion and node not in path[:N-layer]:
            _collect_neighbor_paths(node, exclusion, layer-1, path, paths, N)


def _get_radius_neighbors(vertices, base_point, start_point, end_point, exclusion, radius, N=3):
    def points_within_angle(start_angle, end_angle):
        return [v for v in vertices
                if start_angle < base_point.clockwise_angle(start_point, v) < end_angle]

    def radius_neighbors_with_angle(start_angle, end_angle):
        base_length = radius * (0.5 * base_point.distance_to(start_point) + 0.5 * base_point.distance_to(end_point))
        closest_neighbors = base_point.get_closest_points(
            points_within_angle(start_angle, end_angle),
            exclusion=exclusion,
            max_dist=base_length
        )

        _angle = base_point.clockwise_angle(start_point, Vertex(base_point.x + 1, base_point.y))

        closest_neighbors.append(base_point +
                                Vertex(base_length * math.cos(_angle - (start_angle + end_angle) / 2),
                                       base_length * math.sin((_angle - (start_angle + end_angle) / 2))))
        return closest_neighbors

    angle = base_point.clockwise_angle(start_point, end_point)
    neighbors = [radius_neighbors_with_angle((i - 1) * angle / N, i * angle / N) for i in range(1, N+1)]
    return list(itertools.product(*reversed(neighbors)))


def write_samples(file_name, samples):
    with open(file_name, 'w') as fw:
        json.dump(samples, fw)
