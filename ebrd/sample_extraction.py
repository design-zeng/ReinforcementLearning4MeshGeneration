import json
import math
import itertools

from general.geometry import Vertex
from general.quality import QuadQuality


def extract_samples(mesh, dataset, n_neighbor, n_fan, fan_radius, metric=QuadQuality.edge_angle, quality_threshold=0.7):
    for element in mesh.generated_quads:
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
            fan_points = _fan_points(mesh.vertices(), reference, left, right,
                                     [reference, left, right, target], fan_radius=fan_radius, n_fan=n_fan)

            combinations = itertools.product(
                [path for path in right_paths if len(path) == n_neighbor],
                fan_points,
                [path for path in left_paths if len(path) == n_neighbor])

            for right_path, fan_path, left_path in combinations:
                if target in right_path and target in left_path:
                    continue
                base_length = (reference.distance_to(right_path[0]) +
                    sum([right_path[j].distance_to(right_path[j-1]) for j in range(1, len(right_path))]) +
                    sum([left_path[j].distance_to(left_path[j-1]) for j in range(1, len(left_path))]) +
                    reference.distance_to(left_path[0])) / (2 * n_neighbor)

                def encode(p):
                    return [reference.distance_to(p) / (base_length * fan_radius),
                            reference.clockwise_angle(p, right) % round(2 * math.pi, 4)]

                sample = []
                for p in right_path:
                    sample.extend(encode(p))
                for p in fan_path:
                    sample.extend(encode(p))
                for p in reversed(left_path):
                    sample.extend(encode(p))

                dataset['output_types'].append([1] if target in right_path else [0] if target in left_path else [0.5])
                dataset['samples'].append(sample)
                dataset['outputs'].append(encode(target))


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


def _fan_points(vertices, base_point, start_point, end_point, exclusion, fan_radius, n_fan=3):
    def points_within_angle(start_angle, end_angle):
        return [v for v in vertices
                if start_angle < base_point.clockwise_angle(start_point, v) < end_angle]

    def fan_points_with_angle(start_angle, end_angle):
        base_length = fan_radius * (0.5 * base_point.distance_to(start_point) + 0.5 * base_point.distance_to(end_point))
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
    fan_slices = [fan_points_with_angle((i - 1) * angle / n_fan, i * angle / n_fan) for i in range(1, n_fan+1)]
    return list(itertools.product(*reversed(fan_slices)))


def write_samples(file_name, samples):
    with open(file_name, 'w') as fw:
        json.dump(samples, fw)
