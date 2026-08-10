import json
import math
import itertools
from typing import Any

from general.components import *


class SampleExtractionMixin:
    # Provided by MeshGeneration; this mixin is only ever combined into it.
    boundary: Any
    get_quality: Any

    def get_nodes(self, root, exclusion, layer, path, paths, N):
        if root is None:
            return

        if len(path) < N:
            path.append(root)
        else:
            path[-layer-1] = root
        if layer == 0:
            paths.append([v for v in path])
            return
        else:
            nodes = [v for v in root.get_connected_vertices() if v not in exclusion and v not in path[:N-layer]]
            for i in range(len(nodes)):
                self.get_nodes(nodes[i], exclusion, layer-1, path, paths, N)

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

                        for p in rr:
                            _sample.extend([rp.distance_to(p) / (base_length * radius),
                                            rp.to_find_clockwise_angle(p, r_p) % round(2 * math.pi, 4)])
                        for p in mm:
                            _sample.extend([rp.distance_to(p) / (base_length * radius),
                                            rp.to_find_clockwise_angle(p, r_p) % round(2 * math.pi, 4)])
                        for p in reversed(ll):
                            _sample.extend([rp.distance_to(p) / (base_length * radius),
                                            rp.to_find_clockwise_angle(p, r_p) % round(2 * math.pi, 4)])
                        _target = [rp.distance_to(target) / (base_length * radius),
                                   rp.to_find_clockwise_angle(target, r_p) % round(2 * math.pi, 4)]

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

    def save_samples(self, file_name, res, _type=1):
        if _type == 1:
            res['samples'] = [Vertex.points_as_array(s) for s in res['samples']]
            res['outputs'] = [Vertex.points_as_array(s) for s in res['outputs']]
        with open(file_name, 'w') as fw:
            json.dump(res, fw)
