import time
import math
import json
import random
from pathlib import Path

import numpy as np
from numpy.random import uniform
from multiprocessing import Process, Manager

from general.components import Segment, Vertex, Mesh


base_path = Path(__file__).parent.parent
augmentation_path = base_path / "ebrd" / "output" / "data_augmentation"

ACTION_TYPES = [0, 0.5, 1]


def sample_state(angle, action_type):
    obs = []
    if action_type == ACTION_TYPES[1]:
        obs = [uniform(0.01, 0.45), 0,
               uniform(0.1, 0.77), uniform(-0.5, 0.9) * math.pi,
               uniform(0.33, 1), uniform(-0.5, 0.9) * math.pi,
               uniform(0.15, 1), uniform(0, 1 / 3 * angle),
               uniform(0.15, 1), uniform(1 / 3 * angle, 2 / 3 * angle),
               uniform(0.15, 1), uniform(2 / 3 * angle, angle),
               uniform(0.33, 1), uniform(0.1, 1.5) * math.pi,
               uniform(0.1, 0.77), uniform(0.01, 1.5) * math.pi,
               uniform(0.01, 0.45), angle,
               ]
    elif action_type == ACTION_TYPES[0]:
        obs = [uniform(0.01, 0.45), 0,
               uniform(0.1, 0.77), uniform(-0.5 * math.pi, angle),
               uniform(0.33, 1), uniform(-0.5, 0.9) * math.pi,
               uniform(0.15, 1), uniform(0, 1 / 3 * angle),
               uniform(0.15, 1), uniform(1 / 3 * angle, 2 / 3 * angle),
               uniform(0.15, 1), uniform(2 / 3 * angle, angle),
               uniform(0.33, 1), uniform(0.1, 1.5) * math.pi,
               uniform(0.1, 0.77), uniform(0.01, angle),
               uniform(0.01, 0.45), angle,
               ]
    elif action_type == ACTION_TYPES[2]:
        obs = [uniform(0.01, 0.45), 0,
               uniform(0.1, 0.77), uniform(0.01, angle),
               uniform(0.33, 1), uniform(-0.5, 0.9) * math.pi,
               uniform(0.15, 1), uniform(0, 1 / 3 * angle),
               uniform(0.15, 1), uniform(1 / 3 * angle, 2 / 3 * angle),
               uniform(0.15, 1), uniform(2 / 3 * angle, angle),
               uniform(0.33, 1), uniform(0.1, 1.5) * math.pi,
               uniform(0.1, 0.77), uniform(0.01, 1.5) * math.pi,
               uniform(0.01, 0.45), angle,
               ]
    return obs


def sample_action(obs, angle, action_type, n_candidates):
    acts = []
    if action_type == ACTION_TYPES[1]:
        while len(acts) < n_candidates:
            act = [ACTION_TYPES[1], uniform(0.01, 0.56), uniform(0.01, angle)]
            acts.append(act)

    elif action_type == ACTION_TYPES[0]:
        acts.append([ACTION_TYPES[0], obs[-4], obs[-3]])
    else:
        acts.append([ACTION_TYPES[2], obs[2], obs[3]])
    return acts


def get_angle_distribution(N):
    m = 11
    keys = ['0.4', '0.6', '0.8', '1', '1.2', '1.4', '1.6', '1.8', '2', '2.2', '2.4']
    return {k: [math.ceil(N / m), 0] for k in keys}


def sample_all_types(n, threshold, file_name=None):
    observations, actions = [], []
    start_time = time.time()

    t1 = sample_for_type(0.5 * n, threshold, ACTION_TYPES[1])
    t2 = sample_for_type(0.25 * n, threshold, ACTION_TYPES[0])
    t3 = sample_for_type(0.25 * n, threshold, ACTION_TYPES[2])

    observations.extend(t1[0])
    actions.extend(t1[1])

    observations.extend(t2[0])
    actions.extend(t2[1])

    observations.extend(t3[0])
    actions.extend(t3[1])

    observations, actions = np.asarray(observations), np.asarray(actions)
    if actions.size == 0:
        # No samples cleared the quality threshold; return an empty dataset
        # instead of indexing an empty array.
        training_data = {'samples': [], 'output_types': [], 'outputs': []}
    else:
        training_data = {'samples': observations.tolist(),
                         'output_types': actions[:, 0:1].tolist(),
                         'outputs': actions[:, 1:].tolist()}

    print("Sampling completed in: ", time.time() - start_time, 's')

    if file_name is not None:
        with open(file_name, 'w+') as f:
            json.dump(training_data, f)
    return training_data


def sample_for_type(n, threshold, action_type):
    observations, actions = [], []
    i = 0
    _i = 0
    angle_distribution = get_angle_distribution(n)

    for k, v in angle_distribution.items():
        while angle_distribution[k][1] < v[0]:
            angle = uniform(float(k) - 0.2, float(k))
            obs = sample_state(angle, action_type)

            if is_degenerate_state(obs):
                continue

            sample_acts = sample_action(obs, angle, action_type, n_candidates=20)  # n_candidates only matters for type 0.5
            for act in sample_acts:
                ele_quality, boundary_quality = compute_quality(obs, act)

                lam = 0.618
                quality = (1 - lam) * ele_quality + lam * boundary_quality

                max_ele_quality, max_boun_quality = estimate_max_quality(obs, act)
                max_quality = (1 - lam) * max_ele_quality + lam * max_boun_quality

                _i += 1

                if quality / max_quality >= threshold and ele_quality / max_ele_quality >= threshold:
                    angle_distribution[k][1] += 1
                    if angle_distribution[k][1] >= angle_distribution[k][0]:
                        break

                    observations.append(obs)
                    actions.append(act)
                    if i % 1000 == 0:
                        print(f"Sampling length: {i} out of {n} for type {action_type} with {_i} tries")
                    i += 1
    print(f"Sampling length: {i} out of {n} for type {action_type} with {_i} tries")
    return observations, actions


def is_degenerate_state(obs):
    points = [Vertex(obs[i] * math.cos(obs[i+1]), obs[i] * math.sin(obs[i+1])) for i in range(0, len(obs) - 1, 2)]

    neighbor_points = [points[-3], points[-2], points[-1], Vertex(0, 0), points[0], points[1], points[2]]
    radius_points = [points[3], points[4], points[5]]

    for i in range(1, len(neighbor_points)):
        for j in range(i + 2, len(neighbor_points)):
            s1 = Segment(neighbor_points[i], neighbor_points[i - 1])
            s2 = Segment(neighbor_points[j], neighbor_points[j - 1])
            if s1.is_cross(s2):
                return True

        for k in range(1, len(radius_points)):
            s1 = Segment(neighbor_points[i], neighbor_points[i - 1])
            s2 = Segment(radius_points[k], radius_points[k - 1])
            if s1.is_cross(s2):
                return True

    lengths = [neighbor_points[i].distance_to(neighbor_points[i-1]) for i in range(1, len(neighbor_points))]
    if max(lengths) / min(lengths) >= 5:
        return True
    return False


def decode_geometry(obs, action):
    reference_vertex = Vertex(0, 0)

    L = len(obs)
    N = int((L-6) / 2)
    left = [-i-2 for i in reversed(range(0, N, 2))]
    right = [i for i in range(0, N, 2)]
    all = left + right

    neighbor_points = [Vertex(obs[j] * math.cos(obs[j + 1]), obs[j] * math.sin(obs[j + 1]))
                       for j in all]
    neighbor_points.insert(int(N/2), reference_vertex)

    medium = [N + i for i in range(0, 6, 2)]

    radius_points = [Vertex(obs[j] * math.cos(obs[j + 1]), obs[j] * math.sin(obs[j + 1]))
                     for j in medium]

    # base_length = 1: the synthetic obs are already in normalized units. (In the paper,
    # real extracted coordinates are scaled by 1/|P0 P_r1| -- FreeMesh-S Eq. 13.)
    base_length = 1

    rule_type, new_point = action[0], Vertex(base_length * action[1] * math.cos(action[2]),
                                             base_length * action[1] * math.sin(action[2]))

    return neighbor_points, radius_points, rule_type, new_point


def estimate_max_quality(obs, action, method=3):
    """Ideal (max-achievable) element/boundary quality for this neighborhood, used to
    normalize the measured quality in sample_for_type. Element quality is the geometric
    mean sqrt(q_a * q_e) (FreeMesh-S Eq. 14).

    method 2 -> angle quality q_a is Eq. 16: (prod (1 - |a-90|/90))^(1/4) over the 4
                element angles (the 3 unknown angles assumed equal at rem_angle).
    method 3 -> (current live default) ad-hoc min/max angle ratio (NOT Eq. 16); edge
                quality q_e uses Eq. 15's 4th root.
    Neither is fully Eq.15+Eq.16 faithful: method 2 has Eq.16's angle but a 1/2-power
    edge term, method 3 has Eq.15's edge but a non-paper angle. A perpendicular-bisector
    "ideal element" variant was removed -- it matches no equation in any source paper.
    """
    neighbor_points, radius_points, rule_type, new_point = decode_geometry(obs, action)
    index = int(len(neighbor_points) / 2)

    left_v = neighbor_points[index - 1]
    right_v = neighbor_points[index + 1]

    angle = neighbor_points[index].to_find_clockwise_angle(left_v, right_v)
    rem_angle = (2 * math.pi - angle) / 3
    l1, l2 = neighbor_points[index].distance_to(left_v), neighbor_points[index].distance_to(right_v)
    edge_product = 1
    area = l2 * l1 * math.sin(angle)
    for edge in [l1, l2]:
        edge_product *= math.pow(edge / math.sqrt(area), 1 if math.sqrt(area) - edge > 0 else -1)

    segments = [Segment(radius_points[0], radius_points[1]), Segment(radius_points[1], radius_points[2])]
    dists = [s.distance(neighbor_points[index]) for s in segments]
    boundary_quality = math.sqrt(min(1, max(dists) / ((math.sqrt(2) / 2 + 0.5) * (l1 + l2))))

    if method == 2:
        # angle quality q_a -- FreeMesh-S Eq. 16
        angle_product = 1
        for _angle in [angle, rem_angle, rem_angle, rem_angle]:
            angle_product *= 1 - (math.fabs(math.degrees(_angle) - 90) / 90)
        angle_quality = 0 if angle_product < 0 else math.pow(angle_product, 1 / 4)
        edge_quality = math.pow(edge_product, 1 / 2)
    else:  # method == 3 (live): edge quality is Eq. 15's 4th root; angle is ad-hoc
        angle_quality = min(angle, rem_angle) / max(angle, rem_angle)
        edge_quality = math.pow(edge_product, 1 / 4)

    return math.sqrt(angle_quality * edge_quality), boundary_quality


def compute_quality(obs, action):
    '''
    Compute the quality for the sample
    :param obs: observation (local boundary neighborhood)
    :param action: [type, radius, angle]
    :return: the integration of element quality and boundary quality
    '''
    neighbor_points, radius_points, rule_type, new_point = decode_geometry(obs, action)

    index = int(len(neighbor_points) / 2)
    if rule_type == ACTION_TYPES[0]:
        quad = Mesh([  # vertices in clockwise direction
            neighbor_points[index - 2],
            neighbor_points[(index + 1) % len(neighbor_points)],
            neighbor_points[index],
            neighbor_points[index - 1]
        ])
    elif rule_type == ACTION_TYPES[2]:
        quad = Mesh([
            neighbor_points[index - 1],
            neighbor_points[(index + 2) % len(neighbor_points)],
            neighbor_points[
                (index + 1) % len(neighbor_points)],
            neighbor_points[index]
        ])
    else:
        quad = Mesh([new_point,
                     neighbor_points[
                         (index + 1) % len(neighbor_points)],
                     neighbor_points[index],
                     neighbor_points[index - 1]
                     ])

    if quad.is_valid(0) and \
            not quad_crosses_boundary(quad, neighbor_points, radius_points) and \
            not quad_encloses_point(quad, radius_points + [neighbor_points[0], neighbor_points[-1]]):
        element_quality = quad.get_quality(type='strong')
        boundary_quality = compute_boundary_quality(quad, neighbor_points, radius_points)
        return element_quality, boundary_quality
    else:
        return 0, 0


def quad_crosses_boundary(quad, neighbor_points, radius_points):
    for i in range(4):
        s1 = Segment(quad.vertices[i], quad.vertices[i - 1])
        for j in range(1, len(neighbor_points)):
            s = Segment(neighbor_points[j], neighbor_points[j-1])
            if quad.vertices[i] not in [neighbor_points[j], neighbor_points[j-1]] and \
                quad.vertices[i-1] not in [neighbor_points[j], neighbor_points[j-1]]:
                if s1.is_cross(s):
                    return True

        for j in range(1, len(radius_points)):
            s = Segment(radius_points[j], radius_points[j-1])
            if quad.vertices[i] not in [radius_points[j], radius_points[j-1]] and \
                quad.vertices[i-1] not in [radius_points[j], radius_points[j-1]]:
                if s1.is_cross(s):
                    return True
    return False


def quad_encloses_point(quad, radius_points):
    centroid = quad.get_centriod()

    for v in radius_points:
        s1 = Segment(v, centroid)
        crossed = False
        for i in range(4):
            s = Segment(quad.vertices[i], quad.vertices[i - 1])
            if s1.is_cross(s):
                crossed = True
                break
        if not crossed:
            return True
    return False


def compute_boundary_quality(quad, neighbor_points, radius_points):
    new_v = [(i, v) for i, v in enumerate(quad.vertices) if v not in neighbor_points]

    if len(new_v):
        # Has newly generated vertex
        id, new_v = new_v[0]
        # Angle quality
        left_v = quad.vertices[id-1]
        right_v = quad.vertices[(id + 1) % 4]
        left_angle = left_v.to_find_clockwise_angle(neighbor_points[neighbor_points.index(left_v) - 1], new_v)
        right_angle = right_v.to_find_clockwise_angle(new_v, neighbor_points[neighbor_points.index(right_v) + 1])
        angles = []
        if left_angle < math.pi / 2:
            angles.append(left_angle)
        if right_angle < math.pi / 2:
            angles.append(right_angle)
        q1 = 2 * min(angles) / math.pi if len(angles) else 1

        # Distance quality
        dist = new_v.distance_to(left_v) + new_v.distance_to(right_v)
        if len(neighbor_points) > 5:
            segments = [Segment(neighbor_points[0], neighbor_points[1]), Segment(neighbor_points[-1], neighbor_points[-2]),
                        Segment(radius_points[0], radius_points[1]), Segment(radius_points[1], radius_points[2])]
        else:
            segments = [Segment(radius_points[0], radius_points[1]), Segment(radius_points[1], radius_points[2])]
        dists = []
        for s in segments:
            dists.append(s.distance(new_v))

        if len(dists):
            m_d = min(dists)
            q2 = m_d / dist if m_d < dist else 1
        else:
            q2 = 1

        # compute smoothness
        targt_len = dist / 2
        k = int(len(neighbor_points) / 2)
        _dists = [neighbor_points[k - 2], neighbor_points[k - 1], new_v, neighbor_points[k + 1], neighbor_points[k + 2]]
        mean_dist = sum([_dists[i].distance_to(_dists[i - 1]) for i in range(1, len(_dists))]) / (len(_dists) - 1)

        smoothness = min(mean_dist, targt_len) / max(mean_dist, targt_len)
        return math.pow(q1 * q2 * smoothness, 1 / 3)
    else:
        indices = sorted([neighbor_points.index(v) for v in quad.vertices])
        left_ind, right_ind = indices[0], indices[-1]

        angles = []

        if left_ind == 0:
            right_angle = neighbor_points[right_ind].to_find_clockwise_angle(neighbor_points[left_ind],
                                                                       neighbor_points[right_ind + 1])
            if right_angle < math.pi / 2:
                angles.append(right_angle)

            _dists = [neighbor_points[left_ind], neighbor_points[right_ind], neighbor_points[right_ind + 1]]
        elif right_ind == len(neighbor_points) - 1:
            left_angle = neighbor_points[left_ind].to_find_clockwise_angle(neighbor_points[left_ind - 1],
                                                                     neighbor_points[right_ind])

            if left_angle < math.pi / 2:
                angles.append(left_angle)
            _dists = [neighbor_points[left_ind - 1], neighbor_points[left_ind], neighbor_points[right_ind]]
        else:
            left_angle = neighbor_points[left_ind].to_find_clockwise_angle(neighbor_points[left_ind - 1],
                                                                     neighbor_points[right_ind])
            right_angle = neighbor_points[right_ind].to_find_clockwise_angle(neighbor_points[left_ind],
                                                                       neighbor_points[right_ind + 1])
            if left_angle < math.pi / 2:
                angles.append(left_angle)
            if right_angle < math.pi / 2:
                angles.append(right_angle)
            _dists = [neighbor_points[left_ind - 1], neighbor_points[left_ind], neighbor_points[right_ind], neighbor_points[right_ind + 1]]

        # angle quality
        angle_quality = 2 * min(angles) / math.pi if len(angles) else 1

        # compute smoothness
        targt_len = neighbor_points[left_ind].distance_to(neighbor_points[right_ind])
        mean_dist = sum([_dists[i].distance_to(_dists[i - 1]) for i in range(1, len(_dists))]) / (
                len(_dists) - 1)

        smoothness = min(mean_dist, targt_len) / max(mean_dist, targt_len)
        boundary_quality = math.sqrt(angle_quality * smoothness)
        return boundary_quality


def load_samples(filename):
    with open(filename, 'r') as fr:
        data = json.load(fr)

    return data


def sampling_worker(i, n, threshold, data):
    random.seed(999 + i)      # per-worker seed for reproducible sampling
    np.random.seed(999 + i)
    data[i] = sample_all_types(n, threshold)


def sampling_main(pool, N, threshold, file_name=None):
    manager = Manager()
    return_dict = manager.dict()

    jobs = []
    for i in range(pool):
        p = Process(target=sampling_worker,
                    args=(i, N / pool, threshold, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    sample_data = {'samples': [],
                   'output_types': [],
                   'outputs': []}

    for p in return_dict.values():
        for k, v in p.items():
            sample_data[k].extend(v)

    if file_name is not None:
        with open(file_name, 'w+') as f:
            json.dump(sample_data, f)

    return sample_data


if __name__ == '__main__':
    # Experience Extraction (FreeMesh-S): sample FNN training data in parallel.
    #   pool      = number of worker processes
    #   N         = total number of samples to generate
    #   threshold = minimum normalized element quality to keep a sample
    out_file = augmentation_path / "1" / "training_samples.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sampling_main(10, 40000, 0.7, file_name=str(out_file))

    # Inspect the generated dataset with ebrd.data_augmentation_plotting.scatter_plot.
