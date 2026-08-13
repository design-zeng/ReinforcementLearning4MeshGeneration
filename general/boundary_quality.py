import math

from general.geometry import Segment


def sort_segments_by_length(boundary, reverse=False):
    return sorted(((seg, seg.length()) for seg in boundary.own_segments()),
                  key=lambda x: x[1], reverse=reverse)


def estimate_area_range(boundary):
    lengths = [length for _, length in sort_segments_by_length(boundary)]
    L = sum(lengths) / len(lengths)
    max_L = min(lengths[-2], 2 * L)
    min_L = min(L / math.sqrt(2), lengths[1])
    return min_L, (max_L + 3 * min_L) / 4


def compute_boundary_quality(boundary, added_vertex):
    v = boundary.vertices
    n = len(v)
    index = v.index(added_vertex)

    angles = [a for i in (1, -1)
              if (a := v[(index + i) % n].clockwise_angle(v[(index + i + 1) % n], v[index + i - 1])) < math.pi / 3]
    q1 = 3 * min(angles) / math.pi if angles else 1

    dist = added_vertex.distance_to(v[(index + 1) % n]) + added_vertex.distance_to(v[index - 1])
    excluded = {v[index], v[(index + 1) % n], v[(index + 2) % n], v[index - 1], v[index - 2]}
    close_vs = []
    for i, vv in enumerate(v):
        if vv not in excluded and added_vertex.distance_to(vv) < dist and i - 1 not in close_vs:
            close_vs.append(i)
    dists = [Segment(v[(i + 1) % n], v[i]).distance(added_vertex) for i in close_vs]

    target_len = dist / 2
    ring = [(index + i) % n for i in range(-2, 3)]
    mean_dist = sum(v[ring[i]].distance_to(v[ring[i + 1]]) for i in range(len(ring) - 1)) / (len(ring) - 1)
    smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)

    if dists and min(dists) < 0.5 * dist:
        q2 = min(dists) / (0.5 * dist)
    else:
        q2 = 1

    return math.pow(smoothness * q1 * q2, 1 / 3)


def compute_element_boundary_quality(boundary, element):
    v = boundary.vertices
    n = len(v)
    new_vs = [x for x in element.vertices
              if len(x.get_neighbors()) == 2 and x in v]

    if new_vs:
        return compute_boundary_quality(boundary, new_vs[0])

    target_vs = [x for x in element.vertices if x in v]
    if not target_vs:
        return 1

    angles = []
    for x in target_vs:
        index = v.index(x)
        angle = x.clockwise_angle(v[(index + 1) % n], v[index - 1])
        if angle < math.pi / 3:
            angles.append(angle)
    index = min(v.index(target_vs[0]), v.index(target_vs[1]))

    target_len = target_vs[0].distance_to(target_vs[1])
    ring = [(index + i) % n for i in range(-2, 4)]
    mean_dist = sum(v[ring[i]].distance_to(v[ring[i + 1]]) for i in range(len(ring) - 1)) / (len(ring) - 1)

    smoothness = min(mean_dist, target_len) / max(mean_dist, target_len)
    angle_quality = 3 * min(angles) / math.pi if angles else 1
    return math.pow(angle_quality * smoothness, 1 / 2)
