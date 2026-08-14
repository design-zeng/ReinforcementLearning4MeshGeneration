import math

from general.geometry import Segment, circle_line_intersection, clockwise_vertices


def smooth_mesh(mesh, boundary, vertices, lr_1=0.999, lr_2=0.999, iteration=400):
    def reposition_separated_components():
        for vertex in vertices:
            if vertex not in mesh.original_vertices:
                _smooth_vertex(mesh, boundary, vertex, lr_1, lr_2)
        return sum(v.x + v.y for v in vertices)

    _resolve_until_settled(reposition_separated_components, iteration)


def smooth_pave(mesh, boundary, vertices, current_boundary_vertices, iteration=400, interior=False):
    if not interior:
        _smooth_front(boundary, mesh.original_vertices)
    _smooth_fixed_vertices(mesh, [v for v in vertices if v not in current_boundary_vertices], iteration)


def _resolve_until_settled(run_pass, iteration):
    settled_sum = 0
    diffs = 100
    i_iteration = 0
    while diffs > 0.001 and i_iteration < iteration:
        i_iteration += 1
        new_sum = run_pass()
        diffs = math.fabs(new_sum - settled_sum)
        settled_sum = new_sum


def _first_inside(vertex, guard_a, guard_b, proposals):
    for candidate in proposals:
        ring = clockwise_vertices(vertex, vertex.get_neighbors())
        if _stays_inside_ring(vertex, candidate, ring, guard_a, guard_b):
            return candidate
    return vertex


def _smooth_fixed_vertices(mesh, vertices, iteration):
    def reposition_separated_components():
        checksum = 0
        for vertex in vertices:
            if vertex in mesh.original_vertices:
                continue
            neighbors = vertex.get_neighbors()
            if not neighbors:
                continue
            count = len(neighbors)
            vertex.x = sum(v.x + vertex.x for v in neighbors) / (2 * count)
            vertex.y = sum(v.y + vertex.y for v in neighbors) / (2 * count)
            checksum += vertex.x + vertex.y
        return checksum

    _resolve_until_settled(reposition_separated_components, iteration)


def _smooth_vertex(mesh, boundary, vertex, lr_1, lr_2):
    neighbors = vertex.get_neighbors()
    quad_count = len(mesh.find_related_quads(vertex))

    if quad_count == 1 and len(neighbors) == 2:
        origins = [v for v in neighbors[0].get_common_neighbors(neighbors[1]) if v is not vertex]
        origin = vertex.sorted_by_distance(origins)[0][0]
        front_distances = origin.sorted_by_distance(
            [v for v in boundary.vertices
             if v not in neighbors and v is not vertex])
        if not front_distances:
            return
        estimate = _estimate_4th_vertex(origin, neighbors[0], neighbors[1], suggest_dist=front_distances[0][1])
        vertex.x, vertex.y = estimate.x, estimate.y
        return

    if quad_count == 2:
        on_front = [v for v in neighbors if v in boundary.vertices]
        interior = [v for v in neighbors if v not in boundary.vertices]
        if len(interior) == 1 and len(on_front) == 2:
            inside = interior[0]
            origins1 = [v for v in inside.get_common_neighbors(on_front[0]) if v is not vertex]
            origins2 = [v for v in inside.get_common_neighbors(on_front[1]) if v is not vertex]
            e1 = _estimate_4th_vertex(vertex.sorted_by_distance(origins1)[0][0], on_front[0], inside, factor=0.7)
            e2 = _estimate_4th_vertex(vertex.sorted_by_distance(origins2)[0][0], on_front[1], inside, factor=0.7)
            vertex.x = (e1.x + e2.x) / 2
            vertex.y = (e1.y + e2.y) / 2
            return

    if quad_count not in (1, 2):
        if neighbors:
            count = len(neighbors)
            vertex.x = sum(v.x + vertex.x for v in neighbors) / (2 * count)
            vertex.y = sum(v.y + vertex.y for v in neighbors) / (2 * count)
        return

    lr = lr_1 if quad_count == 1 else lr_2
    for neighbor in neighbors:
        vertex.x = lr * vertex.x + (1 - lr) * neighbor.x
        vertex.y = lr * vertex.y + (1 - lr) * neighbor.y


def _smooth_front(boundary, original_vertices):
    n = len(boundary.vertices)
    for i in range(n):
        cur = boundary.vertices[i]
        if cur in original_vertices:
            continue
        left = boundary.vertices[(i + 1) % n]
        right = boundary.vertices[i - 1]
        v_angle = math.degrees(cur.clockwise_angle(left, right))

        if v_angle <= 90:
            new_v = _find_middle_vertex(cur, left, right, v_angle)
        elif v_angle <= 180:
            left_angle = boundary.compute_boundary_angle(left)
            right_angle = boundary.compute_boundary_angle(right)
            if right_angle < 45:
                new_v = _find_side_vertex(cur, left, right, boundary.vertices[i - 2], right_angle)
            elif left_angle < 45:
                new_v = _find_side_vertex(cur, right, left, boundary.vertices[(i + 2) % n], left_angle)
            else:
                new_v = _find_indention_vertex(boundary, cur, v_angle)
        elif v_angle <= 270:
            new_v = _find_indention_vertex(boundary, cur, v_angle)
        else:
            inner = _inner_vertex(boundary, cur, 45)
            cur.x, cur.y = inner.x, inner.y
            new_v = _find_indention_vertex(boundary, cur, v_angle)

        cur.x, cur.y = new_v.x, new_v.y


def _find_middle_vertex(vertex, left_v, right_v, v_angle):
    def proposals():
        target_angle = v_angle if v_angle >= 45 else 45
        while True:
            candidate = _middle_vertex(vertex, left_v, right_v, target_angle)
            if target_angle >= 135:
                return
            yield candidate
            target_angle += 5

    return _first_inside(vertex, left_v, right_v, proposals())


def _find_side_vertex(vertex, far_neighbor, near_neighbor, beyond_neighbor, corner_angle):
    dist = (vertex.distance_to(far_neighbor) + vertex.distance_to(near_neighbor) + near_neighbor.distance_to(beyond_neighbor)) / 3

    def proposals():
        target_angle = 45
        while True:
            candidate = _side_vertex(vertex, near_neighbor, beyond_neighbor, target_angle, dist)
            if target_angle <= corner_angle:
                return
            yield candidate
            target_angle -= 5

    return _first_inside(vertex, far_neighbor, near_neighbor, proposals())


def _find_indention_vertex(boundary, vertex, v_angle):
    index = boundary.vertices.index(vertex)
    n = len(boundary.vertices)
    left_v = boundary.vertices[(index + 1) % n]
    right_v = boundary.vertices[index - 1]
    dist = (vertex.distance_to(left_v) + vertex.distance_to(right_v)) / 2

    c_neighbors = vertex.get_closest_points(
        boundary.vertices,
        [boundary.vertices[index - 2], right_v, left_v, boundary.vertices[(index + 2) % n]], dist)
    neighbors = boundary.find_closest_segments(vertex, dist)
    if not (neighbors or c_neighbors):
        return vertex

    def proposals():
        times = 4
        while True:
            candidate = _indention_vertex(vertex, left_v, right_v, (360 - v_angle) / 2, dist / times)
            if times >= 10:
                return
            yield candidate
            times += 1

    return _first_inside(vertex, left_v, right_v, proposals())


def _inner_vertex(boundary, vertex, angle):
    index = boundary.vertices.index(vertex)
    left_v = boundary.vertices[(index + 1) % len(boundary.vertices)]
    right_v = boundary.vertices[index - 1]

    midpoint = (left_v + right_v) / 2
    depth = midpoint.distance_to(right_v) * math.tan(math.radians(angle))
    offset = vertex - midpoint
    scale = math.sqrt(depth ** 2 / (offset.x ** 2 + offset.y ** 2))
    return midpoint + offset * scale


def _estimate_4th_vertex(origin, left, right, factor=0.5, suggest_dist=None):
    distance = (origin.distance_to(left) + origin.distance_to(right)) * factor
    if suggest_dist is not None:
        distance = min(distance, 0.6 * suggest_dist)
    ray = Segment.get_ray_segment(Segment(origin, left), Segment(origin, right), distance)
    return ray.point2


def _middle_vertex(vertex, left_v, right_v, target_angle):
    midpoint = (left_v + right_v) / 2
    A = right_v.x - left_v.x
    B = right_v.y - left_v.y
    D = left_v.distance_to(midpoint) / math.tan(math.radians(target_angle / 2))

    V1, V2 = circle_line_intersection(midpoint.x, midpoint.y, A, B, 0, D)
    return V1 if V1.distance_to(vertex) < V2.distance_to(vertex) else V2


def _side_vertex(vertex, near_neighbor, beyond_neighbor, angle, dist):
    W = dist * near_neighbor.distance_to(beyond_neighbor) * math.cos(math.radians(angle))
    V1, V2 = circle_line_intersection(near_neighbor.x, near_neighbor.y, beyond_neighbor.x - near_neighbor.x, beyond_neighbor.y - near_neighbor.y, W, dist)
    return V1 if V1.distance_to(vertex) < V2.distance_to(vertex) else V2


def _indention_vertex(vertex, left_v, right_v, angle, dist):
    W = dist * vertex.distance_to(left_v) * math.cos(math.radians(angle))
    V1, V2 = circle_line_intersection(vertex.x, vertex.y, left_v.x - vertex.x, left_v.y - vertex.y, W, dist)
    return V1 if V1.clockwise_angle(left_v, right_v) < V2.clockwise_angle(left_v, right_v) else V2


def _stays_inside_ring(original, candidate, ring, left_v, right_v):
    return all(
        (candidate.clockwise_angle(ring[i], ring[i - 1]) < math.pi) ==
        (original.clockwise_angle(ring[i], ring[i - 1]) < math.pi)
        for i in range(len(ring))
        if not (left_v in (ring[i], ring[i - 1]) and right_v in (ring[i], ring[i - 1])))
