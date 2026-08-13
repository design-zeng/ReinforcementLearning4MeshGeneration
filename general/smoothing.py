import math

from general.geometry import Segment, clockwise_vertices, circle_line_intersection


# --- mesh relaxation (interior smoothing) ---

def smooth_pave(mesh, boundary, vertices, current_boundary_vertices, lr_1=None, lr_2=None, iteration=400, interior=False):
    if not interior:
        smooth_front(boundary, mesh.original_vertices)
    smooth_fixed_vertices(mesh, [v for v in vertices if v not in current_boundary_vertices], iteration)
    boundary.find_reference_candidates(target_angle=0)


def smooth_fixed_vertices(mesh, vertices, iteration):
    sum_coordinates = 0
    diffs = 100
    i_iteration = 0
    while diffs > 0.001 and i_iteration < iteration:
        i_iteration += 1
        new_sum_coordinates = 0
        for vertex in vertices:
            if vertex in mesh.original_vertices:
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


def smooth_mesh(mesh, boundary, vertices, lr_1=0.999, lr_2=0.999, iteration=400):
    sum_coordinates = 0
    diffs = 100
    i_iteration = 0
    while diffs > 0.001 and i_iteration < iteration:
        i_iteration += 1
        for vertex in vertices:
            if vertex not in mesh.original_vertices:
                smooth_vertex(mesh, boundary, vertex, lr_1, lr_2)
        new_sum_coordinates = sum(v.x + v.y for v in vertices)
        diffs = math.fabs(new_sum_coordinates - sum_coordinates)
        sum_coordinates = new_sum_coordinates

    boundary.find_reference_candidates(target_angle=0)


def smooth_vertex(mesh, boundary, vertex, lr_1, lr_2):
    connected = vertex.get_neighbors()
    n_meshes = len(mesh.find_related_quads(vertex))

    if n_meshes == 1 and len(connected) == 2:
        origins = [v for v in connected[0].get_common_neighbors(connected[1]) if v is not vertex]
        origin = vertex.sorted_by_distance(origins)[0][0]
        p_dist = origin.sorted_by_distance(
            [v for v in boundary.vertices
             if v not in connected and v is not vertex])
        if not p_dist:
            return
        estimate = estimate_4th_vertex(origin, connected[0], connected[1], suggest_dist=p_dist[0][1])
        vertex.x, vertex.y = estimate.x, estimate.y
        return

    if n_meshes == 2:
        on_front = [v for v in connected if v in boundary.vertices]
        interior = [v for v in connected if v not in boundary.vertices]
        if len(interior) == 1 and len(on_front) == 2:
            inside = interior[0]
            origins1 = [v for v in inside.get_common_neighbors(on_front[0]) if v is not vertex]
            origins2 = [v for v in inside.get_common_neighbors(on_front[1]) if v is not vertex]
            e1 = estimate_4th_vertex(vertex.sorted_by_distance(origins1)[0][0], on_front[0], inside, factor=0.7)
            e2 = estimate_4th_vertex(vertex.sorted_by_distance(origins2)[0][0], on_front[1], inside, factor=0.7)
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


def estimate_4th_vertex(origin, left, right, factor=0.5, suggest_dist=None):
    distance = (origin.distance_to(left) + origin.distance_to(right)) * factor
    if suggest_dist is not None:
        distance = min(distance, 0.6 * suggest_dist)
    s = Segment.get_ray_segment(Segment(origin, left), Segment(origin, right), distance)
    return s.point2


# --- advancing-front smoothing ---

def smooth_front(boundary, original_vertices):
    n = len(boundary.vertices)
    for i in range(n):
        cur = boundary.vertices[i]
        if cur in original_vertices:
            continue
        left = boundary.vertices[(i + 1) % n]
        right = boundary.vertices[i - 1]
        v_angle = math.degrees(cur.clockwise_angle(left, right))

        if v_angle <= 90:
            new_v = find_middle_vertex(cur, left, right, v_angle)
        elif v_angle <= 180:
            left_angle = boundary.compute_boundary_angle(left)
            right_angle = boundary.compute_boundary_angle(right)
            if right_angle < 45:
                new_v = find_side_vertex(cur, left, right, boundary.vertices[i - 2], right_angle)
            elif left_angle < 45:
                new_v = find_side_vertex(cur, right, left, boundary.vertices[(i + 2) % n], left_angle)
            else:
                new_v = find_indention_vertex(boundary, cur, v_angle)
        elif v_angle <= 270:
            new_v = find_indention_vertex(boundary, cur, v_angle)
        else:
            inner = inner_vertex(boundary, cur, 45)
            cur.x, cur.y = inner.x, inner.y
            new_v = find_indention_vertex(boundary, cur, v_angle)

        cur.x, cur.y = new_v.x, new_v.y


def find_middle_vertex(vertex, left_v, right_v, v_angle):
    target_angle = v_angle if v_angle >= 45 else 45
    while True:
        n_v = middle_vertex(vertex, left_v, right_v, target_angle)
        if target_angle >= 135:
            return vertex
        clockwise_boundary = clockwise_vertices(vertex, vertex.get_neighbors())
        if stays_inside_ring(vertex, n_v, clockwise_boundary, left_v, right_v):
            return n_v
        target_angle += 5


def find_side_vertex(vertex, _next_v, next_v, next_next_v, v_angle):
    dist = (vertex.distance_to(_next_v) + vertex.distance_to(next_v) + next_v.distance_to(next_next_v)) / 3
    target_angle = 45
    while True:
        n_v = side_vertex(vertex, next_v, next_next_v, target_angle, dist)
        if target_angle <= v_angle:
            return vertex
        clockwise_boundary = clockwise_vertices(vertex, vertex.get_neighbors())
        if stays_inside_ring(vertex, n_v, clockwise_boundary, _next_v, next_v):
            return n_v
        target_angle -= 5


def find_indention_vertex(boundary, vertex, v_angle):
    index = boundary.vertices.index(vertex)
    n = len(boundary.vertices)
    left_v = boundary.vertices[(index + 1) % n]
    right_v = boundary.vertices[index - 1]
    dist = (vertex.distance_to(left_v) + vertex.distance_to(right_v)) / 2

    c_neighbors = vertex.get_closest_points(
        boundary.vertices,
        [boundary.vertices[index - 2], right_v, left_v, boundary.vertices[(index + 2) % n]], dist)
    neighbors = find_closest_segments(boundary, vertex, dist)
    if not (neighbors or c_neighbors):
        return vertex

    times = 4
    while True:
        n_v = indention_vertex(vertex, left_v, right_v, (360 - v_angle) / 2, dist / times)
        if times >= 10:
            return vertex
        clockwise_boundary = clockwise_vertices(vertex, vertex.get_neighbors())
        if stays_inside_ring(vertex, n_v, clockwise_boundary, left_v, right_v):
            return n_v
        times += 1


def inner_vertex(boundary, vertex, angle):
    index = boundary.vertices.index(vertex)
    left_v = boundary.vertices[(index + 1) % len(boundary.vertices)]
    right_v = boundary.vertices[index - 1]

    m_v = (left_v + right_v) / 2
    d = m_v.distance_to(right_v) * math.tan(math.radians(angle))
    diff = vertex - m_v
    s = math.sqrt(d ** 2 / (diff.x ** 2 + diff.y ** 2))
    return m_v + diff * s


def find_closest_segments(boundary, vertex, dist):
    closest = []
    for i in range(len(boundary.vertices)):
        prev, curr = boundary.vertices[i - 1], boundary.vertices[i]
        if vertex in (prev, curr):
            continue
        seg = Segment(prev, curr)
        _, perp_dist, inner = seg.perpendicular_point(vertex)
        if inner and perp_dist <= dist:
            closest.append(seg)
    return closest


def middle_vertex(vertex, left_v, right_v, target_angle):
    m_v = (left_v + right_v) / 2
    A = right_v.x - left_v.x
    B = right_v.y - left_v.y
    D = left_v.distance_to(m_v) / math.tan(math.radians(target_angle / 2))

    V1, V2 = circle_line_intersection(m_v.x, m_v.y, A, B, 0, D)
    return V1 if V1.distance_to(vertex) < V2.distance_to(vertex) else V2


def side_vertex(vertex, next_v, next_next_v, angle, dist):
    W = dist * next_v.distance_to(next_next_v) * math.cos(math.radians(angle))
    V1, V2 = circle_line_intersection(next_v.x, next_v.y, next_next_v.x - next_v.x, next_next_v.y - next_v.y, W, dist)
    return V1 if V1.distance_to(vertex) < V2.distance_to(vertex) else V2


def indention_vertex(vertex, left_v, right_v, angle, dist):
    W = dist * vertex.distance_to(left_v) * math.cos(math.radians(angle))
    V1, V2 = circle_line_intersection(vertex.x, vertex.y, left_v.x - vertex.x, left_v.y - vertex.y, W, dist)
    return V1 if V1.clockwise_angle(left_v, right_v) < V2.clockwise_angle(left_v, right_v) else V2


def stays_inside_ring(original, candidate, ring, left_v, right_v):
    for i in range(len(ring)):
        if left_v in [ring[i], ring[i - 1]] and right_v in [ring[i], ring[i - 1]]:
            continue
        if (candidate.clockwise_angle(ring[i], ring[i - 1]) < math.pi) != \
                (original.clockwise_angle(ring[i], ring[i - 1]) < math.pi):
            return False
    return True
