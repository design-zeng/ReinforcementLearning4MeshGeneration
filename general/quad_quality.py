import math


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
