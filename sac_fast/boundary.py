from __future__ import annotations
from typing import Literal
import math

import numpy as np
import numpy.typing as npt

from sac_fast.polygon import Polygon
from sac_fast.geometry_lib import segment_intersect, rotation_matrix, vector_angle, polar_to_Cartesian


class Boundary(Polygon):
    def __init__(self, initial_boundary: npt.NDArray[np.floating], v=1, k=4, M_angle=math.pi/3, alpha=1, beta=2, n_rv=2, g=3):
        super().__init__(initial_boundary)

        self.initial_area = self.get_area()

        self.v = v
        self.k = k
        self.M_angle = M_angle
        self.alpha = alpha
        self.beta = beta
        self.n_rv = n_rv
        self.g = g

    def get_sharpest_vertex_index(self) -> np.intp:
        angle_sum = np.zeros(self.vertices.shape[0])
        for j in range(1, self.n_rv + 1):
            angle_sum += self.get_angles(j)
        # finding smallest average is equivalent to smallest sum, no need to divide for average
        index = np.argmin(angle_sum)
        return index

    def get_average_surrounding_lengths(self, vertex_index: np.intp) -> float:
        N = self.vertices.shape[0]
        lengths = self.get_lengths()
        indices = np.mod(np.arange(vertex_index - self.n_rv, vertex_index + self.n_rv), N)
        selected_lengths = lengths[indices]
        return float(np.mean(selected_lengths))

    def get_coordinate_space_radius(self, vertex_index: np.intp) -> float:
        return self.alpha * self.get_average_surrounding_lengths(vertex_index)

    def get_vision_space_L_r(self, vertex_index: np.intp) -> float:
        return self.beta * self.get_average_surrounding_lengths(vertex_index)

    def get_rotation_matrix_at(self, vertex_index: np.intp) -> npt.NDArray[np.floating]:
        right_vector = self.v_at(vertex_index + 1) - self.v_at(vertex_index)
        right_angle = vector_angle(right_vector)
        rotation_matrix_at = rotation_matrix(-right_angle)
        return rotation_matrix_at

    # inverse transformation of vertex relative to the vertex at vertex_index
    def get_inverse_transformed_vertex(self, local_vertex: npt.NDArray[np.floating], vertex_index: np.intp) -> npt.NDArray[np.floating]:
        transform_matrix = self.get_rotation_matrix_at(vertex_index)
        inverse_matrix = transform_matrix.T #inverse of orthogonal transformation is transpose
        return inverse_matrix @ local_vertex + self.v_at(vertex_index)

    def get_transformed_vertex(self, local_vertex: npt.NDArray[np.floating], vertex_index: np.intp) -> npt.NDArray[np.floating]:
        transform_matrix = self.get_rotation_matrix_at(vertex_index)
        return transform_matrix @ (local_vertex - self.v_at(vertex_index))

    # relative to the vertex at vertex_index
    def get_transformed_boundary(self, vertex_index: np.intp) -> npt.NDArray[np.floating]:
        rotation_matrix = self.get_rotation_matrix_at(vertex_index)
        boundary_transformed = (self.vertices - self.v_at(vertex_index)) @ rotation_matrix.T
        return boundary_transformed

    def get_opposite_neighbors_local(self, vertex_index: np.intp) -> npt.NDArray[np.floating]:
        N = self.vertices.shape[0]
        Lr = self.get_vision_space_L_r(vertex_index)

        local_boundary = self.get_transformed_boundary(vertex_index)

        left_vertex = local_boundary[(vertex_index - 1) % N]
        vertex_angle = vector_angle(left_vertex)

        angles_to_origin = np.arctan2(local_boundary[:, 1], local_boundary[:, 0]) % (2 * math.pi)
        distances_to_origin = np.sqrt(np.sum(np.square(local_boundary), axis=1))

        opposite_neighbor_vertices_local = np.zeros((self.g, 2))

        for i in range(self.g):
            low_angle = vertex_angle * i / self.g
            high_angle = vertex_angle * (i + 1) / self.g
            mid_angle = (low_angle + high_angle) / 2

            # closest vertex
            closest_vertex = None
            mask = (angles_to_origin > low_angle) & (angles_to_origin < high_angle) & (distances_to_origin < Lr)
            indices = np.where(mask)[0]
            if indices.shape[0] > 0:
                closest_vertex = local_boundary[indices[np.argmin(distances_to_origin[indices])]]

            # intersection between bisector and vision sector bound
            vision_boundary_vertex = polar_to_Cartesian(mid_angle, Lr)

            # from Claude, for closest edge/bisector intersection
            d = polar_to_Cartesian(mid_angle, 1)
            E = np.roll(local_boundary, -1, axis=0) - local_boundary
            det = E[:, 0] * d[1] - E[:, 1] * d[0]
            safe = np.where(np.abs(det) > 1e-12, det, 1.0)
            t = (E[:, 0] * local_boundary[:, 1] - E[:, 1] * local_boundary[:, 0]) / safe
            s = (d[0] * local_boundary[:, 1] - d[1] * local_boundary[:, 0]) / safe
            hit = (np.abs(det) > 1e-12) & (s >= 0) & (s <= 1) & (t > 1e-9) & (t <= Lr)
            edge_intersection_vertex = d * t[hit].min() if np.any(hit) else None

            # choice
            if closest_vertex is not None:
                opposite_neighbor_vertices_local[i] = closest_vertex
            elif edge_intersection_vertex is not None:
                opposite_neighbor_vertices_local[i] = edge_intersection_vertex
            else:
                opposite_neighbor_vertices_local[i] = vision_boundary_vertex

        return opposite_neighbor_vertices_local

    """
    Three main functions here:
    - state
    - reward
    - update (D = D - s)
    """

    def get_state(self) -> npt.NDArray[np.floating]:
        N = self.vertices.shape[0]
        sharpest_index = self.get_sharpest_vertex_index()
        local_boundary = self.get_transformed_boundary(sharpest_index)
        radius = self.get_vision_space_L_r(sharpest_index)

        #neighbors includes self
        neighbors = np.array([local_boundary[(sharpest_index + i) % N] for i in range(-self.n_rv, self.n_rv + 1)]) / max(radius, 1e-6)
        opposite_neighbors = self.get_opposite_neighbors_local(sharpest_index) / max(radius, 1e-6)
        area_ratio = np.array([self.get_area() / self.initial_area])
        return np.concatenate([neighbors.flatten(), opposite_neighbors.flatten(), area_ratio]).astype(np.float32)

    def get_reward(self, action_type: Literal[-1, 0, 1], polar_pair=(None, None)) -> tuple[float, bool]:
        if action_type == 0:
            return self.type0_r(polar_pair)
        if action_type == -1:
            return self.type1_r("left")
        if action_type == 1:
            return self.type1_r("right")

    def update_boundary(self, action_type: Literal[-1, 0, 1], polar_pair=(None, None)):
        if action_type == 0:
            self.type0_update_boundary_with_polar_pair(polar_pair)
        elif action_type == -1:
            self.type1_update_boundary("left")
        elif action_type == 1:
            self.type1_update_boundary("right")

    def u(self, quad: Polygon) -> float:
        quad_area = quad.get_area()

        e_min = self.get_shortest_length()
        e_max = self.get_longest_length()
        A_min = self.v * (e_min ** 2)
        A_max = self.v * ((e_max - e_min) / self.k + e_min) ** 2

        if quad_area < A_min:
            return -1
        elif quad_area >= A_max:
            return 0
        else:
            return (quad_area - A_min) / (A_max - A_min)

    def n_b_type0(self, new_vertex: npt.NDArray[np.floating], vertex_index: np.intp) -> float:
        N = self.vertices.shape[0]

        left_vertex = self.v_at(vertex_index - 1)
        right_vertex = self.v_at(vertex_index + 1)
        left_left_vertex = self.v_at(vertex_index - 2)
        right_right_vertex = self.v_at(vertex_index + 2)

        vector_left_left = left_left_vertex - left_vertex
        vector_right_right = right_right_vertex - right_vertex
        vector_left = left_vertex - new_vertex
        vector_right = right_vertex - new_vertex

        angle_left_left = vector_angle(vector_left_left)
        angle_left = vector_angle(vector_left)
        angle_right_right = vector_angle(vector_right_right)
        angle_right = vector_angle(vector_right)

        a1 = (angle_left_left - angle_left) % (2 * math.pi)
        a2 = (angle_right - angle_right_right) % (2 * math.pi)

        min_angle = min(a1, a2, self.M_angle)

        right_vertices = np.roll(self.vertices, -1, axis=0)
        right_vectors = right_vertices - self.vertices
        edge_lengths = self.get_lengths()
        y2_y1x0 = right_vectors[:, 1] * new_vertex[0]
        x2_x1y0 = right_vectors[:, 0] * new_vertex[1]
        x2y1_y2x1 = right_vertices[:, 0] * self.vertices[:, 1] - right_vertices[:, 1] * self.vertices[:, 0]
        distance_to_edges = np.abs(y2_y1x0 - x2_x1y0 + x2y1_y2x1) / np.clip(edge_lengths, 1e-6, None)

        d_min = np.min(distance_to_edges)

        new_edge_length1 = math.sqrt(vector_left[0] ** 2 + vector_left[1] ** 2)
        new_edge_length2 = math.sqrt(vector_right[0] ** 2 + vector_right[1] ** 2)
        average_new_edges = (new_edge_length1 + new_edge_length2) / 2

        q_dist = min(d_min / max(average_new_edges, 1e-6), 1)

        return math.sqrt(min_angle / self.M_angle * q_dist) - 1

    def n_b_type1(self, direction: Literal["left", "right"]) -> float:
        vertex_index1, vertex_index2 = self.get_type1_selected_indices(direction)

        vector_left = self.v_at(vertex_index1 - 1) - self.v_at(vertex_index1)
        vector_right = self.v_at(vertex_index2 + 1) - self.v_at(vertex_index2)
        vector_mid = self.v_at(vertex_index2) - self.v_at(vertex_index1)

        a1 = (vector_angle(vector_left) - vector_angle(vector_mid)) % (2 * math.pi)
        a2 = (vector_angle(-vector_mid) - vector_angle(vector_right)) % (2 * math.pi)

        min_angle = min(a1, a2, self.M_angle)

        return math.sqrt(min_angle / self.M_angle) - 1

    def n_e(self, quad: Polygon) -> float:
        angles = quad.get_angles()
        q_angle = np.min(angles) / max(np.max(angles), 1e-6)

        v1 = quad.vertices[0]
        v2 = quad.vertices[1]
        v3 = quad.vertices[2]
        v4 = quad.vertices[3]

        d1 = math.sqrt((v3[0] - v1[0]) ** 2 + (v3[1] - v1[1]) ** 2)
        d2 = math.sqrt((v4[0] - v2[0]) ** 2 + (v4[1] - v2[1]) ** 2)
        d_max = max(d1, d2)
        L_min = quad.get_shortest_length()
        q_edge = math.sqrt(2) * L_min / max(d_max, 1e-6)

        return math.sqrt(q_angle * q_edge)

    def type0_r(self, new_normalized_polar_pair: tuple[float, float]) -> tuple[float, bool]:
        new_normalized_local_vertex = self.type0_normalized_polar_to_local_Cartesian(new_normalized_polar_pair)
        N = self.vertices.shape[0]
        vertex_index = self.get_sharpest_vertex_index()
        r = self.get_coordinate_space_radius(vertex_index)
        local_vertex = new_normalized_local_vertex * r
        actual_vertex = self.get_inverse_transformed_vertex(local_vertex, vertex_index)

        if self.type0_is_intersecting(new_normalized_polar_pair):
            return -10.0, False

        try:
            quad = Polygon(np.array([
                self.v_at(vertex_index),
                self.v_at(vertex_index + 1),
                actual_vertex,
                self.v_at(vertex_index - 1)
            ]))
        except:
            return -10.0, False

        u = self.u(quad)
        n_e = self.n_e(quad)
        n_b = self.n_b_type0(actual_vertex, vertex_index)

        mt = n_e + n_b + u
        return mt, True

    def type0_update_boundary_with_normalized_local(self, new_normalized_local_vertex: npt.NDArray[np.floating]):
        vertex_index = self.get_sharpest_vertex_index()
        actual_vertex = self.type0_local_Cartesian_to_actual(new_normalized_local_vertex)
        self.vertices[vertex_index] = actual_vertex

    def type0_local_Cartesian_to_actual(self, new_normalized_local_vertex: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        vertex_index = self.get_sharpest_vertex_index()
        r = self.get_coordinate_space_radius(vertex_index)
        actual_vertex = self.get_inverse_transformed_vertex(new_normalized_local_vertex * r, vertex_index)
        return actual_vertex

    def type0_normalized_polar_to_local_Cartesian(self, new_normalized_polar_pair: tuple[float, float]) -> npt.NDArray[np.floating]:
        """
        new_normalized_polar_pair: (angle, radius)
        """
        vertex_index = self.get_sharpest_vertex_index()
        opening_angle = self.get_angle_at_index(vertex_index)

        angle = (0.9 * new_normalized_polar_pair[0] + 0.05) * opening_angle
        radius = 0.9 * new_normalized_polar_pair[1] + 0.05
        return np.array([radius * math.cos(angle), radius * math.sin(angle)])

    def type0_update_boundary_with_polar_pair(self, new_normalized_polar_pair: tuple[float, float]):
        new_normalized_local_vertex = self.type0_normalized_polar_to_local_Cartesian(new_normalized_polar_pair)
        self.type0_update_boundary_with_normalized_local(new_normalized_local_vertex)

    def type0_is_intersecting(self, new_normalized_polar_pair: tuple[float, float]) -> bool:
        N = self.vertices.shape[0]
        vertex_index = self.get_sharpest_vertex_index()
        left_vertex_index = (vertex_index - 1) % N
        right_vertex_index = (vertex_index + 1) % N

        left_vertex = self.v_at(left_vertex_index)
        right_vertex = self.v_at(right_vertex_index)

        new_normalized_local_vertex = self.type0_normalized_polar_to_local_Cartesian(new_normalized_polar_pair)
        r = self.get_coordinate_space_radius(vertex_index)
        actual_vertex = self.get_inverse_transformed_vertex(new_normalized_local_vertex * r, vertex_index)

        new_edge1 = (left_vertex, actual_vertex)
        new_edge2 = (actual_vertex, right_vertex)

        for i in range(N):
            if not (i == (left_vertex_index - 1) % N or i == left_vertex_index):
                if segment_intersect(
                    new_edge1[0], new_edge1[1],
                    self.v_at(i), self.v_at(i + 1)
                ):
                    return True
            if not (i == (right_vertex_index - 1) % N or i == right_vertex_index):
                if segment_intersect(
                    new_edge2[0], new_edge2[1],
                    self.v_at(i), self.v_at(i + 1)
                ):
                    return True
        return False

    def type1_is_intersecting(self, direction: Literal["left", "right"]) -> bool:
        N = self.vertices.shape[0]
        vertex_index1, vertex_index2 = self.get_type1_selected_indices(direction)
        edge = (self.v_at(vertex_index1), self.v_at(vertex_index2))
        for i in range(N):
            if not (i == (vertex_index1 - 1) % N or i == vertex_index1 or i == (vertex_index2 - 1) % N or i == vertex_index2):
                if segment_intersect(
                    edge[0], edge[1],
                    self.v_at(i), self.v_at(i + 1)
                ):
                    return True
        return False

    def get_type1_selected_indices(self, direction: Literal["left", "right"]) -> tuple[np.intp, np.intp]:
        N = self.vertices.shape[0]
        vertex_index = self.get_sharpest_vertex_index()
        if direction == "left":
            return (vertex_index - 1) % N, vertex_index
        else:
            return vertex_index, (vertex_index + 1) % N

    def type1_r(self, direction: Literal["left", "right"]) -> tuple[float, bool]:
        vertex_index1, vertex_index2 = self.get_type1_selected_indices(direction)

        try:
            quad = Polygon(np.array([
                self.v_at(vertex_index1),
                self.v_at(vertex_index2),
                self.v_at(vertex_index2 + 1),
                self.v_at(vertex_index1 - 1)
            ]))
        except:
            return -10.0, False

        if self.type1_is_intersecting(direction):
            return -10.0, False

        u = self.u(quad)
        n_e = self.n_e(quad)
        n_b = self.n_b_type1(direction)

        mt = n_e + n_b + u
        return mt, True

    def type1_update_boundary(self, direction: Literal["left", "right"]):
        vertex_index1, vertex_index2 = self.get_type1_selected_indices(direction)
        self.vertices = np.delete(self.vertices, [vertex_index1, vertex_index2], axis=0)

    def reset_with_boundary(self, boundary: npt.NDArray[np.floating]):
        super().__init__(boundary)

    # not quad left, more like pentagon left
    def is_quad_left(self) -> bool:
        return self.vertices.shape[0] <= 5
