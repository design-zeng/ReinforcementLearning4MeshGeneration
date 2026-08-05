import math
import numpy as np
import numpy.typing as npt

from geometry_lib import segment_intersect


# array of N 2-tuples, counter-clockwise
class Polygon:
    def __init__(self, vertices: npt.NDArray[np.floating]):
        self.vertices = vertices.astype(float)
        if self.is_self_intersecting():
            raise Exception("Boundary self-intersects")

        # check counter-clockwise boundary
        if self.get_signed_area() < 0:
            self.vertices = np.flip(self.vertices, axis=0)

    def v_at(self, vertex_index: np.intp | int) -> npt.NDArray[np.floating]:
        N = self.vertices.shape[0]
        return self.vertices[vertex_index % N]

    def is_self_intersecting(self) -> bool:
        N = self.vertices.shape[0]
        for i in range(N):
            start_j = i + 2
            end_j = N - 1 if i == 0 else N
            for j in range(start_j, end_j):
                if segment_intersect(
                    self.vertices[i], self.vertices[(i + 1) % N],
                    self.vertices[j], self.vertices[(j + 1) % N]
                ):
                    return True
        return False

    #from ChatGPT
    def get_signed_area(self) -> float:
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def get_area(self) -> float:
        return abs(self.get_signed_area())

    # j determines how far the neighbor, j=1 is direct neighbor
    def get_angles(self, j=1) -> npt.NDArray[np.floating]:
        Vi = self.vertices
        Vlj_sub_Vi = np.roll(Vi, j, axis=0) - Vi
        Vrj_sub_Vi = np.roll(Vi, -j, axis=0) - Vi
        l_angle = np.arctan2(Vlj_sub_Vi[:, 1], Vlj_sub_Vi[:, 0])
        r_angle = np.arctan2(Vrj_sub_Vi[:, 1], Vrj_sub_Vi[:, 0])
        angles = (l_angle - r_angle) % (2 * math.pi)
        return angles

    def get_angle_at_index(self, vertex_index: int | np.intp) -> float:
        N = self.vertices.shape[0]
        left_vector = self.vertices[(vertex_index - 1) % N] - self.vertices[vertex_index]
        right_vector = self.vertices[(vertex_index + 1) % N] - self.vertices[vertex_index]
        left_vector_angle = math.atan2(left_vector[1], left_vector[0])
        right_vector_angle = math.atan2(right_vector[1], right_vector[0])
        return (left_vector_angle - right_vector_angle) % (2 * math.pi)

    def get_lengths(self) -> npt.NDArray[np.floating]:
        return np.sqrt(np.sum(np.square(np.roll(self.vertices, -1, axis=0) - self.vertices), axis=1))

    def get_shortest_length(self) -> float:
        return float(np.min(self.get_lengths()))

    def get_longest_length(self) -> float:
        return float(np.max(self.get_lengths()))
