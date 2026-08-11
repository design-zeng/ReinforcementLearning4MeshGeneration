import math
from typing import Any

import numpy as np

from general.mesh import Mesh
from general.geometry import Vertex
from general.geometry import detransformation
from general.plotting import render_boundary, close_render


class MeshEnv:
    def __init__(self, boundary):
        self.mesh = Mesh(boundary)
        self.current_area = self.mesh.original_area

        self.neighbor_num = 6
        self.radius_num = 3
        self.radius = 4

        self.current_ref_state: Any = None
        self.not_valid_points = []
        self.last_not_valid_points = []
        self.target_angle = 0

    def reset(self, static=False):
        self.mesh.reset()
        self.not_valid_points = []
        self.current_area = self.mesh.original_area
        self.current_ref_state = None
        return self.find_next_state(static=static)

    def find_next_state(self, not_valid_points=None, static=False):
        r_p = self.mesh.updated_boundary.find_reference_point(not_valid_points, target_angle=self.target_angle)

        if r_p:
            self.current_ref_state = self.mesh.updated_boundary.reference_state(
                r_p, self.neighbor_num, self.radius_num,
                self.current_area / self.mesh.original_area, self.radius, static)
            return np.array(self.current_ref_state.state).astype(np.float32)
        else:
            return None

    def get_middle_points(self, state):
        v1 = np.asarray([state[self.neighbor_num], state[self.neighbor_num + 1]], dtype=float)
        v2 = np.asarray([state[self.neighbor_num + 2], state[self.neighbor_num + 3]], dtype=float)
        return v1, v2

    def detransformation(self, point, is_move=False):
        v1, v2 = self.get_middle_points(Vertex.points_as_array(self.current_ref_state.neighbors))

        de_point = detransformation(point, self.current_ref_state.base_length if not is_move else 1, v1, v2)
        return Vertex(round(de_point[0], 4), round(de_point[1], 4))

    def close(self):
        close_render()

    def render(self, mode='human'):
        print(f'Generated elements: {len(self.mesh.generated_quads)}')
        render_boundary(self.mesh.boundary)
