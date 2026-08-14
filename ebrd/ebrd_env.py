import math

import numpy as np

from general.config import NEIGHBOR_NUM, RADIUS
from general.geometry import Quad, Vertex, Polygon, Mesh
from general.recognition import reference_state, reference_candidates, next_reference_point, updated_candidates
from general.action import from_local_frame
from general.smoothing import smooth_pave
from general.plotting import render_boundary, close_render


class Ebrd_Env:
    TYPE_THRESHOLD = 0.3

    def __init__(self, boundary):
        self.problem = boundary.deep_copy()
        self.mesh = Mesh(boundary)
        self.boundary = boundary.copy()
        self.current_area = self.mesh.original_area

        self.current_ref_state = None
        self.candidates = None
        self.not_valid_points = []
        self.last_not_valid_points = []

    def reset(self, static=False):
        fresh = self.problem.deep_copy()
        self.mesh = Mesh(fresh)
        self.boundary = fresh.copy()
        self.candidates = reference_candidates(self.boundary)
        self.not_valid_points = []
        self.current_area = self.mesh.original_area
        self.current_ref_state = None
        return self.find_next_state(static=static)

    def find_next_state(self, not_valid_points=None, static=False):
        r_p = next_reference_point(self.candidates, not_valid_points)

        if r_p:
            self.current_ref_state = reference_state(
                self.boundary, r_p, self.current_area / self.mesh.original_area, static)
            return np.array(self.current_ref_state['state']).astype(np.float32)
        else:
            return None

    def get_middle_points(self, state):
        v1 = np.asarray([state[NEIGHBOR_NUM], state[NEIGHBOR_NUM + 1]], dtype=float)
        v2 = np.asarray([state[NEIGHBOR_NUM + 2], state[NEIGHBOR_NUM + 3]], dtype=float)
        return v1, v2

    def detransformation(self, point, is_move=False):
        v1, v2 = self.get_middle_points(Vertex.flatten(self.current_ref_state['neighbors']))

        de_point = from_local_frame(point, self.current_ref_state['base_length'] if not is_move else 1, v1, v2)
        return Vertex(round(de_point[0], 4), round(de_point[1], 4))

    def close(self):
        close_render()

    def render(self):
        render_boundary(Polygon(self.mesh.vertices()))

    def move(self, new_point, rule_type):
        x = self.current_ref_state['base_length'] * RADIUS * new_point[0] * math.cos(new_point[1])
        y = self.current_ref_state['base_length'] * RADIUS * new_point[0] * math.sin(new_point[1])

        new_point = self.detransformation([round(x, 6), round(y, 6)], is_move=True)

        done = False
        next_state = None
        not_valid_element = True
        is_complete = True

        reference_point = self.current_ref_state['reference_point']

        if len(self.boundary.vertices) <= 5:
            done = True

        else:
            index = self.boundary.vertices.index(reference_point)
            quad = None

            if rule_type <= self.TYPE_THRESHOLD:
                quad = self.boundary.rule_quad(-1, index)
            elif rule_type >= 1 - self.TYPE_THRESHOLD:
                quad = self.boundary.rule_quad(1, index)
            else:
                if self.boundary.contains_point(new_point):
                    quad = self.boundary.rule_quad(0, index, new_point)

            if quad is None:
                pass
            elif self.boundary.can_add_quad(quad, reference_point):

                not_valid_element = False
                self.mesh.add_quad(quad)
                retired, rescored = self.boundary.update_boundary(quad)
                self.candidates = updated_candidates(self.candidates, self.boundary, retired, rescored)

                next_state = self.find_next_state(self.not_valid_points, static=True)

                if len(self.boundary.vertices) <= 5:
                    done = True
                    if len(self.boundary.vertices) == 4:
                        quad = Quad(self.boundary.vertices)
                        self.mesh.generated_quads.append(quad)

            if not_valid_element:
                if reference_point not in self.not_valid_points:
                    self.not_valid_points.append(reference_point)
                next_state = self.find_next_state(self.not_valid_points, static=True)
            else:
                self.not_valid_points = []

            if len(self.boundary.vertices) > 4:
                is_complete = False
                if next_state is None:
                    smooth_pave(self.mesh, self.boundary)
                    self.candidates = reference_candidates(self.boundary)

                    if len(self.last_not_valid_points) > 0 and len(self.not_valid_points) > 0:
                        if self.last_not_valid_points[0] == self.not_valid_points[0] and \
                            self.last_not_valid_points[-1] == self.not_valid_points[-1] and \
                                len(self.not_valid_points) == len(self.last_not_valid_points):
                            done = True

                    self.last_not_valid_points = self.not_valid_points
                    self.not_valid_points = []

                    next_state = self.find_next_state(not_valid_points=self.not_valid_points, static=True)
                    if next_state is None:
                        done = True

            else:
                is_complete = True

        return next_state, 0, done, {'is_complete': is_complete}
