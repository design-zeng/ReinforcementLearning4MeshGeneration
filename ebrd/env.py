import math

from general.geometry import Quad
from general.mesh_env import MeshEnv


class EbrdEnv(MeshEnv):
    TYPE_THRESHOLD = 0.3

    def move(self, new_point, rule_type, lr_1=None, lr_2=None):
        x = self.current_ref_state.base_length * self.radius * new_point[0] * math.cos(new_point[1])
        y = self.current_ref_state.base_length * self.radius * new_point[0] * math.sin(new_point[1])

        new_point = self.detransformation([round(x, 6), round(y, 6)], is_move=True)

        done = False
        next_state = None
        not_valid_element = True
        is_complete = True

        reference_point = self.current_ref_state.reference_point

        if len(self.boundary.vertices) <= 5:
            reward = 10
            done = True

        else:
            index = self.boundary.vertices.index(reference_point)
            quad = None

            if rule_type <= self.TYPE_THRESHOLD:
                quad = self.boundary.rule_element(-1, index)
            elif rule_type >= 1 - self.TYPE_THRESHOLD:
                quad = self.boundary.rule_element(1, index)
            else:
                if self.boundary.contains_point(new_point):
                    quad = self.boundary.rule_element(0, index, new_point)

            if quad is None:
                pass
            elif self.mesh.can_commit_quad(self.boundary, quad, reference_point):

                not_valid_element = False
                self.mesh.commit_quad(self.boundary, quad, reference_point)

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
                    if lr_1 and lr_2:
                        self.mesh.smooth_pave(self.boundary, self.mesh.boundary.vertices, self.boundary.vertices, lr_1, lr_2, iteration=400)
                    else:
                        self.mesh.smooth_pave(self.boundary, self.mesh.boundary.vertices, self.boundary.vertices, iteration=400)

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
