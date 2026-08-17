import numpy as np

from general.geometry import Quad, Polygon, Mesh
from general.recognition import reference_state, reference_candidates, next_reference_point, updated_candidates
from general.action import polar_action_point, rule_quad
from general.quality import can_add_quad
from general.smoothing import smooth_pave
from general.plotting import render_boundary, close_render


class Ebrd_Env:
    def __init__(self, boundary):
        self.problem = boundary.deep_copy()
        self.last_failed_points = []

        self.reset()

    def reset(self, static=False):
        fresh = self.problem.deep_copy()
        self.mesh = Mesh(fresh)
        self.boundary = fresh.copy()
        self.candidates = reference_candidates(self.boundary)
        self.current_area = self.mesh.original_area
        self.failed_points = []
        self.current_ref_state = None

        reference_point = next_reference_point(self.candidates)
        if not reference_point:
            return None
        self.current_ref_state = reference_state(
            self.boundary, reference_point, self.current_area / self.mesh.original_area, static)
        return np.array(self.current_ref_state['state']).astype(np.float32)

    def step(self, action, rule_type):
        done = False
        absorbed = False
        is_complete = True
        observation = None

        reference_point = self.current_ref_state['reference_point']

        if len(self.boundary.vertices) <= 5:
            done = True
        else:
            index = self.boundary.vertices.index(reference_point)
            new_point = polar_action_point(self.current_ref_state, action)

            if rule_type <= 0.3:
                quad = rule_quad(self.boundary, -1, index)
            elif rule_type >= 0.7:
                quad = rule_quad(self.boundary, 1, index)
            elif self.boundary.contains_point(new_point):
                quad = rule_quad(self.boundary, 0, index, new_point)
            else:
                quad = None

            if quad is not None and can_add_quad(self.boundary, quad, reference_point):
                absorbed = True
                self.mesh.add_quad(quad)
                retired, rescored = self.boundary.update_boundary(quad)
                self.candidates = updated_candidates(self.candidates, self.boundary, retired, rescored)

                if len(self.boundary.vertices) <= 5:
                    done = True
                    if len(self.boundary.vertices) == 4:
                        self.mesh.add_quad(Quad(self.boundary.vertices))
            elif reference_point not in self.failed_points:
                self.failed_points.append(reference_point)

            next_point = next_reference_point(self.candidates, self.failed_points)
            if next_point:
                self.current_ref_state = reference_state(
                    self.boundary, next_point, self.current_area / self.mesh.original_area, True)
                observation = np.array(self.current_ref_state['state']).astype(np.float32)
            if absorbed:
                self.failed_points = []

            if len(self.boundary.vertices) > 4:
                is_complete = False
                if observation is None:
                    # the paper stops a trajectory once no valid element can be
                    # extracted; here the accumulated conflicts are resolved
                    # (smooth) and extraction retried, giving up only when the
                    # same failure set recurs
                    smooth_pave(self.mesh, self.boundary)
                    self.candidates = reference_candidates(self.boundary)

                    if len(self.last_failed_points) > 0 and len(self.failed_points) > 0 and \
                            self.last_failed_points[0] == self.failed_points[0] and \
                            self.last_failed_points[-1] == self.failed_points[-1] and \
                            len(self.failed_points) == len(self.last_failed_points):
                        done = True

                    self.last_failed_points = self.failed_points
                    self.failed_points = []

                    next_point = next_reference_point(self.candidates, self.failed_points)
                    if next_point:
                        self.current_ref_state = reference_state(
                            self.boundary, next_point, self.current_area / self.mesh.original_area, True)
                        observation = np.array(self.current_ref_state['state']).astype(np.float32)
                    else:
                        done = True

        return observation, 0, done, {'is_complete': is_complete}

    def render(self):
        render_boundary(Polygon(self.mesh.vertices()))

    def close(self):
        close_render()
