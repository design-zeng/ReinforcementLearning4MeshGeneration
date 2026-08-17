import numpy as np

from general.geometry import Polygon, Mesh
from general.boundary import Boundary
from general.recognition import recognize
from general.action import polar_act, act_last
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
        self.candidates = []
        Boundary.score_candidates(self.boundary, self.candidates)
        self.failed_points = []
        self.view = {}

        recognize(self.boundary, self.candidates, self.view,
                  self.mesh.area_ratio(), static)
        if not self.view['reference_point']:
            return None
        return np.array(self.view['state']).astype(np.float32)

    def step(self, action, rule_type):
        done = False
        absorbed = False
        is_complete = True
        observation = None

        reference_point = self.view['reference_point']

        if len(self.boundary.vertices) <= 5:
            done = True
        else:
            outcome = {}
            polar_act(self.view, action, rule_type, self.boundary, self.mesh, outcome)

            if outcome['absorbed']:
                absorbed = True

                if len(self.boundary.vertices) <= 5:
                    done = True
                    if len(self.boundary.vertices) == 4:
                        act_last(self.boundary, self.mesh)
            elif reference_point not in self.failed_points:
                self.failed_points.append(reference_point)

            recognize(self.boundary, self.candidates, self.view,
                      self.mesh.area_ratio(), True, self.failed_points)
            if self.view['reference_point']:
                observation = np.array(self.view['state']).astype(np.float32)
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
                    Boundary.score_candidates(self.boundary, self.candidates)

                    if len(self.last_failed_points) > 0 and len(self.failed_points) > 0 and \
                            self.last_failed_points[0] == self.failed_points[0] and \
                            self.last_failed_points[-1] == self.failed_points[-1] and \
                            len(self.failed_points) == len(self.last_failed_points):
                        done = True

                    self.last_failed_points = self.failed_points
                    self.failed_points = []

                    recognize(self.boundary, self.candidates, self.view,
                              self.mesh.area_ratio(), True, self.failed_points)
                    if self.view['reference_point']:
                        observation = np.array(self.view['state']).astype(np.float32)
                    else:
                        done = True

        return observation, 0, done, {'is_complete': is_complete}

    def render(self):
        render_boundary(Polygon(self.mesh.vertices()))

    def close(self):
        close_render()
