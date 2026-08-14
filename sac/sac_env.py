import numpy as np
import gym
from gym import spaces

from general.geometry import Quad, Polygon
from general.mesh import Mesh
from general.recognition import reference_state, reference_candidates, next_reference_point, updated_candidates
from general.action import action_point
from general.quality import combined_quality
from general.plotting import render_boundary, close_render


class Sac_Env(gym.Env):
    def __init__(self, boundary, neighbor_num=6, radius_num=3, radius=4, target_angle=0):
        self.problem = boundary.deep_copy()
        self.mesh = Mesh(boundary)
        self.boundary = boundary.copy()
        self.current_area = self.mesh.original_area

        self.neighbor_num = neighbor_num
        self.radius_num = radius_num
        self.radius = radius
        self.target_angle = target_angle

        self.current_ref_state = None
        self.reference_candidates = None
        self.failed_num = 0
        self.estimated_area_range = None
        self.history_info = {
            -1: [],
            1: [],
            0: []
        }

        self.action_space = spaces.Box(np.array([-1, -1.5, 0]), np.array([1, 1.5, 1.5]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-999, high=999,
            shape=(2 * (self.neighbor_num + self.radius_num), ),
            dtype=np.float32)

    def reset(self, static=False):
        fresh = self.problem.deep_copy()
        self.mesh = Mesh(fresh)
        self.boundary = fresh.copy()
        self.reference_candidates = reference_candidates(self.boundary, self.target_angle)
        self.current_area = self.mesh.original_area
        self.current_ref_state = None
        self.failed_num = 0
        self.estimated_area_range = self.boundary.estimate_area_range()
        return self.recognize(static=static)

    def step(self, action):
        done = False
        failed = True
        reward = 0

        if not self.conflicts_remain():
            reward = 10
            done = True
        else:
            solution, rule, new_vertex = self.synthesize(action)

            if self.conflict_free(solution):
                self.absorb(solution)
                reward += self.resolution_quality(solution, new_vertex)
                self.history_info[rule].append(reward)
                failed = False

                if not self.conflicts_remain():
                    reward += 10
                    done = True
                    self.absorb_final_cell()
            else:
                reward += self.rejection_penalty()

        next_state = self.recognize()

        is_complete = True
        if not failed:
            self.failed_num = 0
        else:
            self.failed_num += 1
            if self.failed_num >= 100:
                done = True
                is_complete = False
        return next_state, np.float64(reward), done, {'is_complete': is_complete}

    def conflicts_remain(self):
        return len(self.boundary.vertices) > 5

    def recognize(self, static=False):
        reference_point = next_reference_point(self.reference_candidates)

        if reference_point:
            self.current_ref_state = reference_state(
                self.boundary, reference_point, self.neighbor_num, self.radius_num,
                self.current_area / self.mesh.original_area, self.radius, static)
            return np.array(self.current_ref_state['state']).astype(np.float32)
        else:
            return None

    def synthesize(self, action):
        rule_type = action[0]
        new_point = action_point(self.current_ref_state,
                                 [round(action[1], 4), round(action[2], 4)], self.neighbor_num)
        reference_point = self.current_ref_state['reference_point']
        index = self.boundary.vertices.index(reference_point)

        if rule_type <= -0.5:
            return self.boundary.rule_quad(-1, index), -1, None
        if rule_type >= 0.5:
            return self.boundary.rule_quad(1, index), 1, None
        if not self.boundary.contains_point(new_point):
            return None, 0, None
        if self.boundary.find_same_point(new_point):
            return self.boundary.rule_quad(-1, index), 0, None
        return self.boundary.rule_quad(0, index, new_point), 0, new_point

    def conflict_free(self, solution):
        return solution is not None and \
            self.mesh.can_commit_quad(self.boundary, solution, self.current_ref_state['reference_point'])

    def absorb(self, solution):
        retired, rescored = self.mesh.commit_quad(self.boundary, solution)
        self.reference_candidates = updated_candidates(self.reference_candidates, self.boundary, retired, rescored)
        self.current_area -= solution.area()

    def absorb_final_cell(self):
        if len(self.boundary.vertices) == 4:
            final_cell = Quad(self.boundary.vertices)
            final_cell.connect_vertices()
            self.mesh.generated_quads.append(final_cell)

    def resolution_quality(self, solution, new_vertex=None):
        quality = combined_quality(self.boundary, solution, new_vertex)

        solution_area = solution.area()
        min_area, critical_area = self.estimated_area_range
        if min_area <= solution_area < critical_area:
            speed_penalty = (solution_area - critical_area) / (critical_area - min_area)
        elif solution_area < min_area:
            speed_penalty = -1
        else:
            speed_penalty = 0
        return quality + speed_penalty

    def rejection_penalty(self):
        n = len(self.mesh.generated_quads)
        return -1 / n if n else -1

    def seed(self, seed=None):
        return [seed]

    def render(self, mode='human'):
        render_boundary(Polygon(self.mesh.vertices()))

    def close(self):
        close_render()
