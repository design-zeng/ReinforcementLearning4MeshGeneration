from typing import Literal

import numpy as np

from boundary import Boundary

class Mesh:
    def __init__(self, initial_boundary: Boundary):
        self.vertices = initial_boundary.vertices.copy()

        N = initial_boundary.vertices.shape[0]
        self.edges = np.stack((np.arange(N), np.roll(np.arange(N), 1))).T
        # edges = [[0, 1], [1, 2], ..., [N - 1, 0]]

        self.boundary = initial_boundary
        self.boundary_indices_mapping = np.arange(N)
    
    def type0_update_mesh(self, polar_pair=(None, None)):
        N = self.boundary.vertices.shape[0]

        boundary_index = self.boundary.get_sharpest_vertex_index()

        new_normalized_vertex = self.boundary.type0_normalized_polar_to_local_Cartesian(polar_pair)
        new_actual_vertex = self.boundary.type0_local_Cartesian_to_actual(new_normalized_vertex)

        # find index of left and right vertex among the mesh vertices
        # which is different from boundary_index, which follows
        # the counterclockwise order on the boundary vertices
        left_vertex_mesh_index = self.boundary_indices_mapping[(boundary_index - 1) % N]
        right_vertex_mesh_index = self.boundary_indices_mapping[(boundary_index + 1) % N]

        # we will append the new vertex at the end of the mesh vertices
        # hence, its index is the length of the current list of vertices
        new_edge1 = np.array(((left_vertex_mesh_index, self.vertices.shape[0]), ))
        new_edge2 = np.array(((right_vertex_mesh_index, self.vertices.shape[0]), ))

        # update the indices mapping to point to the new mesh vertex
        self.boundary_indices_mapping[boundary_index] = self.vertices.shape[0]

        self.vertices = np.vstack((self.vertices, new_actual_vertex))

        self.edges = np.vstack((self.edges, new_edge1, new_edge2))
    
    def type1_update_mesh(self, direction: Literal["left", "right"]):
        N = self.boundary.vertices.shape[0]

        boundary_index1, boundary_index2 = self.boundary.get_type1_selected_indices(direction)

        left_vertex_mesh_index = self.boundary_indices_mapping[(boundary_index1 - 1) % N]
        right_vertex_mesh_index = self.boundary_indices_mapping[(boundary_index2 + 1) % N]

        new_edge = np.array(((left_vertex_mesh_index, right_vertex_mesh_index), ))
        self.edges = np.vstack((self.edges, new_edge))

        self.boundary_indices_mapping = np.delete(self.boundary_indices_mapping, [boundary_index1, boundary_index2])
    
    def update_mesh(self, action_type: Literal[-1, 0, 1], polar_pair=(None, None)):
        if action_type == 0:
            self.type0_update_mesh(polar_pair)
        elif action_type == 1:
            self.type1_update_mesh("right")
        elif action_type == -1:
            self.type1_update_mesh("left")
