import numpy as np
from cmad.fem_utils.mesh import Mesh
from cmad.fem_utils import interpolants
from cmad.fem_utils import quadrature_rule

class fem_problem():
    def __init__(self, problem_type, order):

        # evaluate triangle basis functions at quadrature points
        self._quad_rule_2D \
            = quadrature_rule.create_quadrature_rule_on_triangle(order)
        gauss_pts_2D = self._quad_rule_2D.xigauss
        self._shape_func_triangle = interpolants.shape_triangle(gauss_pts_2D)

        # evaluate tetrahedton basis functions at quadrature points
        self._quad_rule_3D \
            = quadrature_rule.create_quadrature_rule_on_tetrahedron(order)
        gauss_pts_3D = self._quad_rule_3D.xigauss
        self._shape_func_tetra = interpolants.shape_tetrahedron(gauss_pts_3D)

        if problem_type == "hole_block_example":

            self._mesh = Mesh("hole_block")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._colume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            self._dof_node = 3
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._colume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            pres_nodes = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][0] == 0.0:
                    pres_nodes.append(i)
            NUM_PRES_NODES = len(pres_nodes)

            self._disp_node = np.zeros((NUM_PRES_NODES \
                                        * self._dof_node, 2), dtype = int)
            for i in range(NUM_PRES_NODES):
                for j in range(self._dof_node):
                    self._disp_node[i * self._dof_node + j][0] \
                        = pres_nodes[i]
                    self._disp_node[i * self._dof_node + j][1] = j + 1
            self._disp_val = np.zeros(NUM_PRES_NODES * self._dof_node)

            # normal traction on plane x = 1
            self._surf_traction_vector = np.array([1.0, 0.0, 0.0])
            pres_surf = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 1 * np.ones(3)).all():
                    pres_surf.append(surface)
            self._pres_surf = np.array(pres_surf)

        if problem_type == "uniaxial_stress":

            self._mesh = Mesh("bar")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._colume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            self._dof_node = 3
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._colume_conn)
            self._num_nodes_surf = 3

            # prescribe ux = 0 on x = 0, uy = 0 on y = 0, and uz = 0 on z = 0
            disp_node = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype = int))
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype = int))
                if self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 3], dtype = int))
            self._disp_node = np.array(disp_node, dtype = int)
            self._disp_val = np.zeros(len(disp_node))

            # normal traction on x = 2
            self._surf_traction_vector = np.array([1.0, 0.0, 0.0])
            pres_surf = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 2 * np.ones(3)).all():
                    pres_surf.append(surface)
            self._pres_surf = np.array(pres_surf)

        # setup patch test form B
        if problem_type == "patch_B":

            self._mesh = Mesh("cube")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._colume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            self._dof_node = 3
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._colume_conn)
            self._num_nodes_surf = 3

            #impose linear displacement field on the boundary
            disp_node = []
            disp_val = []
            self.UUR_true = np.zeros((self._num_nodes, self._dof_node))

            for i in range(self._num_nodes):
                coord = self._nodal_coords[i]
                x = coord[0]
                y = coord[1]
                z = coord[2]

                ux = (2 * x + y + z - 4) / 2
                uy = (x + 2 * y + z - 4) / 2
                uz = (x + y + 2 * z - 4) / 2

                self.UUR_true[i, :] = np.array([ux, uy, uz])

                if (x == 0.0 or x == 2.0 or y == 0.0
                    or y == 2.0 or z == 0 or z == 2.0):
                    disp_node.append(np.array([i, 1], dtype = int))
                    disp_node.append(np.array([i, 2], dtype = int))
                    disp_node.append(np.array([i, 3], dtype = int))
                    disp_val.append(ux)
                    disp_val.append(uy)
                    disp_val.append(uz)

            self._disp_node = np.array(disp_node, dtype = int)
            self._disp_val = np.array(disp_val)

            # no surface tractions
            self._surf_traction_vector = np.array([0.0, 0.0, 0.0])
            self._pres_surf = np.array([])

        if problem_type == "simple_shear":

            self._mesh = Mesh("cube")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._colume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            self._dof_node = 3
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._colume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane z = 0, and set uz = 0 on plane z = 2
            disp_node = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 1], dtype = int))
                    disp_node.append(np.array([i, 2], dtype = int))
                    disp_node.append(np.array([i, 3], dtype = int))
                if self._nodal_coords[i][2] == 2.0:
                    disp_node.append(np.array([i, 3], dtype = int))

            self._disp_node = np.array(disp_node, dtype = int)
            self._disp_val = np.zeros(len(disp_node))

            # shear traction in x direction on plane z = 2
            self._surf_traction_vector = np.array([1.0, 0.0, 0.0])
            pres_surf = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 2] == 2 * np.ones(3)).all():
                    pres_surf.append(surface)
            self._pres_surf = np.array(pres_surf)

    def get_2D_basis_functions(self):
        return self._quad_rule_2D, self._shape_func_triangle

    def get_3D_basis_functions(self):
        return self._quad_rule_3D, self._shape_func_tetra

    def get_mesh_properties(self):
        return self._dof_node, self._num_nodes, self._num_nodes_elem, \
            self._num_elem, self._num_nodes_surf, self._nodal_coords, \
            self._colume_conn

    def get_boundary_conditions(self):
        return self._disp_node, self._disp_val, \
            self._pres_surf, self._surf_traction_vector

    def save_data(self, filename, data):
        self._mesh.add_point_data(data)
        self._mesh.save_mesh(filename)