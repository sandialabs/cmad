import numpy as np
from cmad.fem_utils.mesh.mesh import Mesh
from cmad.fem_utils.interpolants import interpolants
from cmad.fem_utils.quadrature import quadrature_rule
import meshio

class fem_problem():
    def __init__(self, problem_type, order, mixed = False):

        # evaluate triangle basis functions at quadrature points
        quad_rule_tri \
            = quadrature_rule.create_quadrature_rule_on_triangle(order)
        gauss_pts_tri = quad_rule_tri.xigauss
        shape_func_tri = interpolants.shape_triangle(gauss_pts_tri)

        # evaluate tetrahedron basis functions at quadrature points
        quad_rule_tetra \
            = quadrature_rule.create_quadrature_rule_on_tetrahedron(order)
        gauss_pts_tetra = quad_rule_tetra.xigauss
        shape_func_tetra = interpolants.shape_tetrahedron(gauss_pts_tetra)

        # evaluate brick basis functions at quadrature points
        quad_rule_brick \
            = quadrature_rule.create_quadrature_rule_on_brick(order)
        gauss_pts_brick = quad_rule_brick.xigauss
        shape_func_brick = interpolants.shape_brick(gauss_pts_brick)

        # evaluate square basis functions at quadrature points
        quad_rule_square \
            = quadrature_rule.create_quadrature_rule_on_quad(order)
        gauss_pts_square = quad_rule_square.xigauss
        shape_func_square = interpolants.shape_quad(gauss_pts_square)

        # evaluate 1D basis functions at quadrature points
        self._quad_rule_1D \
            = quadrature_rule.create_quadrature_rule_1D(order)
        gauss_pts_1D = self._quad_rule_1D.xigauss
        self._shape_func_1D = interpolants.shape1d(gauss_pts_1D)

        self._is_mixed = mixed

        # Mechanical only problems
        if problem_type == "uniaxial_stress":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("bar")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # prescribe ux = 0 on x = 0, uy = 0 on y = 0, and uz = 0 on z = 0
            disp_node = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                if self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 3], dtype=int))
            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.zeros(len(disp_node))

            # normal traction on x = 2
            self._surf_traction_vector = np.array([0.1, 0.0, 0.0])
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 2 * np.ones(3)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction)

        if problem_type == "patch_B":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("cube")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
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

                scaling = 1e-4
                ux = scaling * (2 * x + y + z - 4) / 2
                uy = scaling * (x + 2 * y + z - 4) / 2
                uz = scaling * (x + y + 2 * z - 4) / 2

                self.UUR_true[i, :] = np.array([ux, uy, uz])

                if (x == 0.0 or x == 2.0 or y == 0.0
                    or y == 2.0 or z == 0 or z == 2.0):
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(ux)
                    disp_val.append(uy)
                    disp_val.append(uz)

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no surface tractions
            self._surf_traction_vector = np.array([0.0, 0.0, 0.0])
            self._pres_surf_traction = np.array([])

        if problem_type == "simple_shear":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("cube")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane z = 0, and set uz = 0 on plane z = 2
            disp_node = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                if self._nodal_coords[i][2] == 2.0:
                    disp_node.append(np.array([i, 3], dtype=int))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.zeros(len(disp_node))

            # shear traction in x direction on plane z = 2
            self._surf_traction_vector = np.array([0.1, 0.0, 0.0])
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 2] == 2 * np.ones(3)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction)

        if problem_type == "cook_membrane":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("cook")

            self._dt = 1.
            self._num_steps = 10
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            # fix z displacments on plane z = 0
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                elif self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
            if mixed:
                disp_node.append(np.array([0, 4], dtype = int))
                disp_val.append(np.zeros(self._num_steps))
            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # vertical traction on x = 48
            self._surf_traction_vector = np.zeros((self._num_steps, 3))
            self._surf_traction_vector[:, 1] = 2.0 * self._dt * np.arange(1, self._num_steps + 1)
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 48 * np.ones(3)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction)

        if problem_type == "vert_beam":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("vert_beam")

            self._dt = 1.
            self._num_steps = 10
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane z = 0
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
            if mixed:
                disp_node.append(np.array([0, 4], dtype = int))
                disp_val.append(np.zeros(self._num_steps))
            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # diagonal traction on z = 5
            trac_increment = 1.0
            self._surf_traction_vector = np.zeros((self._num_steps, self._ndim))
            self._surf_traction_vector[:, 0] = trac_increment * self._dt * np.arange(1, self._num_steps + 1)
            self._surf_traction_vector[:, 1] = trac_increment * self._dt * np.arange(1, self._num_steps + 1)
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 2] == 5 * np.ones(3)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction)

        if problem_type == "hole_block_disp_rigid":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("hole_block_half")

            self._dt = 1.
            self._num_steps = 100
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            # set incremental displacements on plane x = 1
            increment = 0.0005
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                elif self._nodal_coords[i][0] == 1.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.linspace(increment * self._dt,
                                                increment * self._dt * self._num_steps,
                                                self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                elif self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
            if mixed:
                disp_node.append(np.array([0, 4], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

        if problem_type == "hole_block_disp_sliding":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("hole_block_quarter")

            self._dt = 1.
            self._num_steps = 50
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            # set incremental displacements on plane x = 1
            increment = 0.0005
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if np.abs(self._nodal_coords[i][0]) < 1.e-5:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                
                if self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 0.5:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.linspace(increment * self._dt,
                                                increment * self._dt * self._num_steps,
                                                self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 4], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

        if problem_type == "notched_bar":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("notched_bar")

            self._dt = 1.
            self._num_steps = 20
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            # set incremental displacements on plane x = 1
            increment = 0.0003
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                x = self._nodal_coords[i][0]
                y = self._nodal_coords[i][1]
                z = self._nodal_coords[i][2]
                if x == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))

                if x == 4.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.linspace(increment * self._dt,
                                                increment * self._dt * self._num_steps,
                                                self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                
                if y == 0.0 and (x == 0.0 or x == 4.0):
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 4], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

        if problem_type == "hole_block_disp_sliding_2D":
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_2D = shape_func_tri

            self._ndim = 2
            self._mesh = Mesh("hole_block_2D")

            self._dt = 1.
            self._num_steps = 15
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 3 dofs per node if there are pressure dofs
            self._dof_node = 2 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 3
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 2

            # fix x displacement on x = 0
            # fix y displacement on y = 0
            # set incremental x displacements on x = 1
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 1.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.linspace(0.0001 * self._dt,
                                                0.0001 * self._dt * self._num_steps,
                                                self._num_steps))
            if mixed:
                disp_node.append(np.array([0, 3], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

        if problem_type == "hole_block_traction":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("hole_block_half")

            self._dt = 1.
            self._num_steps = 40
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            # set incremental displacements on plane x = 1
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 4], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # normal traction on plane x = 1
            self._surf_traction_vector = np.zeros((self._num_steps, self._ndim))
            self._surf_traction_vector[:, 0] = 4.5 * self._dt * np.arange(1, self._num_steps + 1)
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 1 * np.ones(3)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction, dtype=int)

        if problem_type == "hole_block_traction_2D":
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_2D = shape_func_tri

            self._ndim = 2
            self._mesh = Mesh("hole_block_2D")

            self._dt = 1.
            self._num_steps = 30
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 3 dofs per node if there are pressure dofs
            self._dof_node = 2 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 3
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 2

            # fix x displacement on x = 0
            # fix y displacement on y = 0
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
            if mixed:
                disp_node.append(np.array([0, 3], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # normal traction on x = 1
            self._surf_traction_vector = np.zeros((self._num_steps, self._ndim))
            self._surf_traction_vector[:, 0] = 1.0 * self._dt * np.arange(1, self._num_steps + 1)
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 1 * np.ones(2)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction, dtype=int)

        if problem_type == "boat_fender":

            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("boat_fender")

            self._dt = 1.
            self._num_steps = 29
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 3 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            # set incremental displacements on plane x = 1
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 4], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # normal traction on plane x = 1
            increment = 0.001
            self._surf_traction_vector = np.zeros((self._num_steps, self._ndim))
            self._surf_traction_vector[:, 0] = increment * self._dt * np.arange(1, self._num_steps + 1)
            self._surf_traction_vector[:, 1] = -2 * increment * self._dt * np.arange(1, self._num_steps + 1)
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 1] == 0.5 * np.ones(3)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction, dtype=int)

        # Thermomechanical problems
        if problem_type == "rectangular_wall":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("rect_prism")

            self._dt = 0.1
            self._num_steps = 20
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 5 dofs per node if there are pressure dofs
            self._dof_node = 4 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane y = 0
            # fix temperature to 300K on plane y = 0
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    # fix displacements
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    #fix temperatures
                    disp_node.append(np.array([i, 4], dtype=int))
                    disp_val.append(300. * np.ones(self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 5], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # prescribe convection bcs on x = 0, x = 2, y = 1, z = 0, and z = 0.1
            pres_surf_flux = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 2 * np.ones(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 0] == np.zeros(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 1] == 1. * np.ones(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 2] == np.zeros(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 2] == 0.1 * np.ones(3)).all():
                    pres_surf_flux.append(surface)
            self._pres_surf_flux = np.array(pres_surf_flux, dtype=int)

            # prescribe initial conditions
            self._init_temp = np.zeros(self._num_nodes)
            for i, node in enumerate(self._nodal_coords):
                self._init_temp[i] = 300. + 500. * node[1]

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

        if problem_type == "rectangular_wall_hex":
            self._quad_rule_3D = quad_rule_brick
            self._quad_rule_2D = quad_rule_square
            self._shape_func_3D = shape_func_brick
            self._shape_func_2D = shape_func_square

            self._ndim = 3
            self._mesh = Mesh("rect_prism_brick")

            self._dt = 0.05
            self._num_steps = 20
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 5 dofs per node if there are pressure dofs
            self._dof_node = 4 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 8
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 4

            # fix all nodes on plane y = 0
            # fix temperature to 300K on plane y = 0
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    # fix displacements
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    #fix temperatures
                    disp_node.append(np.array([i, 4], dtype=int))
                    disp_val.append(300. * np.ones(self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 5], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # prescribe convection bcs on x = 0, x = 2, y = 1, z = 0, and z = 0.1
            pres_surf_flux = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 2 * np.ones(4)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 0] == np.zeros(4)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 1] == 1. * np.ones(4)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 2] == np.zeros(4)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 2] == 0.5 * np.ones(4)).all():
                    pres_surf_flux.append(surface)
            self._pres_surf_flux = np.array(pres_surf_flux, dtype=int)

            # prescribe initial conditions
            self._init_temp = np.zeros(self._num_nodes)
            for i, node in enumerate(self._nodal_coords):
                self._init_temp[i] = 300. + 500. * node[1]

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

        if problem_type == "hole_block_traction_thermoplastic":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("hole_block_half")

            self._dt = 0.05
            self._num_steps = 36
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 4 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # uy = 0 on plane y = 0 
            # ux = uz = 0 on plane x = 0
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 5], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # normal traction on plane x = 1
            self._surf_traction_vector = np.zeros((self._num_steps, self._ndim))
            self._surf_traction_vector[:, 0] = 5.0 * np.arange(1, self._num_steps + 1)
            pres_surf_traction = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 1 * np.ones(3)).all():
                    pres_surf_traction.append(surface)
            self._pres_surf_traction = np.array(pres_surf_traction, dtype=int)

            # prescribe initial conditions
            self._init_temp = 300.0 * np.ones(self._num_nodes)

            # no convection BC
            self._pres_surf_flux = None
        
        if problem_type == "hole_block_disp_thermoplastic":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("hole_block_quarter")

            self._dt = 0.1
            self._num_steps = 50
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 4 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # fix all nodes on plane x = 0
            # set incremental displacements on plane x = 1
            increment = 0.0005
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if np.abs(self._nodal_coords[i][0]) < 1.e-5:
                    self._nodal_coords[i][0] = 0.0
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                
                if self._nodal_coords[i][2] == 0.0:
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 0.5:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.linspace(increment,
                                                increment * self._num_steps,
                                                self._num_steps))

            if mixed:
                disp_node.append(np.array([0, 5], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

            # prescribe initial conditions
            self._init_temp = 300.0 * np.ones(self._num_nodes)

            # no convection BC
            self._pres_surf_flux = None
        
        if problem_type == "uniaxial_stress_thermoplastic":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("dogbone_quarter")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            self._dt = 1.
            self._num_steps = 60
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 4 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # prescribe ux = 0 on x = 0, uy = 0 on y = 0, and uz = 0 on z = 0
            increment = 0.002
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):
                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                elif self._nodal_coords[i][1] == 1.0:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_node.append(np.array([i, 3], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))
                    disp_val.append(np.linspace(increment,
                                                increment * self._num_steps,
                                                self._num_steps))
                    disp_val.append(np.zeros(self._num_steps))
                else:
                    if self._nodal_coords[i][0] == 0.1425:
                        disp_node.append(np.array([i, 1], dtype=int))
                        disp_val.append(np.zeros(self._num_steps))
                    
                    if self._nodal_coords[i][2] == 0.0:
                        disp_node.append(np.array([i, 3], dtype=int))
                        disp_val.append(np.zeros(self._num_steps))
            if mixed:
                disp_node.append(np.array([0, 5], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

            # prescribe initial conditions
            self._init_temp = 300.0 * np.ones(self._num_nodes)

            # no convection BC
            self._pres_surf_flux = None
        
        if problem_type == "uniaxial_plane_strain_thermoplastic":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("rect_sheet_defect")

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            self._dt = 0.1
            self._num_steps = 13
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            # 4 dofs per node if there are pressure dofs
            self._dof_node = 4 + mixed
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # prescribe ux = 0 on x = 0, uy = 0 on y = 0, and uz = 0 on z = 0
            increment = 0.001
            disp_node = []
            disp_val = []
            for i in range(self._num_nodes):

                # set all z displacements to 0 for plane strain
                disp_node.append(np.array([i, 3], dtype=int))
                disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][1] == 0.0:
                    disp_node.append(np.array([i, 2], dtype=int))
                    disp_val.append(np.zeros(self._num_steps))

                if self._nodal_coords[i][0] == 0.5:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(np.linspace(increment,
                                                increment * self._num_steps,
                                                self._num_steps))
                
                if self._nodal_coords[i][0] == -0.5:
                    disp_node.append(np.array([i, 1], dtype=int))
                    disp_val.append(-np.linspace(increment,
                                                increment * self._num_steps,
                                                self._num_steps))
            if mixed:
                disp_node.append(np.array([0, 5], dtype = int))
                disp_val.append(np.zeros(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

            # prescribe initial conditions
            self._init_temp = 300.0 * np.ones(self._num_nodes)

            # no convection BC
            self._pres_surf_flux = None
        # Thermo only problems
        if problem_type == "rectangular_wall_thermo":
            self._quad_rule_3D = quad_rule_tetra
            self._quad_rule_2D = quad_rule_tri
            self._shape_func_3D = shape_func_tetra
            self._shape_func_2D = shape_func_tri

            self._ndim = 3
            self._mesh = Mesh("rect_prism")

            self._dt = 0.1
            self._num_steps = 1000
            self._times = np.linspace(self._dt,
                                      self._dt * self._num_steps,
                                      self._num_steps)

            self._nodal_coords = self._mesh.get_nodal_coordinates()
            self._volume_conn = self._mesh.get_volume_connectivity()
            self._surface_conn = self._mesh.get_surface_connectivity()

            self._dof_node = 1
            self._num_nodes = len(self._nodal_coords)
            self._num_nodes_elem = 4
            self._num_elem = len(self._volume_conn)
            self._num_nodes_surf = 3

            # dirichlet temperature BCs
            disp_node = []
            disp_val = []
            for i, node in enumerate(self._nodal_coords):
                if node[1] == 0.0:
                    disp_node.append(i)
                    disp_val.append(300. * np.ones(self._num_steps))

            self._disp_node = np.array(disp_node, dtype=int)
            self._disp_val = np.array(disp_val)

            pres_surf_flux = []
            for surface in self._surface_conn:
                surface_points = self._nodal_coords[surface, :]
                if (surface_points[:, 0] == 2 * np.ones(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 0] == np.zeros(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 1] == 1. * np.ones(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 2] == np.zeros(3)).all():
                    pres_surf_flux.append(surface)
                if (surface_points[:, 2] == 0.1 * np.ones(3)).all():
                    pres_surf_flux.append(surface)
            self._pres_surf_flux = np.array(pres_surf_flux, dtype=int)

            # prescribe initial conditions
            self._init_temp = np.zeros(self._num_nodes)
            for i, node in enumerate(self._nodal_coords):
                self._init_temp[i] = 300. + 500. * node[1]

            # no tractions
            self._surf_traction_vector = None
            self._pres_surf_traction = None

    def get_surface_basis_functions(self):
        if self._ndim == 2:
            return self._quad_rule_1D, self._shape_func_1D
        else:
            return self._quad_rule_2D, self._shape_func_2D

    def get_volume_basis_functions(self):
        if self._ndim == 2:
            return self._quad_rule_2D, self._shape_func_2D
        else:
            return self._quad_rule_3D, self._shape_func_3D

    def get_mesh_properties(self):
        return self._dof_node, self._num_nodes, self._num_nodes_elem, \
            self._num_elem, self._num_nodes_surf, self._nodal_coords, \
            self._volume_conn, self._ndim

    def get_boundary_conditions(self):
        return self._disp_node, self._disp_val, \
            self._pres_surf_traction, self._surf_traction_vector

    def get_convection_boundary_conditions(self):
        return self._pres_surf_flux

    def get_initial_temp(self):
        return self._init_temp

    def save_data(self, filename, point_data, cell_data=None):
        points = self._mesh.get_nodal_coordinates()
        cells = self._mesh.get_cells()
        with meshio.xdmf.TimeSeriesWriter(filename) as writer:
            writer.write_points_cells(points, cells)
            for i, time in enumerate(self._times):
                if cell_data != None:
                    writer.write_data(time, point_data=point_data[i],
                                      cell_data=cell_data[i])
                else:
                    writer.write_data(time, point_data=point_data[i])

    def is_mixed(self):
        return self._is_mixed

    def num_steps(self):
        return self._num_steps, self._dt