import meshio
import pygmsh

class Mesh():

    def __init__(self, mesh_type):

        self._mesh_type = mesh_type
        with pygmsh.occ.Geometry() as geom:

            # square prism with hole through center
            if self._mesh_type == "hole_block":

                geom.characteristic_length_min = 0.03
                geom.characteristic_length_max = 0.03

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 1.0, 1.0)

                disk = geom.add_disk([0.5, 0.5, 0.0], 0.25)

                flat = geom.boolean_difference(
                    rectangle,
                    disk
                )
                geom.extrude(flat, [0.0, 0.0, 0.0625], num_layers=5)
                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]

            if self._mesh_type == "hole_block_half":

                geom.characteristic_length_min = 0.03
                geom.characteristic_length_max = 0.03

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 1.0, 0.5)

                disk = geom.add_disk([0.5, 0.0, 0.0], 0.25)

                flat = geom.boolean_difference(
                    rectangle,
                    disk
                )
                geom.extrude(flat, [0.0, 0.0, 0.0625], num_layers=5)
                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]
            
            if self._mesh_type == "notched_bar":
                geom.characteristic_length_min = 0.035
                geom.characteristic_length_max = 0.035

                width = 1.0
                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 4.0, width)

                disk1 = geom.add_disk([1.0, 0.0, 0.0], 0.5)
                disk2 = geom.add_disk([3.0, 0.0, 0.0], 0.5)
                disk3 = geom.add_disk([2.0, width, 0.0], 0.5)

                flat = geom.boolean_difference(
                    rectangle,
                    [disk1, disk2, disk3]
                )
                geom.extrude(flat, [0.0, 0.0, 0.0625], num_layers=5)
                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]

            if self._mesh_type == "hole_block_2D":

                geom.characteristic_length_min = 0.02
                geom.characteristic_length_max = 0.02

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 1.0, 0.5)

                disk = geom.add_disk([0.5, 0.0, 0.0], 0.25)

                flat = geom.boolean_difference(
                    rectangle,
                    disk
                )
                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[1].data
                self._surface_conn = self._cells[0].data
                self._mesh.cells = [self._cells[1]]

            # thin beam
            if self._mesh_type == "bar":

                geom.characteristic_length_min = 0.1
                geom.characteristic_length_max = 0.1

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 2, 0.5)

                geom.extrude(rectangle, [0.0, 0.0, 0.5], num_layers=5)
                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]

            # 2 x 2 x 2 cube
            if self._mesh_type == "cube":

                geom.characteristic_length_min = 0.2
                geom.characteristic_length_max = 0.2

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 2, 2)

                geom.extrude(rectangle, [0.0, 0.0, 2], num_layers=10)
                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]

            if self._mesh_type == "cook":

                geom.characteristic_length_min = 0.6
                geom.characteristic_length_max = 0.6

                beam = geom.add_polygon([[0.0, 0.0],
                                         [48., 44.],
                                         [48., 60.],
                                         [0.0, 44.]])

                geom.extrude(beam, [0.0, 0.0, 1.0], num_layers=1)
                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]

            if self._mesh_type == "vert_beam":

                geom.characteristic_length_min = 0.1
                geom.characteristic_length_max = 0.1

                beam = geom.add_polygon([[0.0, 0.0],
                                         [1.0, 0.0],
                                         [1.0, 1.0],
                                         [0.0, 1.0]])

                geom.extrude(beam, [0.0, 0.0, 5.0], num_layers=40)

                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]

            if self._mesh_type == "rect_prism":

                geom.characteristic_length_min = 0.05
                geom.characteristic_length_max = 0.05

                beam = geom.add_polygon([[0.0, 0.0],
                                         [2.0, 0.0],
                                         [2.0, 1.0],
                                         [0.0, 1.0]])

                geom.extrude(beam, [0.0, 0.0, 0.1], num_layers=3)

                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]

            if self._mesh_type == "rect_prism_brick":
                # Define corner points of the base rectangle
                p0 = geom.add_point([0.0, 0.0, 0.0])
                p1 = geom.add_point([2.0, 0.0, 0.0])
                p2 = geom.add_point([2.0, 1.0, 0.0])
                p3 = geom.add_point([0.0, 1.0, 0.0])

                # Create lines and loop
                l0 = geom.add_line(p0, p1)
                l1 = geom.add_line(p1, p2)
                l2 = geom.add_line(p2, p3)
                l3 = geom.add_line(p3, p0)

                loop = geom.add_curve_loop([l0, l1, l2, l3])
                surface = geom.add_plane_surface(loop)

                mesh_size = 0.08
                num_nodes_length = int(2.0 / mesh_size) + 1
                num_nodes_width = int(1.0 / mesh_size) + 1

                # Transfinite meshing for structured quad elements on base
                geom.set_transfinite_curve(l0, num_nodes_length, mesh_type="Progression", coeff=1.0)
                geom.set_transfinite_curve(l1, num_nodes_width, mesh_type="Progression", coeff=1.0)
                geom.set_transfinite_curve(l2, num_nodes_length, mesh_type="Progression", coeff=1.0)
                geom.set_transfinite_curve(l3, num_nodes_width, mesh_type="Progression", coeff=1.0)
                geom.set_transfinite_surface(surface, arrangement="left", corner_pts=[p0, p1, p2, p3])
                geom.set_recombined_surfaces([surface])  # Make it quad

                # Extrude to get 3D hex mesh
                geom.extrude(
                    surface,
                    [0, 0, 0.5],
                    num_layers=7,
                    recombine=True,  # Make it hexahedral
                )

                self._mesh = geom.generate_mesh()

                self._points = self._mesh.points
                self._cells = self._mesh.cells
                self._volume_conn = self._cells[2].data
                self._surface_conn = self._cells[1].data
                self._mesh.cells = [self._cells[2]]


    def get_nodal_coordinates(self):
        return self._points

    def get_cells(self):
        return self._mesh.cells

    def get_volume_connectivity(self):
        return self._volume_conn

    def get_surface_connectivity(self):
        return self._surface_conn

    def add_point_data(self, data):
        self._mesh.point_data = data

    def add_cell_data(self, data):
        self._mesh.cell_data = data

    def save_mesh(self, filename):
        self._mesh.write(filename + ".vtk")
