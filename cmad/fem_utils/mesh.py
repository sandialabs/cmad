import meshio
import pygmsh

class Mesh():

    def __init__(self, mesh_type):

        self._mesh_type = mesh_type
        with pygmsh.occ.Geometry() as geom:

            # square prism with hole through center
            if self._mesh_type == "hole_block":

                geom.characteristic_length_min = 0.05
                geom.characteristic_length_max = 0.05

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 1.0, 1.0)

                disk = geom.add_disk([0.5, 0.5, 0.0], 0.25)

                flat = geom.boolean_difference(
                    rectangle,
                    disk
                )
                geom.extrude(flat, [0.0, 0.0, 0.25], num_layers=10)
                self._mesh = geom.generate_mesh()

            # thin beam
            if self._mesh_type == "bar":

                geom.characteristic_length_min = 0.1
                geom.characteristic_length_max = 0.1

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 2, 0.5)

                geom.extrude(rectangle, [0.0, 0.0, 0.5], num_layers=5)
                self._mesh = geom.generate_mesh()

            # 2 x 2 x 2 cube
            if self._mesh_type == "cube":

                geom.characteristic_length_min = 0.2
                geom.characteristic_length_max = 0.2

                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], 2, 2)

                geom.extrude(rectangle, [0.0, 0.0, 2], num_layers=10)
                self._mesh = geom.generate_mesh()
            
            if self._mesh_type == "cook":
                
                geom.characteristic_length_min = 1.5
                geom.characteristic_length_max = 1.5

                beam = geom.add_polygon([[0.0, 0.0],
                                         [48., 44.],
                                         [48., 60.],
                                         [0.0, 44.]])
                
                geom.extrude(beam, [0.0, 0.0, 1.5], num_layers=1)
                self._mesh = geom.generate_mesh()

        self._points = self._mesh.points
        self._cells = self._mesh.cells
        self._volume_conn = self._cells[2].data
        self._surface_conn = self._cells[1].data


    def get_nodal_coordinates(self):
        return self._points

    def get_volume_connectivity(self):
        return self._volume_conn

    def get_surface_connectivity(self):
        return self._surface_conn

    def add_point_data(self, data):
        self._mesh.point_data = data

    def save_mesh(self, filename):
        self._mesh.write(filename + ".vtk")
