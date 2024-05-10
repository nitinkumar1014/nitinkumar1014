class OBJECT:
    def __init__(self, filename, swap_yz_axes=False):
        """Initialize the OBJECT by loading data from a given .obj file."""
        self.vertices = []
        self.normals = []
        self.uv_coordinates = []
        self.faces = []
        self.materials = {}
        self.current_material = None
        self.load_object(filename, swap_yz_axes)

    def load_object(self, filename, swap_yz_axes):
        """
        Load the object from the specified file.

        Args:
        filename (str): The path to the .obj file to be loaded.
        swap_yz_axes (bool): If True, swaps the Y and Z coordinates.
        """
        try:
            with open(filename, "r") as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    values = line.split()
                    if not values:
                        continue
                    if values[0] == 'v':
                        self.process_vertex(values, swap_yz_axes)
                    elif values[0] == 'vn':
                        self.process_normal(values, swap_yz_axes)
                    elif values[0] == 'vt':
                        self.process_texture_coordinate(values)
                    elif values[0] in ('usemat', 'usemtl'):
                        self.current_material = values[1]
                    elif values[0] == 'mtllib':
                        self.load_materials(values[1])
                    elif values[0] == 'f':
                        self.process_face(values)
        except IOError:
            print(f"Error reading {filename}. File does not exist or is unreadable.")
        except ValueError:
            print("Error parsing the OBJECT file.")

    def process_vertex(self, values, swap_yz_axes):
        """
        Process a vertex line.

        Args:
        values (list of str): The split line from the OBJ file that starts with 'v'.
        swap_yz_axes (bool): If True, swaps the Y and Z coordinates for this vertex.
        """
        vertex = list(map(float, values[1:4]))
        if swap_yz_axes:
            vertex = vertex[0], vertex[2], vertex[1]
        self.vertices.append(vertex)

    def process_normal(self, values, swap_yz_axes):
        """
        Process a normal vector line.

        Args:
        values (list of str): The split line from the OBJ file that starts with 'vn'.
        swap_yz_axes (bool): If True, swaps the Y and Z coordinates for this normal vector.
        """
        normal = list(map(float, values[1:4]))
        if swap_yz_axes:
            normal = normal[0], normal[2], normal[1]
        self.normals.append(normal)

    def process_texture_coordinate(self, values):
        """
        Process a texture coordinate line.

        Args:
        values (list of str): The split line from the OBJ file that starts with 'vt'.
        """
        texture_coordinate = list(map(float, values[1:3]))
        self.uv_coordinates.append(texture_coordinate)

    @staticmethod
    def load_materials(filename):
        """
        (Placeholder) Load material properties from a .mtl file.

        Args:
        filename (str): The path to the .mtl file.
        """
        print(f"Material library {filename} loading not implemented.")

    def process_face(self, values):
        """
        Process a face line.

        Args:
        values (list of str): The split line from the OBJ file that starts with 'f'.
        """
        face = []
        uv_coords = []
        norms = []
        for face_component in values[1:]:
            vertex_data = face_component.split('/')
            face.append(int(vertex_data[0]))
            if len(vertex_data) >= 2 and vertex_data[1]:
                uv_coords.append(int(vertex_data[1]))
            else:
                uv_coords.append(0)

            if len(vertex_data) >= 3 and vertex_data[2]:
                norms.append(int(vertex_data[2]))
            else:
                norms.append(0)
        self.faces.append((face, norms, uv_coords, self.current_material))
