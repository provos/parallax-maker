# (c) 2024 Niels Provos
#
# This file contains functions for creating and exporting glTF files.
# We generate a glTF file representing a 3D scene with a camera, cards, and image slices.
# The resulting file can be opened in a 3D application like Blender, Houdini or Unreal.
#
import base64

import numpy as np
import pygltflib as gltf
from PIL import Image


def rotation_quaternion_y(y_rot_degrees):
    """Calculates the rotation quaternion for a rotation around the y-axis.

    Args:
        y_rot_degrees: The rotation angle in degrees.

    Returns:
        A NumPy array representing the rotation quaternion (x, y, z, w).
    """

    # Convert to radians and half the angle
    theta = np.radians(y_rot_degrees) / 2
    axis = np.array([0, 1, 0])  # Rotation around the y-axis

    quaternion = np.zeros(4)
    quaternion[:3] = axis * np.sin(theta)
    quaternion[3] = np.cos(theta)

    return quaternion.tolist()


def create_camera(
    gltf_obj, focal_length, aspect_ratio, translation, rotation_quarternion
):
    """
    Creates a camera in the glTF object with the specified parameters.

    Args:
        gltf_obj (gltf.Gltf): The glTF object to add the camera to.
        focal_length (float): The focal length of the camera.
        aspect_ratio (float): The aspect ratio of the camera.
        translation (List[float]): The translation of the camera node.
        rotation_quarternion (List[float]): The rotation of the camera node as a quaternion.

    Returns:
        int: The index of the created camera.

    """
    camera_index = len(gltf_obj.cameras)

    sensor_width = 35.0  # Sensor width in mm
    sensor_height = sensor_width / aspect_ratio

    # Create the camera object
    camera = gltf.Camera(
        type="perspective",
        name=f"Camera_{camera_index}",
        perspective=gltf.Perspective(
            aspectRatio=aspect_ratio,
            yfov=2 * np.arctan(sensor_height / focal_length),
            znear=0.01,
            zfar=10000,
        ),
    )
    gltf_obj.cameras.append(camera)

    # Create the camera node
    camera_node = gltf.Node(
        translation=translation, rotation=rotation_quarternion, camera=camera_index
    )
    gltf_obj.nodes.append(camera_node)

    return camera_index


def create_buffer_and_view(gltf_obj, data, target=gltf.ARRAY_BUFFER):
    """
    Creates a buffer and buffer view in a glTF object.

    Args:
        gltf_obj (gltf.Gltf): The glTF object to add the buffer and buffer view to.
        data (numpy.ndarray): The data to be stored in the buffer.
        target (int): The target usage of the buffer view (default: gltf.ARRAY_BUFFER).

    Returns:
        int: the index of the created buffer view.
    """
    tmp_buffer = gltf.Buffer(
        byteLength=data.nbytes,
        uri=f"data:application/octet-stream;base64,{base64.b64encode(data.tobytes()).decode()}",
    )
    gltf_obj.buffers.append(tmp_buffer)
    tmp_buffer_index = len(gltf_obj.buffers) - 1

    tmp_buffer_view = gltf.BufferView(
        buffer=tmp_buffer_index, byteOffset=0, byteLength=data.nbytes, target=target
    )
    gltf_obj.bufferViews.append(tmp_buffer_view)
    tmp_buffer_view_index = len(gltf_obj.bufferViews) - 1

    return tmp_buffer_view_index


def subdivide_geometry(coords, subdivisions, dimension):
    """
    Subdivides a plane into a grid with the specified number of subdivisions. Can handle both 3D and 2D geometries.

    Args:
        coords (numpy.ndarray): The corner coordinates of the geometry (3D for spatial coordinates, 2D for texture coordinates).
        subdivisions (int): The number of subdivisions to create.
        dimension (int): The dimension of the target points (3 for spatial coordinates, 2 for texture coordinates).

    Returns:
        numpy.ndarray: The corner coordinates of the subdivided geometry.
    """
    x = np.linspace(coords[0, 0], coords[1, 0], subdivisions + 1, dtype=np.float32)
    y = np.linspace(coords[0, 1], coords[3, 1], subdivisions + 1, dtype=np.float32)
    x, y = np.meshgrid(x, y)

    if dimension == 3:
        z = np.zeros_like(x)
        points = np.stack([x, y, z], axis=-1)
    elif dimension == 2:
        points = np.stack([x, y], axis=-1)

    return points.reshape(-1, dimension)


def triangle_indices_from_grid(vertices):
    """
    Generates triangle indices for a grid of vertices.

    Args:
        vertices (numpy.ndarray): The 3D corner coordinates of the grid.

    Returns:
        numpy.ndarray: The triangle indices for the grid.
    """
    # Calculate the number of vertices in each row
    row_length = int(np.sqrt(len(vertices)))

    # Create the indices for the triangles
    indices = []
    for i in range(row_length - 1):
        for j in range(row_length - 1):
            # Calculate the indices for the current quad
            tl = i * row_length + j
            tr = tl + 1
            bl = (i + 1) * row_length + j
            br = bl + 1

            # Create the two triangles for the quad
            indices.append([tl, tr, bl])
            indices.append([bl, tr, br])

    return np.array(indices, dtype=np.uint32)


def displace_vertices(vertices, depth_map, displacement_scale=10.0):
    """
    Displaces the vertices of a plane based on a depth map.

    Args:
        vertices (numpy.ndarray): The 3D corner coordinates of the plane.
        depth_map (numpy.ndarray): The depth map to displace the vertices with. Normalized to [0, 1].

    Returns:
        numpy.ndarray: The displaced vertices.
    """
    # Get the dimensions of the depth map
    depth_map_width, depth_map_height = depth_map.shape

    # Calculate the texture coordinates for the vertices
    tex_coords = vertices[:, :2].copy()
    tex_coords[:, 1] = -tex_coords[:, 1]  # flip Y-axis

    # normalize the texture coordinates to [0, 1]
    tex_min_x, tex_min_y = tex_coords.min(axis=0)
    tex_max_x, tex_max_y = tex_coords.max(axis=0)
    tex_coords -= [tex_min_x, tex_min_y]
    tex_coords /= [tex_max_x - tex_min_x, tex_max_y - tex_min_y]

    # Calculate the pixel coordinates for the texture coordinates
    pixel_coords = (tex_coords * [depth_map_height - 1, depth_map_width - 1]).astype(
        int
    )

    # Get the depth values for the pixel coordinates
    depths = depth_map[pixel_coords[:, 1], pixel_coords[:, 0]] * displacement_scale

    # Displace the vertices based on the depth values
    vertices[:, 2] = depths

    return vertices


def create_card(
    gltf_obj, i, corners_3d, subdivisions=300, depth_map=None, displacement_scale=0.0
):
    """
    Creates a card (plane) in the glTF object with the specified parameters.

    Args:
        gltf_obj (gltf.Gltf): The glTF object to add the card to.
        i (int): The index of the card.
        corners_3d (numpy.ndarray): The 3D corner coordinates for the card.
        subdivisions (int, optional): The number of subdivisions for the card. Defaults to 300.
        depth_map (numpy.ndarray, optional): The depth map for the card. Defaults to None.
        displacement_scale (float, optional): The scale of the displacement. Defaults to 0.0.

    Returns:
        int: The index of the created mesh.
    """
    # Set the vertices and indices for the plane

    # negate the y coordinates of corners_3d
    vertices = np.array(corners_3d, dtype=np.float32)
    vertices[:, 1] = -vertices[:, 1]

    # reorder the vertices of the 4 point plane
    tl = vertices[0]
    tr = vertices[1]
    bl = vertices[3]
    br = vertices[2]

    vertices = np.array([tl, tr, bl, br], dtype=np.float32)

    tex_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)

    if displacement_scale > 0.0 and depth_map is not None:
        vertices = subdivide_geometry(vertices, subdivisions, 3)
        vertices = displace_vertices(
            vertices, depth_map, displacement_scale=displacement_scale
        )
        tex_coords = subdivide_geometry(tex_coords, subdivisions, 2)

    indices = triangle_indices_from_grid(vertices)

    # Create the buffer and buffer view for vertices
    vertex_bufferview_index = create_buffer_and_view(
        gltf_obj, vertices, target=gltf.ARRAY_BUFFER
    )

    # Create the buffer and buffer view for texture coordinates
    tex_coord_bufferview_index = create_buffer_and_view(
        gltf_obj, tex_coords, target=gltf.ARRAY_BUFFER
    )

    # Create the buffer and buffer view for indices
    index_bufferview_index = create_buffer_and_view(
        gltf_obj, indices, target=gltf.ELEMENT_ARRAY_BUFFER
    )

    # Create the accessor for texture coordinates
    tex_coord_accessor = gltf.Accessor(
        bufferView=tex_coord_bufferview_index,
        componentType=gltf.FLOAT,
        count=len(tex_coords),
        type=gltf.VEC2,
        max=tex_coords.max(axis=0).tolist(),
        min=tex_coords.min(axis=0).tolist(),
    )
    gltf_obj.accessors.append(tex_coord_accessor)
    tex_coord_accessor_index = len(gltf_obj.accessors) - 1

    # Create the accessor for vertices
    vertex_accessor = gltf.Accessor(
        bufferView=vertex_bufferview_index,
        componentType=gltf.FLOAT,
        count=len(vertices),
        type=gltf.VEC3,
        max=vertices.max(axis=0).tolist(),
        min=vertices.min(axis=0).tolist(),
    )
    gltf_obj.accessors.append(vertex_accessor)
    vertex_accessor_index = len(gltf_obj.accessors) - 1

    # Create the accessor for indices
    index_accessor = gltf.Accessor(
        bufferView=index_bufferview_index,
        componentType=gltf.UNSIGNED_INT,
        count=indices.size,
        type=gltf.SCALAR,
    )
    gltf_obj.accessors.append(index_accessor)
    index_accessor_index = len(gltf_obj.accessors) - 1

    card_name = f"Card_{i}"

    # Create the mesh for the plane
    mesh = gltf.Mesh(
        name=card_name,
        primitives=[
            gltf.Primitive(
                attributes=gltf.Attributes(
                    POSITION=vertex_accessor_index,
                    TEXCOORD_0=tex_coord_accessor_index,
                ),
                indices=index_accessor_index,
                material=i,
            )
        ],
    )

    return mesh


def export_gltf(
    output_path,
    cam,
    image_slices,
    image_paths,
    depth_paths=[],
    displacement_scale=0.0,
    inline_images=True,
    support_dof=False,
):
    """
    Export the camera, cards, and image slices to a glTF file.

    Args:
        output_path (str): The path to save the glTF file.
        aspect_ratio (float): The aspect ratio of the camera.
        focal_length (float): The focal length of the camera.
        camera_distance (float): The distance of the camera from the origin.
        cam (Camera): The camera object for the scene.
        image_slices (list): List of 3D corner coordinates for each card.
        image_paths (list): List of file paths for each image slice.
        depth_paths (list, optional): List of file paths for each depth map. Defaults to [].
        displacement_scale (float, optional): The scale of the displacement. Defaults to 0.0.
        inline_images (bool, optional): Whether to inline the images in the glTF file. Defaults to True.
    """

    # compute pre-requisites
    image_height, image_width = image_slices[0].image.shape[:2]
    camera_matrix = cam.camera_matrix(image_width, image_height)
    aspect_ratio = float(camera_matrix[0, 2]) / camera_matrix[1, 2]
    focal_length = cam.focal_length
    camera_distance = cam.camera_distance

    # Create a new glTF object
    gltf_obj = gltf.GLTF2(scene=0)

    # Create the scene
    scene = gltf.Scene()
    gltf_obj.scenes.append(scene)

    camera_index = create_camera(
        gltf_obj,
        focal_length,
        aspect_ratio,
        [0, 0, -camera_distance],
        rotation_quaternion_y(180),
    )
    # Add the camera node to the scene
    scene.nodes.append(camera_index)

    subdivisions = 500

    alpha_mode = "MASK" if support_dof else "BLEND"

    # Create the card objects (planes)
    for i, image_slice in enumerate(image_slices):
        corners_3d = image_slice.create_card(image_height, image_width, cam)
        # Translaton hack so that we can put the depth on the node
        z_transform = corners_3d[0][2]
        corners_3d[:, 2] -= z_transform

        depth_map = None
        if len(depth_paths) > i:
            depth_map = Image.open(depth_paths[i])
            width, height = depth_map.size
            depth_map = depth_map.resize(
                (subdivisions + 1, subdivisions + 1), Image.BICUBIC
            )
            depth_map = depth_map.resize((width, height), Image.BICUBIC)
            depth_map = np.array(depth_map)
            depth_map = depth_map.astype(np.float32) / 255.0

        mesh = create_card(
            gltf_obj,
            i,
            corners_3d,
            subdivisions,
            depth_map,
            displacement_scale=displacement_scale,
        )
        gltf_obj.meshes.append(mesh)

        # Create the material and assign the texture
        material = gltf.Material(
            name=f"Material_{i}",
            pbrMetallicRoughness=gltf.PbrMetallicRoughness(
                baseColorTexture=gltf.TextureInfo(index=i)
            ),
            # Set the emissive color (RGB values)
            emissiveFactor=[1.0, 1.0, 1.0],
            emissiveTexture=gltf.TextureInfo(index=i),
            alphaMode=alpha_mode,
            alphaCutoff=0.5 if alpha_mode == "MASK" else None,
            doubleSided=True,
        )

        image = gltf.Image(uri=str(image_paths[i]))
        gltf_obj.images.append(image)

        texture = gltf.Texture(
            source=i,
        )
        gltf_obj.textures.append(texture)

        gltf_obj.materials.append(material)

        # Create the card node and add it to the scene
        card_node = gltf.Node(
            mesh=i,
            translation=[0, 0, int(z_transform)],
            rotation=rotation_quaternion_y(180),
        )
        gltf_obj.nodes.append(card_node)
        scene.nodes.append(len(gltf_obj.nodes) - 1)

    # Save the glTF file
    if inline_images:
        gltf_obj.convert_images(gltf.ImageFormat.DATAURI)
    else:
        gltf_obj.convert_images(gltf.ImageFormat.FILE)

    gltf_obj.save(str(output_path))

    return str(output_path)
