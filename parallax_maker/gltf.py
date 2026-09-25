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


#: Grid resolution for pitched (trapezoidal) cards without displacement: the
#: image-to-card mapping is projective, so a textured quad needs a grid to
#: keep the texture registered (exact at vertices, sub-pixel in between).
PITCHED_CARD_SUBDIVISIONS = 64


def quaternion_from_matrix(matrix):
    """Unit quaternion (x, y, z, w) for a 3x3 rotation matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    quaternion = np.array([x, y, z, w])
    return (quaternion / np.linalg.norm(quaternion)).tolist()


# The scene's world frame (x right, y down, z forward; see camera.py) maps to
# glTF (y up) by a 180 degree turn about z; a glTF camera looks down its local
# -z with +y up, i.e. an OpenCV camera frame flipped in y and z.
_WORLD_TO_GLTF = np.diag([-1.0, -1.0, 1.0])
_GLTF_CAMERA_TO_CV_CAMERA = np.diag([1.0, -1.0, -1.0])


def camera_node_rotation(cam):
    """glTF camera-node rotation for ``cam``'s orientation (its pitch).

    With no pitch this is the 180 degree turn about y the exporter has always
    used (the camera looks down +z towards the cards).
    """
    rotation = (
        _WORLD_TO_GLTF @ cam.rotation_camera_to_world() @ _GLTF_CAMERA_TO_CV_CAMERA
    )
    return quaternion_from_matrix(rotation)


def create_camera(
    gltf_obj,
    focal_length,
    aspect_ratio,
    translation,
    rotation_quarternion,
    sensor_width=35.0,
):
    """
    Creates a camera in the glTF object with the specified parameters.

    Args:
        gltf_obj (gltf.Gltf): The glTF object to add the camera to.
        focal_length (float): The focal length of the camera.
        aspect_ratio (float): The aspect ratio of the camera.
        translation (List[float]): The translation of the camera node.
        rotation_quarternion (List[float]): The rotation of the camera node as a quaternion.
        sensor_width (float, optional): The sensor width in mm. Defaults to 35.0.

    Returns:
        int: The index of the created camera.

    """
    camera_index = len(gltf_obj.cameras)

    sensor_height = sensor_width / aspect_ratio

    # Create the camera object
    camera = gltf.Camera(
        type="perspective",
        name=f"Camera_{camera_index}",
        perspective=gltf.Perspective(
            aspectRatio=aspect_ratio,
            # Pinhole: the half-angle's tangent is half the sensor over f.
            yfov=2 * np.arctan(sensor_height / (2 * focal_length)),
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


def displace_vertices(
    vertices, depth_map, displacement_scale=10.0, camera_distance=None, uvs=None
):
    """
    Displaces the vertices of a plane based on a depth map.

    Args:
        vertices (numpy.ndarray): The 3D corner coordinates of the plane, in the
            card's local frame (plane at z=0, displacement along +z).
        depth_map (numpy.ndarray): The depth map to displace the vertices with. Normalized to [0, 1].
        camera_distance (float, optional): Distance from the camera to the plane,
            with the camera on the local +z axis at (0, 0, camera_distance). When
            given, each vertex moves along its camera ray instead of straight
            along z, so it still projects to the same image point (and keeps
            lining up with its texture and the other cards).
        uvs (numpy.ndarray, optional): Per-vertex texture coordinates to sample
            the depth map at; required for non-rectangular (pitched) cards.
            Defaults to the vertices' normalized x/y extent.

    Returns:
        numpy.ndarray: The displaced vertices.
    """
    # Get the dimensions of the depth map
    depth_map_width, depth_map_height = depth_map.shape

    if uvs is not None:
        tex_coords = np.asarray(uvs, dtype=np.float64)
    else:
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
    if camera_distance is not None:
        if camera_distance <= 0:
            raise ValueError("camera_distance must be positive")
        # Never reach (or pass) the camera, which would collapse or mirror
        # the card; stop just short of it instead.
        depths = np.minimum(depths, 0.99 * camera_distance)
        # Slide along the ray from the camera through the vertex: moving
        # `depth` closer scales x/y by (D - depth) / D.
        scale = (camera_distance - depths) / camera_distance
        vertices[:, 0] *= scale
        vertices[:, 1] *= scale
    vertices[:, 2] = depths

    return vertices


def card_grid(cam, z, image_width, image_height, subdivisions):
    """Vertex grid (card-local: plane at z=0) and UVs for the card at depth ``z``.

    Row-major (subdivisions + 1)^2 grid over the image; every vertex is where
    the reference camera's ray through its image point meets the card plane,
    and its UV is that image point.
    """
    us = np.linspace(0, image_width, subdivisions + 1)
    vs = np.linspace(0, image_height, subdivisions + 1)
    grid_u, grid_v = np.meshgrid(us, vs)
    points = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
    vertices = cam.backproject_to_depth(points, z, image_width, image_height)
    vertices[:, 2] -= z
    uvs = (points / [image_width, image_height]).astype(np.float32)
    return vertices, uvs


def create_card(
    gltf_obj,
    i,
    corners_3d,
    subdivisions=300,
    depth_map=None,
    displacement_scale=0.0,
    camera_distance=None,
    uvs=None,
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
        camera_distance (float, optional): Distance from the camera to the card
            plane; displacement then follows camera rays (see displace_vertices).
        uvs (numpy.ndarray, optional): When given, ``corners_3d`` is already a
            full (n+1) x (n+1) row-major vertex grid and ``uvs`` its texture
            coordinates (used for pitched cards, whose texture mapping is
            projective); no further subdivision happens.

    Returns:
        int: The index of the created mesh.
    """
    # Set the vertices and indices for the plane

    # negate the y coordinates of corners_3d
    vertices = np.array(corners_3d, dtype=np.float32)
    vertices[:, 1] = -vertices[:, 1]

    if uvs is not None:
        tex_coords = np.array(uvs, dtype=np.float32)
        if displacement_scale > 0.0 and depth_map is not None:
            vertices = displace_vertices(
                vertices,
                depth_map,
                displacement_scale=displacement_scale,
                camera_distance=camera_distance,
                uvs=tex_coords,
            )
    else:
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
                vertices,
                depth_map,
                displacement_scale=displacement_scale,
                camera_distance=camera_distance,
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
        camera_node_rotation(cam),
        sensor_width=cam.sensor_width,
    )
    # Add the camera node to the scene
    scene.nodes.append(camera_index)

    subdivisions = 500

    alpha_mode = "MASK" if support_dof else "BLEND"

    # Create the card objects (planes)
    for i, image_slice in enumerate(image_slices):
        corners_3d = image_slice.create_card(image_height, image_width, cam)
        # Translaton hack so that we can put the depth on the node
        z_transform = float(corners_3d[0][2])
        corners_3d[:, 2] -= z_transform
        uvs = None

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

        if cam.pitch != 0:
            # A pitched card is a trapezoid whose texture mapping is
            # projective: build it as a grid of back-projected image points,
            # each with its exact image position as UV.
            displaced = displacement_scale > 0.0 and depth_map is not None
            corners_3d, uvs = card_grid(
                cam,
                z_transform,
                image_width,
                image_height,
                subdivisions if displaced else PITCHED_CARD_SUBDIVISIONS,
            )

        mesh = create_card(
            gltf_obj,
            i,
            corners_3d,
            subdivisions,
            depth_map,
            displacement_scale=displacement_scale,
            camera_distance=z_transform + camera_distance,
            uvs=uvs,
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
            translation=[0, 0, z_transform],
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
