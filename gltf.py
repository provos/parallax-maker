# (c) 2024 Niels Provos
#
# This file contains functions for creating and exporting glTF files.
# We generate a glTF file representing a 3D scene with a camera, cards, and image slices.
# The resulting file can be opened in a 3D application like Blender, Houdini or Unreal.
#
import base64
import numpy as np
import pygltflib as gltf


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


def create_camera(gltf_obj, focal_length, aspect_ratio, translation, rotation_quarternion):
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
            yfov=2*np.arctan(sensor_height / focal_length),
            znear=0.01,
            zfar=10000
        )
    )
    gltf_obj.cameras.append(camera)

    # Create the camera node
    camera_node = gltf.Node(
        translation=translation,
        rotation=rotation_quarternion,
        camera=camera_index
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
    tmp_buffer_index = len(gltf_obj.buffers)-1

    tmp_buffer_view = gltf.BufferView(
        buffer=tmp_buffer_index,
        byteOffset=0,
        byteLength=data.nbytes,
        target=target
    )
    gltf_obj.bufferViews.append(tmp_buffer_view)
    tmp_buffer_view_index = len(gltf_obj.bufferViews)-1

    return tmp_buffer_view_index


def export_gltf(output_path, aspect_ratio, focal_length, card_corners_3d_list, image_paths):
    """
    Export the camera, cards, and image slices to a glTF file.

    Args:
        output_path (str): The path to save the glTF file.
        aspect_ratio (float): The aspect ratio of the camera.
        focal_length (float): The focal length of the camera.
        card_corners_3d_list (list): List of 3D corner coordinates for each card.
        image_paths (list): List of file paths for each image slice.
    """
    # Create a new glTF object
    gltf_obj = gltf.GLTF2(
        scene=0
    )

    # Create the scene
    scene = gltf.Scene()
    gltf_obj.scenes.append(scene)

    camera_index = create_camera(gltf_obj,
                                 focal_length,
                                 aspect_ratio,
                                 [0, 0, -100], rotation_quaternion_y(180))
    # Add the camera node to the scene
    scene.nodes.append(camera_index)

    # Create the card objects (planes)
    for i, corners_3d in enumerate(card_corners_3d_list):
        # Set the vertices and indices for the plane
        
        # negate the y coordinates of corners_3d
        vertices = np.array(corners_3d, dtype=np.float32)
        vertices[:, 1] = -vertices[:, 1]
        
        tex_coords = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        indices = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.uint32)

        # Create the buffer and buffer view for vertices
        vertex_bufferview_index = create_buffer_and_view(
            gltf_obj, vertices, target=gltf.ARRAY_BUFFER)

        # Create the buffer and buffer view for texture coordinates
        tex_coord_bufferview_index = create_buffer_and_view(
            gltf_obj, tex_coords, target=gltf.ARRAY_BUFFER)

        # Create the buffer and buffer view for indices
        index_bufferview_index = create_buffer_and_view(
            gltf_obj, indices, target=gltf.ELEMENT_ARRAY_BUFFER)

        # Create the accessor for texture coordinates
        tex_coord_accessor = gltf.Accessor(
            bufferView=tex_coord_bufferview_index,
            componentType=gltf.FLOAT,
            count=len(tex_coords),
            type=gltf.VEC2,
            max=tex_coords.max(axis=0).tolist(),
            min=tex_coords.min(axis=0).tolist()
        )
        gltf_obj.accessors.append(tex_coord_accessor)
        tex_coord_accessor_index = len(gltf_obj.accessors)-1

        # Create the accessor for vertices
        vertex_accessor = gltf.Accessor(
            bufferView=vertex_bufferview_index,
            componentType=gltf.FLOAT,
            count=len(vertices),
            type=gltf.VEC3,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist()
        )
        gltf_obj.accessors.append(vertex_accessor)
        vertex_accessor_index = len(gltf_obj.accessors)-1

        # Create the accessor for indices
        index_accessor = gltf.Accessor(
            bufferView=index_bufferview_index,
            componentType=gltf.UNSIGNED_INT,
            count=indices.size,
            type=gltf.SCALAR
        )
        gltf_obj.accessors.append(index_accessor)
        index_accessor_index = len(gltf_obj.accessors)-1

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
                    material=i)
            ])
        gltf_obj.meshes.append(mesh)

        # Create the material and assign the texture
        material = gltf.Material(
            name=f"Material_{i}",
            pbrMetallicRoughness=gltf.PbrMetallicRoughness(
                baseColorTexture=gltf.TextureInfo(
                    index=i
                )
            ),
            # Set the emissive color (RGB values)
            emissiveFactor=[1.0, 1.0, 1.0],
            emissiveTexture=gltf.TextureInfo(
                index=i
            ),
            alphaMode="MASK",
            alphaCutoff=0.5,
            doubleSided=True
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
        )
        gltf_obj.nodes.append(card_node)
        scene.nodes.append(len(gltf_obj.nodes)-1)

    # Save the glTF file
    gltf_obj.convert_images(gltf.ImageFormat.DATAURI)
    gltf_obj.save(str(output_path / "model.gltf"))