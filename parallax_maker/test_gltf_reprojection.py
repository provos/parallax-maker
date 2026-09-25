"""Reprojection tests for the glTF export.

Every exported vertex, projected through the exported camera using the glTF
2.0 conventions (camera looks down its local -Z with +Y up; column-vector
node transforms), must land exactly where its texture coordinate says it
came from in the original image. This one property catches a flipped axis,
a wrong field of view, rounded card depths and displacement that drifts off
the camera rays: through the exported camera, the scene must reproduce the
original image.
"""

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .camera import Camera
from .gltf import export_gltf
from .slice import ImageSlice

TOLERANCE_NDC = 1e-3


def _quat_to_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _node_matrix(node):
    matrix = np.eye(4)
    matrix[:3, :3] = _quat_to_matrix(node.get("rotation", [0, 0, 0, 1]))
    matrix[:3, 3] = node.get("translation", [0, 0, 0])
    return matrix


def _read_accessor(doc, index):
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    uri = doc["buffers"][view["buffer"]]["uri"]
    data = base64.b64decode(uri.split(",", 1)[1])
    width = {"VEC2": 2, "VEC3": 3}[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    values = np.frombuffer(
        data[start : start + accessor["count"] * width * 4], np.float32
    )
    return values.reshape(-1, width).astype(np.float64)


def _export_scene(tmp_path, width, height, cam, depths, displacement_scale, depth_map):
    slices, image_paths, depth_paths = [], [], []
    for i, depth in enumerate(depths):
        image = np.zeros((height, width, 4), np.uint8)
        image[..., 3] = 255
        path = tmp_path / f"slice_{i}.png"
        Image.fromarray(image).save(path)
        slices.append(ImageSlice(image, depth, str(path)))
        image_paths.append(path)
        if depth_map is not None:
            depth_path = tmp_path / f"depth_{i}.png"
            Image.fromarray(depth_map).save(depth_path)
            depth_paths.append(depth_path)
    out = export_gltf(
        tmp_path / "scene.gltf",
        cam,
        slices,
        image_paths,
        depth_paths,
        displacement_scale=displacement_scale,
    )
    return json.loads(Path(out).read_text())


def _project_cards(doc):
    """Yields (uv, ndc, depth_in_front) per card, via the exported camera."""
    camera_node = next(node for node in doc["nodes"] if "camera" in node)
    perspective = doc["cameras"][camera_node["camera"]]["perspective"]
    view = np.linalg.inv(_node_matrix(camera_node))
    tan_half = np.tan(perspective["yfov"] / 2)
    aspect = perspective["aspectRatio"]
    for node in doc["nodes"]:
        if "mesh" not in node:
            continue
        primitive = doc["meshes"][node["mesh"]]["primitives"][0]
        positions = _read_accessor(doc, primitive["attributes"]["POSITION"])
        uvs = _read_accessor(doc, primitive["attributes"]["TEXCOORD_0"])
        world = _node_matrix(node) @ np.c_[positions, np.ones(len(positions))].T
        in_camera = (view @ world).T
        forward = -in_camera[:, 2]
        ndc = np.stack(
            [
                in_camera[:, 0] / forward / (tan_half * aspect),
                in_camera[:, 1] / forward / tan_half,
            ],
            axis=1,
        )
        yield uvs, ndc, forward


def _expected_ndc(uvs):
    # glTF UV (0, 0) is the image's top-left; NDC y points up.
    return np.stack([2 * uvs[:, 0] - 1, 1 - 2 * uvs[:, 1]], axis=1)


@pytest.mark.parametrize(
    "width, height, focal_length, distance",
    [
        (1600, 1200, 100, 100),  # landscape
        (480, 856, 50, 100),  # portrait (example/input.png)
        (1920, 1080, 35, 250),  # wide angle, farther camera
    ],
)
def test_flat_cards_reproject_onto_their_texture(
    tmp_path, width, height, focal_length, distance
):
    cam = Camera(distance=distance, max_distance=500, focal_length=focal_length)
    doc = _export_scene(
        tmp_path,
        width,
        height,
        cam,
        [200, 120, 30],
        displacement_scale=0.0,
        depth_map=None,
    )

    cards = list(_project_cards(doc))
    assert len(cards) == 3
    for uvs, ndc, forward in cards:
        assert (forward > 0).all(), "card is behind the camera"
        # Each card fills the frame exactly: UV corners land on NDC corners,
        # top-left on top-left (no flipped or mirrored axis).
        np.testing.assert_allclose(ndc, _expected_ndc(uvs), atol=TOLERANCE_NDC)


def test_displaced_cards_still_reproject_onto_their_texture(tmp_path):
    width, height = 320, 240
    cam = Camera(distance=100, max_distance=500, focal_length=100)
    # Non-uniform relief: a diagonal ramp plus a bump, 0..255.
    ys, xs = np.mgrid[0:height, 0:width]
    relief = 0.5 * xs / width + 0.5 * np.exp(
        -((xs - width / 2) ** 2 + (ys - height / 3) ** 2) / 800.0
    )
    depth_map = np.clip(relief * 255, 0, 255).astype(np.uint8)

    doc = _export_scene(
        tmp_path,
        width,
        height,
        cam,
        [200, 30],
        displacement_scale=20.0,
        depth_map=depth_map,
    )

    for uvs, ndc, forward in _project_cards(doc):
        # Relief actually moved vertices toward the camera...
        assert np.ptp(forward) > 5
        # ...but only along their camera rays, so the image is unchanged.
        np.testing.assert_allclose(ndc, _expected_ndc(uvs), atol=TOLERANCE_NDC)


def test_displacement_never_reaches_or_passes_the_camera():
    from .gltf import displace_vertices

    vertices = np.array([[-10.0, 5.0, 0.0], [10.0, -5.0, 0.0]], np.float32)
    depth_map = np.ones((4, 4), np.float32)
    displaced = displace_vertices(
        vertices.copy(), depth_map, displacement_scale=500.0, camera_distance=100.0
    )
    assert (displaced[:, 2] < 100.0).all()
    # Still in front of the camera: x/y keep their sign (no mirroring).
    assert (np.sign(displaced[:, :2]) == np.sign(vertices[:, :2])).all()

    with pytest.raises(ValueError):
        displace_vertices(vertices.copy(), depth_map, 1.0, camera_distance=0.0)


@pytest.mark.parametrize("pitch", [10.0, -12.0])
@pytest.mark.parametrize("displacement_scale", [0.0, 20.0])
def test_pitched_scene_reprojects_onto_its_texture(tmp_path, pitch, displacement_scale):
    width, height = 480, 360
    cam = Camera(distance=100, max_distance=500, focal_length=50, pitch=pitch)
    depth_map = None
    if displacement_scale:
        ys, xs = np.mgrid[0:height, 0:width]
        depth_map = (255 * (0.3 + 0.7 * ys / height)).astype(np.uint8)

    doc = _export_scene(
        tmp_path,
        width,
        height,
        cam,
        [200, 30],
        displacement_scale=displacement_scale,
        depth_map=depth_map,
    )

    for uvs, ndc, forward in _project_cards(doc):
        assert (forward > 0).all()
        np.testing.assert_allclose(ndc, _expected_ndc(uvs), atol=TOLERANCE_NDC)

    if not displacement_scale:
        # Cards stay vertical in the world (orthogonal to a future ground):
        # every vertex of a flat card has the same world z.
        for node in doc["nodes"]:
            if "mesh" in node:
                primitive = doc["meshes"][node["mesh"]]["primitives"][0]
                positions = _read_accessor(doc, primitive["attributes"]["POSITION"])
                world = _node_matrix(node) @ np.c_[positions, np.ones(len(positions))].T
                assert np.ptp(world[2]) == pytest.approx(0, abs=1e-3)
