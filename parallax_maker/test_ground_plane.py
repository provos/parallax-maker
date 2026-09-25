"""Ground plane: geometry, rendering, export, persistence and API.

The hard requirement: with a ground plane, the reference camera (default
position and orientation) still reproduces the original image.
"""

import io
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from .camera import Camera
from .controller import AppState
from .e2e_support import create_fixture_state
from .scene import build_layers, ground_extent, ground_layers, project, render_scene
from .slice import ImageSlice
from .test_gltf_reprojection import (
    TOLERANCE_NDC,
    _expected_ndc,
    _node_matrix,
    _project_cards,
    _read_accessor,
)

W, H = 320, 240


def _camera(**kwargs):
    return Camera(distance=100, max_distance=500, focal_length=50, **kwargs)


def _layer_image(rows=None, color=None, rng=None):
    """An RGBA slice: opaque (random or ``color``) on ``rows``, else clear."""
    image = np.zeros((H, W, 4), np.uint8)
    rows = slice(None) if rows is None else rows
    if color is None:
        image[rows, :, :3] = rng.integers(0, 256, image[rows, :, :3].shape)
    else:
        image[rows, :, :3] = color
    image[rows, :, 3] = 255
    return image


def _ground_slice(image):
    ground = ImageSlice(image, 128)
    ground.is_ground_plane = True
    return ground


# --- Geometry -------------------------------------------------------------------


def test_ground_height_follows_the_bottom_row_and_needs_a_visible_ground():
    cam = _camera()
    fl_px = cam.focal_length_px(W)
    # Level camera: the bottom row (H/2 below the axis) meets the ground at
    # the camera distance when ground_near is 0.
    assert cam.ground_height(W, H) == pytest.approx((H / 2) * 100 / fl_px, rel=1e-4)
    cam.ground_near = 150.0
    assert cam.ground_height(W, H) == pytest.approx((H / 2) * 250 / fl_px, rel=1e-4)

    looking_up = _camera(pitch=cam.pitch_for_horizon(H + 5, W, H))
    with pytest.raises(ValueError):
        looking_up.ground_height(W, H)


@pytest.mark.parametrize("pitch", [0.0, 8.0, -6.0])
def test_ground_is_horizontal_and_covers_its_image_rows(pitch):
    cam = _camera(pitch=pitch)
    rng = np.random.default_rng(1)
    horizon = cam.horizon_row(W, H)
    image = _layer_image(rows=slice(int(horizon) + 10, None), rng=rng)
    layer = ground_layers(image, cam)[0]

    # Horizontal: every corner at camera height.
    assert np.ptp(layer.corners[:, 1]) == pytest.approx(0, abs=1e-3)
    assert layer.corners[0, 1] == pytest.approx(cam.ground_height(W, H), rel=1e-4)
    # Its far edge is the slice's first visible row or the max distance,
    # whichever is nearer; it projects back exactly.
    _, far_row, _ = ground_extent(image, cam)
    assert layer.source_quad[0, 1] == pytest.approx(max(int(horizon) + 10, far_row))
    np.testing.assert_allclose(
        project(layer.corners, cam, cam.reference_position(), W, H),
        layer.source_quad,
        atol=1e-2,
    )


def test_ground_ends_at_the_max_distance_and_a_backdrop_holds_the_far_band():
    cam = _camera(pitch=4.0)
    everything = np.full((H, W, 4), 255, np.uint8)  # ground mask up to the top
    ground, backdrop = ground_layers(everything, cam)
    first_row, far_row, far_z = ground_extent(everything, cam)

    # The ground is bounded: from the bottom edge out to the max distance.
    assert ground.kind == "ground" and backdrop.kind == "backdrop"
    assert ground.corners[:, 2].max() == pytest.approx(far_z, rel=1e-4)
    assert ground.source_quad[0, 1] == pytest.approx(far_row)
    assert far_row > cam.horizon_row(W, H)
    # The backdrop stands upright on the ground's far edge and covers the
    # rows from the slice's first visible row down to that edge.
    assert np.ptp(backdrop.corners[:, 2]) == pytest.approx(0, abs=1e-3)
    assert backdrop.corners[:, 2].max() == pytest.approx(far_z, rel=1e-4)
    assert backdrop.corners[2, 1] == pytest.approx(ground.corners[0, 1], rel=1e-4)
    assert backdrop.source_quad[0, 1] == first_row == 0
    # Every row belongs to one of the two (one row of overlap at the seam).
    boundary = int(np.ceil(far_row))
    assert ground.image[:boundary, :, 3].max() == 0
    assert ground.image[boundary:, :, 3].min() == 255
    assert backdrop.image[: boundary + 1, :, 3].min() == 255
    assert backdrop.image[boundary + 1 :, :, 3].max() == 0

    # A ground mask that starts below the far edge needs no backdrop.
    near_only = np.zeros((H, W, 4), np.uint8)
    near_only[boundary + 5 :, :, 3] = 255
    assert [layer.kind for layer in ground_layers(near_only, cam)] == ["ground"]

    cam.ground_near = cam.max_distance
    with pytest.raises(ValueError):
        ground_layers(everything, cam)


# --- Rendering ------------------------------------------------------------------


def _scene(cam, rng):
    horizon = int(cam.horizon_row(W, H))
    sky = ImageSlice(_layer_image(rows=slice(0, horizon + 8), rng=rng), 0)
    ground = _ground_slice(_layer_image(rows=slice(horizon + 8, None), rng=rng))
    thing = np.zeros((H, W, 4), np.uint8)
    rows = slice(horizon - 40, min(horizon + 50, H))
    thing[rows, 120:190, :3] = rng.integers(0, 256, thing[rows, 120:190, :3].shape)
    thing[rows, 120:190, 3] = 255
    return [sky, ground, ImageSlice(thing, 200)]


def _painted(slices):
    """The original composite: ground at the back, then the cards in order."""
    out = np.zeros((H, W, 3), np.float64)
    for image_slice in sorted(slices, key=lambda s: not s.is_ground_plane):
        alpha = image_slice.image[..., 3:4] / 255.0
        out = out * (1 - alpha) + image_slice.image[..., :3] * alpha
    return out


@pytest.mark.parametrize("pitch", [0.0, 9.0, -5.0])
def test_reference_view_with_a_ground_plane_reproduces_the_image(pitch):
    cam = _camera(pitch=pitch)
    rng = np.random.default_rng(2)
    slices = _scene(cam, rng)

    rendered = render_scene(build_layers(slices, cam), cam, cam.reference_position())

    diff = np.abs(rendered[..., :3].astype(float) - _painted(slices))
    assert diff.max() <= 1.0
    assert diff.mean() < 0.05


def test_the_ground_is_behind_every_card_regardless_of_list_order():
    cam = _camera()
    horizon = cam.horizon_row(W, H)
    far_card = ImageSlice(_layer_image(color=(255, 0, 0)), 0)  # z = 500, opaque
    ground = _ground_slice(
        _layer_image(rows=slice(int(horizon) + 1, None), color=(0, 255, 0))
    )
    # A foreground rock whose card sits behind its own ground contact: its
    # lower pixels are "below" the ground, yet covered it in the original.
    rock = np.zeros((H, W, 4), np.uint8)
    rock[H - 30 :, 40:80] = (0, 0, 255, 255)
    rock_card = ImageSlice(rock, 60)

    for slices in ([far_card, ground, rock_card], [rock_card, far_card, ground]):
        rendered = render_scene(
            build_layers(slices, cam), cam, cam.reference_position()
        )
        assert tuple(rendered[H - 5, 60, :3]) == (0, 0, 255)  # the rock stays in front
        assert tuple(rendered[H - 5, 200, :3]) == (
            255,
            0,
            0,
        )  # the card covers the ground


def test_moving_the_camera_keeps_the_horizon_fixed():
    cam = _camera(pitch=6.0)
    h = cam.ground_height(W, H)
    very_far = np.array([[0.0, h, 1e7]])
    horizon = cam.horizon_row(W, H)
    for offset in ([-5, 0, 0], [0, -4, 0], [3, 2, 10]):
        position = cam.reference_position() + np.array(offset, np.float32)
        row = project(very_far, cam, position, W, H)[0, 1]
        assert row == pytest.approx(horizon, abs=0.1)


# --- glTF -----------------------------------------------------------------------


@pytest.mark.parametrize("pitch", [0.0, 9.0])
def test_exported_ground_is_horizontal_and_reprojects_onto_its_texture(tmp_path, pitch):
    from PIL import Image

    from .gltf import export_gltf

    cam = _camera(pitch=pitch)
    rng = np.random.default_rng(3)
    slices = _scene(cam, rng)
    paths = []
    for i, image_slice in enumerate(slices):
        path = tmp_path / f"slice_{i}.png"
        Image.fromarray(image_slice.image).save(path)
        image_slice.filename = str(path)
        paths.append(path)

    doc = json.loads(
        Path(export_gltf(tmp_path / "scene.gltf", cam, slices, paths)).read_text()
    )

    for uvs, ndc, forward in _project_cards(doc):
        assert (forward > 0).all()
        np.testing.assert_allclose(ndc, _expected_ndc(uvs), atol=TOLERANCE_NDC)

    ground_node = [n for n in doc["nodes"] if "mesh" in n][1]
    ground, backdrop = doc["meshes"][ground_node["mesh"]]["primitives"]

    def world_of(primitive):
        positions = _read_accessor(doc, primitive["attributes"]["POSITION"])
        return _node_matrix(ground_node) @ np.c_[positions, np.ones(len(positions))].T

    # Horizontal in glTF (y up): constant height, below the camera, out to
    # the max distance; the backdrop is upright at that distance.
    world = world_of(ground)
    assert np.ptp(world[1]) == pytest.approx(0, abs=1e-3)
    assert world[1, 0] == pytest.approx(-cam.ground_height(W, H), rel=1e-4)
    assert world[2].max() == pytest.approx(cam.max_distance, rel=1e-4)
    assert np.ptp(world_of(backdrop)[2]) == pytest.approx(0, abs=1e-3)
    # Its node is the farthest one, so renderers sorting transparent objects
    # by node position draw it first (behind the cards).
    card_nodes = [n for n in doc["nodes"] if "mesh" in n and n is not ground_node]
    assert ground_node["translation"][2] > max(n["translation"][2] for n in card_nodes)


# --- Persistence and API ----------------------------------------------------------


def test_ground_flag_survives_a_json_round_trip():
    state = AppState()
    state.filename = str(Path.cwd() / "appstate-ground-test")
    state.imgThresholds = [0, 128, 255]
    state.image_slices = [
        ImageSlice(depth=0, filename=f"{state.filename}/image_slice_0.png"),
        ImageSlice(depth=128, filename=f"{state.filename}/image_slice_1.png"),
    ]
    state.image_slices[1].is_ground_plane = True
    state.camera.ground_near = 42.0

    restored = AppState.from_json(state.to_json())
    assert [s.is_ground_plane for s in restored.image_slices] == [False, True]
    assert restored.camera.ground_near == 42.0

    del_ground = json.loads(state.to_json())
    del del_ground["ground_plane_slice"]
    legacy = AppState.from_json(json.dumps(del_ground))
    assert not any(s.is_ground_plane for s in legacy.image_slices)


def _restore_fixture(client) -> dict:
    state_path = create_fixture_state(
        Path.cwd(), state_name=f"appstate-e2e-ground-{uuid4().hex[:8]}"
    )
    with open(state_path, "rb") as fh:
        response = client.post(
            "/api/v1/projects/restore",
            data={"state": (fh, "appstate.json")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    return response.get_json()


def test_api_marks_one_ground_slice_and_persists_it(client):
    view = _restore_fixture(client)
    project_id = view["id"]
    assert [s["isGround"] for s in view["slices"]] == [False] * len(view["slices"])

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/1/ground", json={"isGround": True}
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    assert [s["isGround"] for s in body["slices"]] == [False, True, False]

    # Marking another slice moves the ground; repeating is a no-op.
    body = client.put(
        f"/api/v1/projects/{project_id}/slices/0/ground", json={"isGround": True}
    ).get_json()
    assert [s["isGround"] for s in body["slices"]] == [True, False, False]
    again = client.put(
        f"/api/v1/projects/{project_id}/slices/0/ground", json={"isGround": True}
    ).get_json()
    assert again["changed"] is False

    logs = client.get(f"/api/v1/projects/{project_id}/logs?after=0").get_json()[
        "entries"
    ]
    assert any(e["message"] == "Slice 0 is now the ground plane" for e in logs)

    # Navigation renders with the ground plane.
    moved = client.post(
        f"/api/v1/projects/{project_id}/camera/navigate", json={"direction": "up"}
    )
    assert moved.status_code == 200

    client.post(f"/api/v1/projects/{project_id}/save")
    raw = client.get(f"/api/v1/projects/{project_id}/state-file").data
    restored = client.post(
        "/api/v1/projects/restore",
        data={"state": (io.BytesIO(raw), "appstate.json")},
        content_type="multipart/form-data",
    ).get_json()
    assert [s["isGround"] for s in restored["slices"]] == [True, False, False]


def test_api_rejects_a_ground_plane_without_a_visible_ground(client):
    view = _restore_fixture(client)
    project_id = view["id"]
    camera = {
        k: view["settings"]["camera"][k]
        for k in ("distance", "focalLength", "maxDistance")
    }

    width, height = view["image"]["width"], view["image"]["height"]
    cam = Camera(focal_length=camera["focalLength"])
    horizon_below = cam.pitch_for_horizon(height + 20, width, height)
    client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "pitch": horizon_below}},
    )

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/1/ground", json={"isGround": True}
    )
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"
    assert "horizon" in response.get_json()["error"]["message"]

    # With a ground plane marked, a pitch that hides the ground is rejected.
    client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "pitch": 0.0}},
    )
    assert (
        client.put(
            f"/api/v1/projects/{project_id}/slices/1/ground", json={"isGround": True}
        ).status_code
        == 200
    )
    response = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "pitch": horizon_below}},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"

    # ...and so is a ground that would start beyond the max distance.
    response = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "groundNear": camera["maxDistance"] + 1}},
    )
    assert response.status_code == 400


def test_settings_expose_ground_near_and_horizon_row(client):
    view = _restore_fixture(client)
    project_id = view["id"]
    camera = {
        k: view["settings"]["camera"][k]
        for k in ("distance", "focalLength", "maxDistance")
    }

    body = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "groundNear": 25.0, "pitch": 4.0}},
    ).get_json()
    settings_camera = body["settings"]["camera"]
    assert settings_camera["groundNear"] == 25.0
    expected = Camera(focal_length=camera["focalLength"], pitch=4.0).horizon_row(
        body["image"]["width"], body["image"]["height"]
    )
    assert settings_camera["horizonRow"] == pytest.approx(expected)


def test_exported_ground_texture_stays_registered_between_vertices():
    """Linear interpolation inside each grid cell must stay within half a
    pixel of the true (projective) mapping, even right below the horizon."""
    from .gltf import GROUND_SUBDIVISIONS, ground_grids

    width, height = 1024, 768
    cam = Camera(distance=100, max_distance=500, focal_length=26)
    cam.pitch = cam.pitch_for_horizon(0.672 * height, width, height)
    image = np.zeros((height, width, 4), np.uint8)
    image[int(0.672 * height) + 1 :, :, 3] = 255

    n = GROUND_SUBDIVISIONS
    (grid, *_), z_node = ground_grids(cam, image, width, height, n)
    vertices, uvs, _ = grid
    world = vertices.astype(np.float64)
    world[:, 2] = z_node - world[:, 2]
    grid = world.reshape(n + 1, n + 1, 3)
    texture = (uvs * [width, height]).reshape(n + 1, n + 1, 2)
    centers = ((grid[:-1, :-1] + grid[1:, 1:]) / 2).reshape(-1, 3)
    interpolated = ((texture[:-1, :-1] + texture[1:, 1:]) / 2).reshape(-1, 2)

    error = np.abs(
        project(centers, cam, cam.reference_position(), width, height) - interpolated
    )
    assert error.max() < 0.5
