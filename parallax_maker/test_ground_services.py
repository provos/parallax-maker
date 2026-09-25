"""Ground-plane UI helpers: auto fit, horizon-row settings, side profile."""

import io
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from .camera import Camera
from .controller import AppState
from .e2e_support import create_fixture_state
from .ground_services import (
    GroundNotReady,
    fit_ground,
    mask_top_row,
    nearest_bottom_card_depth,
    scene_profile,
)
from .slice import ImageSlice

W, H = 320, 240


def _slice(rows, depth, cols=slice(None)):
    image = np.zeros((H, W, 4), np.uint8)
    image[rows, cols] = (10, 20, 30, 255)
    return ImageSlice(image, depth)


def _state(slices):
    state = AppState()
    state.imgData = Image.new("RGB", (W, H))
    state.image_slices = slices
    state._camera = Camera(distance=100, max_distance=500, focal_length=50)
    return state


def test_fit_puts_the_horizon_on_the_mask_top_and_the_ground_under_the_nearest_foot():
    sky = _slice(slice(0, 140), 0)
    ocean = _slice(slice(150, None), 60)
    ocean.is_ground_plane = True
    speck = ocean.image.copy()
    speck[100, 5] = 255  # a stray pixel above the edge is ignored
    ocean.image = speck
    far_rock = _slice(slice(200, None), 40, cols=slice(0, 30))  # reaches the bottom
    cliff = _slice(
        slice(120, None), 215, cols=slice(200, None)
    )  # nearer, reaches the bottom
    floating = _slice(slice(50, 90), 250)  # nearest, but doesn't reach the bottom
    state = _state([sky, ocean, far_rock, cliff, floating])

    assert mask_top_row(ocean.image) == 150
    fit = fit_ground(state)
    assert fit.horizon_row == 150
    assert state.camera.pitch_for_horizon(150, W, H) == pytest.approx(fit.pitch)
    assert fit.ground_near == pytest.approx(cliff._depth_to_z(215, state.camera))
    assert nearest_bottom_card_depth(state) == pytest.approx(fit.ground_near)


def test_fit_needs_a_ground_slice():
    with pytest.raises(GroundNotReady):
        fit_ground(_state([_slice(slice(0, None), 0)]))


@pytest.mark.parametrize("image", [None, np.zeros((H, W, 4), np.uint8)])
def test_fit_needs_a_ground_slice_with_content(image):
    ground = _slice(slice(0, None), 60)
    ground.image = image
    ground.is_ground_plane = True
    with pytest.raises(GroundNotReady, match="empty"):
        fit_ground(_state([ground]))


def test_scene_profile_describes_cards_camera_and_ground():
    sky = _slice(slice(0, 130), 0)
    ocean = _slice(slice(125, None), 60)
    ocean.is_ground_plane = True
    cliff = _slice(slice(150, None), 215)
    state = _state([sky, ocean, cliff])

    profile = scene_profile(state)
    assert profile.camera_z == -100
    assert [c.index for c in profile.cards] == [0, 2]
    far, near = profile.cards
    assert far.z == pytest.approx(500) and near.z < far.z
    assert far.top < 0 < far.bottom  # the frame spans above and below the camera
    ground = profile.ground
    assert ground.height == pytest.approx(state.camera.ground_height(W, H))
    assert ground.far_z == pytest.approx(500)
    assert ground.near_z == pytest.approx(state.camera.ground_near, abs=1e-3)
    assert ground.backdrop_top is not None and ground.backdrop_top < ground.height


# --- API ---------------------------------------------------------------------------


def _restore_fixture(client) -> dict:
    state_path = create_fixture_state(
        Path.cwd(), state_name=f"appstate-e2e-fit-{uuid4().hex[:8]}"
    )
    with open(state_path, "rb") as fh:
        response = client.post(
            "/api/v1/projects/restore",
            data={"state": (fh, "appstate.json")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    return response.get_json()


def test_api_fit_ground(client):
    view = _restore_fixture(client)
    project_id = view["id"]
    response = client.post(f"/api/v1/projects/{project_id}/ground/fit")
    assert response.status_code == 409

    client.put(
        f"/api/v1/projects/{project_id}/slices/1/ground", json={"isGround": True}
    )
    raw = client.get(f"/api/v1/projects/{project_id}/slices/1/download").data
    expected_top = mask_top_row(np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA")))

    body = client.post(f"/api/v1/projects/{project_id}/ground/fit").get_json()
    assert body["settings"]["camera"]["horizonRow"] == pytest.approx(
        expected_top, abs=1e-6
    )
    logs = client.get(f"/api/v1/projects/{project_id}/logs?after=0").get_json()[
        "entries"
    ]
    assert logs[-1]["message"].startswith("Fitted the ground plane: horizon at row")
    assert body["sceneProfile"]["ground"] is not None


def test_api_horizon_row_sets_the_pitch(client):
    view = _restore_fixture(client)
    project_id = view["id"]
    camera = {
        k: view["settings"]["camera"][k]
        for k in ("distance", "focalLength", "maxDistance")
    }
    row = 0.6 * view["image"]["height"]

    body = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "horizonRow": row}},
    ).get_json()
    assert body["settings"]["camera"]["horizonRow"] == pytest.approx(row)
    assert body["settings"]["camera"]["pitch"] > 0

    response = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "focalLength": 5.0, "horizonRow": 1e6}},
    )
    assert response.status_code == 400
    assert view["sceneProfile"]["cameraZ"] == -camera["distance"]
    assert len(view["sceneProfile"]["cards"]) == len(view["slices"])
