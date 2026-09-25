"""HTTP tests for the manual 3D-camera-navigation route.

Reproduces the removed Dash ``navigate_image`` callback's behavior
(``camera_services.navigate_camera``) over the ``/api/v1`` contract, using the
same fixture state and fake models as the other API test modules
(``parallax_maker.e2e_support.create_fixture_state``/``create_fake_runtime``)
exercised directly through the Flask test client.
"""

from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from ._api_test_helpers import upload_fixture_image
from .controller import AppState
from .e2e_support import create_fixture_state


def _restore_fixture(client) -> dict:
    state_name = f"appstate-e2e-restore-{uuid4().hex[:8]}"
    state_path = create_fixture_state(Path.cwd(), state_name=state_name)
    with open(state_path, "rb") as fh:
        response = client.post(
            "/api/v1/projects/restore",
            data={"state": (fh, "appstate.json")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _select(client, project_id, index) -> dict:
    response = client.put(
        f"/api/v1/projects/{project_id}/selection", json={"slice": index}
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _navigate(client, project_id, direction) -> tuple[int, dict]:
    response = client.post(
        f"/api/v1/projects/{project_id}/camera/navigate",
        json={"direction": direction},
    )
    return response.status_code, response.get_json()


def _logs_text(client, project_id) -> str:
    response = client.get(f"/api/v1/projects/{project_id}/logs")
    assert response.status_code == 200, response.get_json()
    return "\n".join(entry["message"] for entry in response.get_json()["entries"])


def _last_log_message(client, project_id) -> str:
    response = client.get(f"/api/v1/projects/{project_id}/logs")
    assert response.status_code == 200, response.get_json()
    entries = response.get_json()["entries"]
    assert entries, "expected at least one log entry"
    return entries[-1]["message"]


# --- direction -> unit vector -------------------------------------------------


_UNIT_STEPS = {
    "up": (0.0, -1.0, 0.0),
    "down": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "out": (0.0, 0.0, -1.0),
    "in": (0.0, 0.0, 1.0),
}


def test_each_direction_moves_the_camera_by_its_unit_vector(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    for direction, step in _UNIT_STEPS.items():
        before = np.array(
            AppState.from_cache(project_id).camera.camera_position, dtype=np.float32
        )
        status, body = _navigate(client, project_id, direction)
        assert status == 200, body
        assert body["changed"] is True

        state = AppState.from_cache(project_id)
        expected = before + np.array(step, dtype=np.float32)
        np.testing.assert_allclose(state.camera.camera_position, expected)
        # Each move is a fresh (3,) float32 array, not an in-place mutation
        # of the array `before` was read from (see camera_services.py's
        # docstring on why Dash's own `+=` is intentionally not reproduced).
        assert state.camera.camera_position.dtype == np.float32
        assert state.camera.camera_position.shape == (3,)


def test_reset_returns_to_the_default_position(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    status, body = _navigate(client, project_id, "right")
    assert status == 200, body
    status, body = _navigate(client, project_id, "up")
    assert status == 200, body

    status, body = _navigate(client, project_id, "reset")
    assert status == 200, body
    assert body["changed"] is True

    state = AppState.from_cache(project_id)
    expected = np.array([0.0, 0.0, -state.camera.camera_distance], dtype=np.float32)
    np.testing.assert_allclose(state.camera.camera_position, expected)


# --- deselect / main image / log ---------------------------------------------


def test_navigate_deselects_slice_changes_main_image_and_logs(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    selected = _select(client, project_id, 1)
    assert selected["selectedSlice"] == 1
    before_url = selected["mainImage"]["url"]

    status, body = _navigate(client, project_id, "left")

    assert status == 200, body
    assert body["changed"] is True
    assert body["selectedSlice"] is None
    assert body["mainImage"]["url"] != before_url

    state = AppState.from_cache(project_id)
    assert state.selected_slice is None

    log_text = _logs_text(client, project_id)
    assert "Navigated to new camera position" in log_text
    assert str(state.camera.camera_position) in log_text


# --- zero slices ---------------------------------------------------------------


def test_zero_slices_is_unchanged_and_logs_no_slices(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    assert AppState.from_cache(project_id).image_slices == []

    status, body = _navigate(client, project_id, "up")

    assert status == 200, body
    assert body["changed"] is False

    assert _last_log_message(client, project_id) == "No image slices to navigate"


# --- validation ----------------------------------------------------------------


def test_invalid_direction_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    status, body = _navigate(client, project_id, "sideways")

    assert status == 400, body
    assert body["error"]["code"] == "invalid_request"


def test_missing_direction_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/camera/navigate", json={})

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- served asset ----------------------------------------------------------------


def test_main_image_asset_is_served_after_navigate(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    state = AppState.from_cache(project_id)
    width, height = state.imgData.size

    status, body = _navigate(client, project_id, "in")
    assert status == 200, body

    response = client.get(body["mainImage"]["url"])
    assert response.status_code == 200
    image = Image.open(io.BytesIO(response.data))
    assert image.size == (width, height)
