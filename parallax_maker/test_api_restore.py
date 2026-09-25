from __future__ import annotations

import json
import io
from pathlib import Path
from uuid import uuid4

from .e2e_support import create_fixture_state


def _restore(client, filename_json: dict):
    payload = json.dumps(filename_json).encode("utf-8")
    return client.post(
        "/api/v1/projects/restore",
        data={"state": (io.BytesIO(payload), "appstate.json")},
        content_type="multipart/form-data",
    )


def test_restore_accepts_the_legacy_fixture_state(client) -> None:
    state_name = f"appstate-e2e-restore-{uuid4().hex[:8]}"
    state_path = create_fixture_state(Path.cwd(), state_name=state_name)

    with open(state_path, "rb") as fh:
        response = client.post(
            "/api/v1/projects/restore",
            data={"state": (fh, "appstate.json")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    view = response.get_json()
    assert view["id"] == state_name
    assert view["numSlices"] == 3
    assert len(view["thresholds"]) == 4
    assert len(view["slices"]) == 3
    assert view["image"] == {"width": 320, "height": 240}
    assert view["depthModel"] == ""  # depth_estimation_model isn't restored from JSON


def test_restore_rejects_relative_path_traversal(client) -> None:
    response = _restore(client, {"filename": "../x"})

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_restore_rejects_absolute_paths(client) -> None:
    response = _restore(client, {"filename": "/etc/passwd"})

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_restore_rejects_non_appstate_names(client) -> None:
    response = _restore(client, {"filename": "not-an-appstate-dir"})

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_restore_rejects_missing_project_directory(client) -> None:
    response = _restore(client, {"filename": "appstate-does-not-exist-on-disk"})

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


def test_restore_rejects_a_non_json_body(client) -> None:
    response = client.post(
        "/api/v1/projects/restore",
        data={"state": (io.BytesIO(b"not json"), "appstate.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_restore_requires_the_state_field(client) -> None:
    response = client.post(
        "/api/v1/projects/restore", data={}, content_type="multipart/form-data"
    )

    assert response.status_code == 400
