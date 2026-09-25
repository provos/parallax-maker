from __future__ import annotations

import io

from PIL import Image

from ._api_test_helpers import upload_fixture_image
from .e2e_support import create_input_image


def test_upload_returns_a_created_project_view(client) -> None:
    view = upload_fixture_image(client)

    assert view["id"].startswith("appstate-")
    assert view["revision"] == 1
    assert view["image"] == {"width": 320, "height": 240}
    assert view["numSlices"] == 3
    assert view["thresholds"] == []
    assert view["slices"] == []
    assert view["busy"] is None
    assert view["assets"]["depth"] is None
    assert view["assets"]["input"]["url"].startswith(
        f"/api/v1/projects/{view['id']}/assets/input"
    )


def test_upload_requires_a_multipart_image_field(client) -> None:
    response = client.post(
        "/api/v1/projects", data={}, content_type="multipart/form-data"
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_upload_rejects_a_non_image_file(client) -> None:
    response = client.post(
        "/api/v1/projects",
        data={"image": (io.BytesIO(b"not an image"), "input.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_input_asset_is_retrievable_immediately_after_upload(client) -> None:
    view = upload_fixture_image(client)

    response = client.get(f"/api/v1/projects/{view['id']}/assets/input")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers.get("ETag")
    decoded = Image.open(io.BytesIO(response.data))
    assert decoded.size == (320, 240)


def test_input_asset_supports_conditional_get(client) -> None:
    view = upload_fixture_image(client)

    first = client.get(f"/api/v1/projects/{view['id']}/assets/input")
    etag = first.headers["ETag"]

    second = client.get(
        f"/api/v1/projects/{view['id']}/assets/input",
        headers={"If-None-Match": etag},
    )

    assert second.status_code == 304


def test_get_unknown_project_is_404(client) -> None:
    response = client.get("/api/v1/projects/appstate-does-not-exist")

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


def test_unknown_asset_id_is_404(client) -> None:
    view = upload_fixture_image(client)

    response = client.get(f"/api/v1/projects/{view['id']}/assets/not-a-real-asset")

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


def test_slice_asset_before_slices_exist_is_404(client) -> None:
    view = upload_fixture_image(client)

    response = client.get(f"/api/v1/projects/{view['id']}/assets/slice-0")

    assert response.status_code == 404


def test_asset_path_traversal_attempt_is_404(client) -> None:
    view = upload_fixture_image(client)

    for asset_id in (
        "..%2F..%2Fetc%2Fpasswd",
        "slice--1",
        "slice-abc",
        "input%2F..%2F..%2Fetc",
    ):
        response = client.get(f"/api/v1/projects/{view['id']}/assets/{asset_id}")
        assert response.status_code == 404, asset_id


def test_resolve_asset_path_rejects_a_slice_file_outside_the_project_dir(
    tmp_path,
) -> None:
    """Defense in depth: even a corrupted/adversarial slice filename can't escape."""

    from .api.assets import resolve_asset_path
    from .api.errors import NotFound
    from .controller import AppState
    from .slice import ImageSlice

    outside = tmp_path.parent / "outside-secret.png"
    outside.write_bytes(b"not really a png")

    state = AppState()
    state.filename = "appstate-escape"
    state.imgData = create_input_image()
    state.image_slices = [ImageSlice(image=None, depth=1, filename=str(outside))]

    project_dir = tmp_path / state.filename
    project_dir.mkdir()

    try:
        resolve_asset_path(project_dir, state, "slice-0")
        raised = False
    except NotFound:
        raised = True

    assert raised, "a slice filename escaping the project directory must be rejected"
