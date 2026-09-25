"""HTTP tests for slice-editing and mask-tool routes.

Mirrors e2e/slice-editing.spec.ts's scenarios over the ``/api/v1`` contract,
using the same fixture state and fake models as the browser tests
(``parallax_maker.e2e_support.create_fixture_state``/``create_fake_runtime``)
but exercised directly through the Flask test client.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import io

import numpy as np

from PIL import Image

from ._api_test_helpers import poll_job, upload_fixture_image
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


def _click(client, project_id, x, y, *, mode="depth", shift=False, ctrl=False) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/segmentation/click",
        json={"x": x, "y": y, "mode": mode, "shiftKey": shift, "ctrlKey": ctrl},
    )
    assert response.status_code == 202, response.get_json()
    return poll_job(client, response.get_json()["job"]["id"])


def _select(client, project_id, index) -> dict:
    response = client.put(
        f"/api/v1/projects/{project_id}/selection", json={"slice": index}
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


# --- create slice ------------------------------------------------------------


def test_create_slice_from_mask_appends_and_selects(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    job = _click(client, project_id, 80, 96)
    assert job["status"] == "succeeded"

    response = client.post(f"/api/v1/projects/{project_id}/slices/create", json={})

    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    assert len(body["slices"]) == 4
    assert body["selectedSlice"] is not None
    state = AppState.from_cache(project_id)
    assert state.slice_mask is None  # cleared by the post-create refresh


def test_create_slice_with_no_mask_is_empty_at_default_depth(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/slices/create", json={})

    assert response.status_code == 200
    body = response.get_json()
    assert len(body["slices"]) == 4
    new_index = body["selectedSlice"]
    assert body["slices"][new_index]["depth"] == 127


def test_create_slice_requires_an_uploaded_image(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    AppState.from_cache(project_id).imgData = None

    response = client.post(f"/api/v1/projects/{project_id}/slices/create", json={})

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


# --- delete slice --------------------------------------------------------------


def test_delete_slice_removes_it_and_clears_selection(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 1)

    response = client.delete(f"/api/v1/projects/{project_id}/slices/1")

    assert response.status_code == 200
    body = response.get_json()
    assert len(body["slices"]) == 2
    assert body["selectedSlice"] is None


def test_delete_slice_out_of_range_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.delete(f"/api/v1/projects/{project_id}/slices/99")

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"
    assert AppState.from_cache(project_id).image_slices.__len__() == 3


# --- add-mask / remove-mask ------------------------------------------------------


def test_add_mask_requires_the_index_to_match_the_selection(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 1)
    _click(client, project_id, 80, 96)

    response = client.post(f"/api/v1/projects/{project_id}/slices/0/add-mask")

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_add_mask_then_remove_mask_round_trip(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 1)
    _click(client, project_id, 80, 96)

    add_response = client.post(f"/api/v1/projects/{project_id}/slices/1/add-mask")
    assert add_response.status_code == 200
    state = AppState.from_cache(project_id)
    assert state.image_slices[1].filename.endswith("_v2.png")
    # The mask used to add is cleared by the post-mutation refresh (matches
    # Dash's update_slices side effect; see slice_editing_services.py).
    assert state.slice_mask is None

    # Regenerate an equivalent mask before removing (see PARITY.md note on
    # this exact sequencing).
    _click(client, project_id, 80, 96)
    remove_response = client.post(
        f"/api/v1/projects/{project_id}/slices/1/remove-mask"
    )
    assert remove_response.status_code == 200
    assert AppState.from_cache(project_id).image_slices[1].filename.endswith("_v3.png")


def test_add_mask_without_a_mask_is_not_ready(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 0)

    response = client.post(f"/api/v1/projects/{project_id}/slices/0/add-mask")

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


# --- clipboard -----------------------------------------------------------------


def test_copy_requires_a_mask(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/clipboard/copy")

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


def test_copy_and_paste_round_trip(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 1)
    _click(client, project_id, 80, 96)

    copy_response = client.post(f"/api/v1/projects/{project_id}/clipboard/copy")
    assert copy_response.status_code == 200
    assert copy_response.get_json()["clipboard"] is True

    paste_response = client.post(f"/api/v1/projects/{project_id}/clipboard/paste")
    assert paste_response.status_code == 200
    assert AppState.from_cache(project_id).image_slices[1].filename.endswith("_v2.png")
    # The clipboard survives paste; it is not cleared afterward.
    assert paste_response.get_json()["clipboard"] is True


def test_paste_without_a_selected_slice_is_not_ready(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _click(client, project_id, 80, 96)  # no selection required for a depth click
    client.post(f"/api/v1/projects/{project_id}/clipboard/copy")

    response = client.post(f"/api/v1/projects/{project_id}/clipboard/paste")

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


# --- balance ---------------------------------------------------------------------


def test_balance_evenly_redistributes_depths(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    assert [s["depth"] for s in view["slices"]] == [85, 170, 255]

    response = client.post(f"/api/v1/projects/{project_id}/slices/balance", json={})

    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    assert [s["depth"] for s in body["slices"]] == [0, 127, 255]


def test_balance_with_no_slices_is_unchanged(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/slices/balance", json={})

    assert response.status_code == 200
    assert response.get_json()["changed"] is False


# --- set slice depth ---------------------------------------------------------------


def test_set_slice_depth_reorders_and_clears_selection(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 1)  # depth 170, untouched by this edit

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/0/depth", json={"depth": 200}
    )

    assert response.status_code == 200
    body = response.get_json()
    assert [s["depth"] for s in body["slices"]] == [170, 200, 255]
    assert body["selectedSlice"] is None


def test_set_slice_depth_out_of_range_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/9/depth", json={"depth": 1}
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- replace slice image (upload) ---------------------------------------------------


def test_replace_slice_image_matching_aspect_bumps_version(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    replacement = Image.new("RGB", (320, 240), (1, 2, 3))
    buffer = io.BytesIO()
    replacement.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/1/image",
        data={"image": (buffer, "replacement.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert AppState.from_cache(project_id).image_slices[1].filename.endswith("_v2.png")
    body = response.get_json()
    # The recomposed input image is persisted and its URL changes.
    assert body["assets"]["input"]["url"] != view["assets"]["input"]["url"]
    served = Image.open(io.BytesIO(client.get(body["assets"]["input"]["url"]).data))
    expected = AppState.from_cache(project_id).imgData
    assert np.array_equal(np.asarray(served.convert("RGB")), np.asarray(expected.convert("RGB")))
    on_disk = Image.open(Path(project_id) / AppState.IMAGE_FILE)
    assert np.array_equal(np.asarray(on_disk.convert("RGB")), np.asarray(expected.convert("RGB")))


def test_replace_slice_image_requires_the_multipart_field(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/1/image",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- invert / feather mask -----------------------------------------------------------


def test_invert_mask_creates_an_all_zero_mask_first(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/mask/invert")

    assert response.status_code == 200
    state = AppState.from_cache(project_id)
    assert (state.slice_mask == 255).all()


def test_feather_mask_requires_an_existing_mask(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/mask/feather")

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


def test_feather_mask_blurs_an_existing_mask(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _click(client, project_id, 80, 96)
    state = AppState.from_cache(project_id)
    before_nonzero = int((state.slice_mask > 0).sum())

    response = client.post(f"/api/v1/projects/{project_id}/mask/feather")

    assert response.status_code == 200
    after = AppState.from_cache(project_id).slice_mask
    assert int((after > 0).sum()) > before_nonzero
    assert int(after.max()) == 255


# --- checkerboard --------------------------------------------------------------------


def test_set_checkerboard_toggles_and_recomposes_when_selected(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 1)

    response = client.put(
        f"/api/v1/projects/{project_id}/display", json={"useCheckerboard": True}
    )

    assert response.status_code == 200
    assert response.get_json()["useCheckerboard"] is True
    assert AppState.from_cache(project_id).use_checkerboard is True


# --- undo / redo -----------------------------------------------------------------------


def test_undo_and_redo_round_trip(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select(client, project_id, 1)
    _click(client, project_id, 80, 96)
    client.post(f"/api/v1/projects/{project_id}/slices/1/add-mask")
    assert AppState.from_cache(project_id).image_slices[1].filename.endswith("_v2.png")

    undo_response = client.post(f"/api/v1/projects/{project_id}/slices/1/undo")
    assert undo_response.status_code == 200
    assert AppState.from_cache(project_id).image_slices[1].filename.endswith(
        "image_slice_1.png"
    )

    redo_response = client.post(f"/api/v1/projects/{project_id}/slices/1/redo")
    assert redo_response.status_code == 200
    assert AppState.from_cache(project_id).image_slices[1].filename.endswith("_v2.png")


def test_undo_with_no_earlier_version_is_not_ready(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/slices/1/undo")

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


def test_undo_out_of_range_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/slices/99/undo")

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"
