"""HTTP tests for segmentation and slice-selection routes.

Reproduces e2e/parallax-maker.spec.ts scenarios 2-5 (point-mask modifiers,
multi-point queue/commit/toggle, default depth-map clicks, and selected-slice
segmentation source) over the ``/api/v1`` contract, using the same fixture
state and fake models as the browser tests
(``parallax_maker.e2e_support.create_fixture_state``/``create_fake_runtime``)
but exercised directly through the Flask test client instead of a browser.
"""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from flask import Flask

from . import controller
from ._api_test_helpers import poll_job, upload_fixture_image
from .api import create_api_blueprint
from .api.segmentation import modifiers_to_operation, modifiers_to_polarity
from .controller import AppState, CompositeMode
from .e2e_support import create_fixture_state
from .e2e_support.fakes import (
    FakeDepthEstimationModel,
    FakeInpaintingModel,
    FakeSegmentationModel,
    FakeUpscaler,
)
from .runtime import build_runtime
from .segmentation_services import MaskOperation, PointPolarity


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


def _click(client, project_id, x, y, *, mode="instance", shift=False, ctrl=False) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/segmentation/click",
        json={"x": x, "y": y, "mode": mode, "shiftKey": shift, "ctrlKey": ctrl},
    )
    assert response.status_code == 202, response.get_json()
    return poll_job(client, response.get_json()["job"]["id"])


def _main_bytes(client, project_id) -> bytes:
    response = client.get(f"/api/v1/projects/{project_id}/assets/main")
    assert response.status_code == 200, response.get_json()
    return response.data


def _logs_text(client, project_id) -> str:
    response = client.get(f"/api/v1/projects/{project_id}/logs")
    assert response.status_code == 200, response.get_json()
    return "\n".join(entry["message"] for entry in response.get_json()["entries"])


# --- Pure modifier-mapping unit tests (scenario 2/3's Shift/Ctrl routing) ----


@pytest.mark.parametrize(
    "shift_key,ctrl_key,expected",
    [
        (False, False, MaskOperation.REPLACE),
        (True, False, MaskOperation.ADD),
        (False, True, MaskOperation.SUBTRACT),
        # Shift wins when both modifiers are held, exactly like Dash's click_event.
        (True, True, MaskOperation.ADD),
    ],
)
def test_modifiers_to_operation(shift_key, ctrl_key, expected) -> None:
    assert modifiers_to_operation(shift_key, ctrl_key) is expected


@pytest.mark.parametrize(
    "ctrl_key,expected",
    [
        (False, PointPolarity.POSITIVE),
        (True, PointPolarity.NEGATIVE),
    ],
)
def test_modifiers_to_polarity(ctrl_key, expected) -> None:
    assert modifiers_to_polarity(ctrl_key) is expected


def test_instance_click_modifiers_replace_union_subtract_replace(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    job = _click(client, project_id, 80, 96)
    assert job["status"] == "succeeded"
    mask = AppState.from_cache(project_id).slice_mask
    assert mask[96, 80] == 255
    assert mask[8, 8] == 0
    first_nonzero = int(np.count_nonzero(mask))
    assert first_nonzero > 4_000
    first_bytes = _main_bytes(client, project_id)

    job = _click(client, project_id, 200, 96, shift=True)
    assert job["status"] == "succeeded"
    mask = AppState.from_cache(project_id).slice_mask
    assert mask[96, 80] == 255
    assert mask[96, 200] == 255
    assert mask[8, 8] == 0
    assert int(np.count_nonzero(mask)) == first_nonzero * 2
    union_bytes = _main_bytes(client, project_id)
    assert union_bytes != first_bytes

    job = _click(client, project_id, 80, 96, ctrl=True)
    assert job["status"] == "succeeded"
    mask = AppState.from_cache(project_id).slice_mask
    assert mask[96, 80] == 0
    assert mask[96, 200] == 255
    assert int(np.count_nonzero(mask)) == first_nonzero
    subtract_bytes = _main_bytes(client, project_id)
    assert subtract_bytes != union_bytes

    job = _click(client, project_id, 80, 96)
    assert job["status"] == "succeeded"
    mask = AppState.from_cache(project_id).slice_mask
    assert mask[96, 80] == 255
    assert mask[96, 200] == 0
    assert int(np.count_nonzero(mask)) == first_nonzero
    assert _main_bytes(client, project_id) == first_bytes


def test_instance_click_has_mask_view_tracks_replace_state(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    assert view["segmentation"]["hasMask"] is False

    job = _click(client, project_id, 80, 96)
    assert job["project"]["segmentation"]["hasMask"] is True


# --- Scenario 3: multi-point queue/commit and toggle behavior --------------


def test_multi_point_queue_toggle_and_commit(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    initial_main_bytes = _main_bytes(client, project_id)
    assert AppState.from_cache(project_id).slice_mask is None

    enable = client.put(
        f"/api/v1/projects/{project_id}/segmentation/multi-point", json={"enabled": True}
    )
    assert enable.status_code == 200
    assert enable.get_json()["segmentation"] == {
        "multiPointMode": True,
        "queuedPoints": [],
        "slicePixel": None,
        "slicePixelDepth": None,
        "hasMask": False,
    }
    assert AppState.from_cache(project_id).multi_point_mode is True

    positive_job = _click(client, project_id, 89, 95)
    assert positive_job["status"] == "succeeded"
    assert positive_job["project"]["segmentation"]["queuedPoints"] == [
        {"x": 89, "y": 95, "negative": False}
    ]
    assert AppState.from_cache(project_id).slice_mask is None
    assert _main_bytes(client, project_id) == initial_main_bytes

    negative_job = _click(client, project_id, 127, 95, ctrl=True)
    assert negative_job["status"] == "succeeded"
    assert negative_job["project"]["segmentation"]["queuedPoints"] == [
        {"x": 89, "y": 95, "negative": False},
        {"x": 127, "y": 95, "negative": True},
    ]
    assert AppState.from_cache(project_id).slice_mask is None
    assert _main_bytes(client, project_id) == initial_main_bytes

    toggled_off = client.put(
        f"/api/v1/projects/{project_id}/segmentation/multi-point", json={"enabled": False}
    )
    assert toggled_off.status_code == 200
    assert toggled_off.get_json()["segmentation"]["multiPointMode"] is False
    assert toggled_off.get_json()["segmentation"]["queuedPoints"] == []
    assert AppState.from_cache(project_id).points_selected == []
    assert AppState.from_cache(project_id).slice_mask is None
    assert _main_bytes(client, project_id) == initial_main_bytes

    # Re-enable, queue the same points again, and commit.
    client.put(
        f"/api/v1/projects/{project_id}/segmentation/multi-point", json={"enabled": True}
    )
    _click(client, project_id, 89, 95)
    _click(client, project_id, 127, 95, ctrl=True)

    commit_response = client.post(
        f"/api/v1/projects/{project_id}/segmentation/commit", json={}
    )
    assert commit_response.status_code == 202
    commit_job = poll_job(client, commit_response.get_json()["job"]["id"])
    assert commit_job["status"] == "succeeded"

    mask = AppState.from_cache(project_id).slice_mask
    assert mask[95, 89] == 255
    assert mask[95, 127] == 0
    assert mask[8, 8] == 0
    assert int(np.count_nonzero(mask)) > 0
    assert _main_bytes(client, project_id) != initial_main_bytes
    assert "for Segment Anything" in _logs_text(client, project_id)
    # A successful commit retains the queue and multi-point mode (webui.py:251).
    state = AppState.from_cache(project_id)
    assert state.multi_point_mode is True
    assert state.points_selected == [((89, 95), False), ((127, 95), True)]


# --- Scenario 4: default depth-map click ------------------------------------


def test_depth_click_records_pixel_depth_log_and_mask(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    job = _click(client, project_id, 15, 15, mode="depth")
    assert job["status"] == "succeeded"

    state = AppState.from_cache(project_id)
    assert state.slice_pixel == (15, 15)
    assert state.slice_pixel_depth == 1
    assert state.slice_mask[16, 16] == 255
    assert state.slice_mask[96, 160] == 0

    assert (
        "Click event at pixel coordinates (15, 15) at depth 1" in _logs_text(client, project_id)
    )

    segmentation_view = job["project"]["segmentation"]
    assert segmentation_view["slicePixel"] == [15, 15]
    assert segmentation_view["slicePixelDepth"] == 1


def test_depth_click_does_not_append_the_instance_commit_log_line(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    _click(client, project_id, 15, 15, mode="depth")

    assert "Segment Anything" not in _logs_text(client, project_id)


# --- Scenario 5: selected-slice segmentation input --------------------------


def test_selected_slice_segmentation_uses_composed_slice_source(client, monkeypatch) -> None:
    # install_fakes() (browser tests only) tags the composed-slice image with
    # its provenance so FakeSegmentationModel can report what it received;
    # unit tests don't run install_fakes, so patch AppState.slice_image_composed
    # here the same way, per this module's docstring/task instructions.
    original_slice_image_composed = controller.AppState.slice_image_composed

    def slice_image_composed_with_source(state, slice_index, mode=CompositeMode.NONE):
        image = original_slice_image_composed(state, slice_index, mode=mode)
        image.info["e2e-segmentation-source"] = f"slice:{slice_index}:{mode.name}"
        return image

    monkeypatch.setattr(
        controller.AppState, "slice_image_composed", slice_image_composed_with_source
    )

    view = _restore_fixture(client)
    project_id = view["id"]

    selection = client.put(f"/api/v1/projects/{project_id}/selection", json={"slice": 1})
    assert selection.status_code == 200
    assert selection.get_json()["selectedSlice"] == 1

    job = _click(client, project_id, 128, 96)
    assert job["status"] == "succeeded"

    state = AppState.from_cache(project_id)
    assert state.selected_slice == 1
    assert state.slice_mask[96, 128] == 255

    model = state.segmentation_model
    assert model is not None
    assert model.segment_image_calls == 1
    assert model.segment_input_source == "slice:1:NONE"


# --- Selection endpoint ------------------------------------------------------


def test_selection_composes_slice_preview_and_clears_on_deselect(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    input_bytes = client.get(f"/api/v1/projects/{project_id}/assets/input").data
    assert _main_bytes(client, project_id) == input_bytes

    select = client.put(f"/api/v1/projects/{project_id}/selection", json={"slice": 0})
    assert select.status_code == 200
    body = select.get_json()
    assert body["changed"] is True
    assert body["selectedSlice"] == 0
    selected_bytes = _main_bytes(client, project_id)
    assert selected_bytes != input_bytes

    # Re-selecting the same slice is a no-op (changed: false); the frontend
    # sends `slice: null` to deselect instead of toggling on re-click.
    reselect = client.put(f"/api/v1/projects/{project_id}/selection", json={"slice": 0})
    assert reselect.status_code == 200
    assert reselect.get_json()["changed"] is False
    assert reselect.get_json()["selectedSlice"] == 0
    assert _main_bytes(client, project_id) == selected_bytes

    deselect = client.put(f"/api/v1/projects/{project_id}/selection", json={"slice": None})
    assert deselect.status_code == 200
    assert deselect.get_json()["changed"] is True
    assert deselect.get_json()["selectedSlice"] is None
    assert _main_bytes(client, project_id) == input_bytes


def test_selection_out_of_range_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.put(f"/api/v1/projects/{project_id}/selection", json={"slice": 99})

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"
    assert AppState.from_cache(project_id).selected_slice is None


def test_selection_requires_a_json_body(client) -> None:
    view = _restore_fixture(client)

    response = client.put(f"/api/v1/projects/{view['id']}/selection", data="not json")

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- Click bounds precheck --------------------------------------------------


def test_click_out_of_bounds_is_400_before_any_job_runs(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/segmentation/click",
        json={"x": 9999, "y": 5, "mode": "instance", "shiftKey": False, "ctrlKey": False},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"
    assert AppState.from_cache(project_id).slice_mask is None
    project_view = client.get(f"/api/v1/projects/{project_id}").get_json()
    assert project_view["busy"] is None


def test_click_with_negative_coordinate_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/segmentation/click",
        json={"x": -1, "y": 5, "mode": "instance", "shiftKey": False, "ctrlKey": False},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- Errors leave state unchanged (design item 7) ---------------------------


def test_commit_without_queued_points_fails_and_leaves_state_unchanged(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    client.put(
        f"/api/v1/projects/{project_id}/segmentation/multi-point", json={"enabled": True}
    )

    response = client.post(f"/api/v1/projects/{project_id}/segmentation/commit", json={})
    assert response.status_code == 202
    job = poll_job(client, response.get_json()["job"]["id"])

    assert job["status"] == "failed"
    assert "NoPointsQueued" in job["error"]

    state = AppState.from_cache(project_id)
    assert state.slice_mask is None
    assert state.multi_point_mode is True
    assert state.points_selected == []
    project_view = client.get(f"/api/v1/projects/{project_id}").get_json()
    assert project_view["busy"] is None


def test_commit_outside_multi_point_mode_fails_and_leaves_state_unchanged(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(f"/api/v1/projects/{project_id}/segmentation/commit", json={})
    job = poll_job(client, response.get_json()["job"]["id"])

    assert job["status"] == "failed"
    assert "MultiPointModeRequired" in job["error"]
    assert AppState.from_cache(project_id).slice_mask is None


def test_depth_click_without_a_depth_map_fails_and_leaves_state_unchanged(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/segmentation/click",
        json={"x": 5, "y": 5, "mode": "depth", "shiftKey": False, "ctrlKey": False},
    )
    assert response.status_code == 202
    job = poll_job(client, response.get_json()["job"]["id"])

    assert job["status"] == "failed"
    assert "SegmentationNotReady" in job["error"]

    state = AppState.from_cache(project_id)
    assert state.slice_mask is None
    assert state.slice_pixel is None


# --- Busy while a click/commit job holds the project ------------------------


def _blocking_segmentation_runtime(event: threading.Event):
    def blocking_segmentation_factory(model="sam"):
        fake = FakeSegmentationModel(model)
        original_mask_at_point_blended = fake.mask_at_point_blended

        def blocked(point_input):
            event.wait(timeout=5)
            return original_mask_at_point_blended(point_input)

        fake.mask_at_point_blended = blocked
        return fake

    return build_runtime(
        depth_model_factory=FakeDepthEstimationModel,
        segmentation_model_factory=blocking_segmentation_factory,
        inpainting_model_factory=FakeInpaintingModel,
        upscaler_factory=FakeUpscaler,
    )


def test_mutation_is_409_while_a_click_job_is_running(isolated_cwd) -> None:
    del isolated_cwd
    event = threading.Event()
    runtime = _blocking_segmentation_runtime(event)
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(runtime), url_prefix="/api/v1")
    client = app.test_client()

    try:
        view = _restore_fixture(client)
        project_id = view["id"]

        click_response = client.post(
            f"/api/v1/projects/{project_id}/segmentation/click",
            json={"x": 80, "y": 96, "mode": "instance", "shiftKey": False, "ctrlKey": False},
        )
        assert click_response.status_code == 202
        job_id = click_response.get_json()["job"]["id"]

        # The busy slot is reserved synchronously before the 202 response is
        # returned, so a concurrent mutation must be rejected immediately -
        # the segmentation model is still blocked on `event` at this point.
        busy_multi_point = client.put(
            f"/api/v1/projects/{project_id}/segmentation/multi-point", json={"enabled": True}
        )
        assert busy_multi_point.status_code == 409
        assert busy_multi_point.get_json()["error"]["code"] == "busy"

        busy_selection = client.put(
            f"/api/v1/projects/{project_id}/selection", json={"slice": 0}
        )
        assert busy_selection.status_code == 409
        assert busy_selection.get_json()["error"]["code"] == "busy"

        busy_click = client.post(
            f"/api/v1/projects/{project_id}/segmentation/click",
            json={"x": 80, "y": 96, "mode": "instance", "shiftKey": False, "ctrlKey": False},
        )
        assert busy_click.status_code == 409
        assert busy_click.get_json()["error"]["code"] == "busy"
    finally:
        event.set()

    job = poll_job(client, job_id)
    assert job["status"] == "succeeded"

    # The project is usable again once the job completes.
    ok_response = client.put(
        f"/api/v1/projects/{project_id}/segmentation/multi-point", json={"enabled": True}
    )
    assert ok_response.status_code == 200


def test_asset_urls_change_only_with_their_content(client) -> None:
    """Queueing a point bumps the revision but must not reload any image."""

    view = _restore_fixture(client)
    project_id = view["id"]

    def urls(v: dict) -> dict:
        return {
            "main": v["mainImage"]["url"],
            "input": v["assets"]["input"]["url"],
            "depth": v["assets"]["depth"]["url"],
            "thumbs": [s["thumbnail"]["url"] for s in v["slices"]],
        }

    before = urls(view)
    client.put(
        f"/api/v1/projects/{project_id}/segmentation/multi-point", json={"enabled": True}
    )
    queued = _click(client, project_id, 90, 96)["project"]
    assert queued["revision"] > view["revision"]
    assert queued["segmentation"]["queuedPoints"] == [
        {"x": 90, "y": 96, "negative": False}
    ]
    assert urls(queued) == before

    committed = poll_job(
        client,
        client.post(f"/api/v1/projects/{project_id}/segmentation/commit").get_json()[
            "job"
        ]["id"],
    )["project"]
    after = urls(committed)
    assert after["main"] != before["main"]
    assert {k: v for k, v in after.items() if k != "main"} == {
        k: v for k, v in before.items() if k != "main"
    }


def test_resetting_an_input_display_keeps_the_main_url(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    depth = client.post(f"/api/v1/projects/{project_id}/depth", json={"model": "midas"})
    finished = poll_job(client, depth.get_json()["job"]["id"])["project"]
    assert finished["mainImage"]["url"] == view["mainImage"]["url"]
