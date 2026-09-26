from __future__ import annotations

import io
import threading
import time

import cv2
import numpy as np
from flask import Flask
from PIL import Image

from ._api_test_helpers import poll_job, upload_fixture_image
from .api import create_api_blueprint
from .e2e_support import create_input_image
from .e2e_support.fakes import (
    FakeDepthEstimationModel,
    FakeInpaintingModel,
    FakeSegmentationModel,
    FakeUpscaler,
)
from .runtime import build_runtime

# Pixel signatures for the checkerboard-composited display thumbnails, taken
# from e2e/parallax-maker.spec.ts scenario 1 (upload -> deterministic depth ->
# three slices), which uses the same fixture input image and fake depth model.
EXPECTED_THUMBNAIL_PIXELS = {
    0: {
        (16, 16): (30, 21, 70, 255),
        (160, 120): (200, 200, 200, 255),
        (304, 223): (75, 75, 75, 255),
    },
    1: {
        (16, 16): (200, 200, 200, 255),
        (160, 120): (120, 95, 70, 255),
        (304, 223): (75, 75, 75, 255),
    },
    2: {
        (16, 16): (200, 200, 200, 255),
        (160, 120): (200, 200, 200, 255),
        (304, 223): (210, 168, 70, 255),
    },
}


def _run_depth_and_slices(client, project_id: str) -> dict:
    depth_response = client.post(
        f"/api/v1/projects/{project_id}/depth", json={"model": "dinov2"}
    )
    assert depth_response.status_code == 202
    depth_job = poll_job(client, depth_response.get_json()["job"]["id"])
    assert depth_job["status"] == "succeeded"

    slices_response = client.post(f"/api/v1/projects/{project_id}/slices", json={})
    assert slices_response.status_code == 202
    slices_job = poll_job(client, slices_response.get_json()["job"]["id"])
    assert slices_job["status"] == "succeeded"
    return slices_job


def test_depth_job_lifecycle_reaches_succeeded_with_progress(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/depth", json={"model": "dinov2"}
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["job"]["status"] == "queued"

    job = poll_job(client, body["job"]["id"])

    assert job["status"] == "succeeded"
    assert job["progress"] == 1.0
    assert job["error"] is None
    project = job["project"]
    assert project["depthModel"] == "dinov2"
    assert project["numSlices"] == 3
    assert len(project["thresholds"]) == project["numSlices"] + 1
    assert project["assets"]["depth"]["url"].startswith(
        f"/api/v1/projects/{project_id}/assets/depth"
    )
    assert project["revision"] > view["revision"]


def test_depth_asset_pixels_match_the_fake_depth_model(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    job = poll_job(
        client,
        client.post(
            f"/api/v1/projects/{project_id}/depth", json={"model": "dinov2"}
        ).get_json()["job"]["id"],
    )
    project = job["project"]

    response = client.get(project["assets"]["depth"]["url"])
    assert response.status_code == 200
    depth_image = Image.open(io.BytesIO(response.data))

    expected = FakeDepthEstimationModel("dinov2").depth_map(
        np.array(create_input_image().convert("RGB"))
    )
    expected = cv2.normalize(expected, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    assert np.array_equal(np.asarray(depth_image), expected)


def test_slices_job_produces_three_slices_with_e2e_scenario_1_thumbnails(
    client,
) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]

    job = _run_depth_and_slices(client, project_id)
    project = job["project"]

    assert project["numSlices"] == 3
    assert len(project["slices"]) == 3
    assert len(project["thresholds"]) == 4

    for slice_view in project["slices"]:
        index = slice_view["index"]

        raw_response = client.get(slice_view["image"]["url"])
        assert raw_response.status_code == 200
        raw_image = Image.open(io.BytesIO(raw_response.data))
        assert raw_image.mode == "RGBA"

        thumb_response = client.get(slice_view["thumbnail"]["url"])
        assert thumb_response.status_code == 200
        thumb_image = Image.open(io.BytesIO(thumb_response.data)).convert("RGBA")

        for (x, y), expected_pixel in EXPECTED_THUMBNAIL_PIXELS[index].items():
            assert thumb_image.getpixel((x, y)) == expected_pixel, (
                index,
                x,
                y,
                thumb_image.getpixel((x, y)),
            )


def test_slice_asset_conditional_get_returns_304(client) -> None:
    view = upload_fixture_image(client)
    job = _run_depth_and_slices(client, view["id"])
    thumb_url = job["project"]["slices"][0]["thumbnail"]["url"]

    first = client.get(thumb_url)
    second = client.get(thumb_url, headers={"If-None-Match": first.headers["ETag"]})

    assert second.status_code == 304


def test_unknown_job_is_404(client) -> None:
    response = client.get("/api/v1/jobs/does-not-exist")

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


def _blocking_runtime(event: threading.Event):
    def blocking_depth_factory(model="midas"):
        fake = FakeDepthEstimationModel(model)
        original_depth_map = fake.depth_map

        def blocked(image, progress_callback=None):
            event.wait(timeout=5)
            return original_depth_map(image, progress_callback=progress_callback)

        fake.depth_map = blocked
        return fake

    return build_runtime(
        depth_model_factory=blocking_depth_factory,
        segmentation_model_factory=FakeSegmentationModel,
        inpainting_model_factory=FakeInpaintingModel,
        upscaler_factory=FakeUpscaler,
    )


def test_mutating_request_is_409_while_a_job_is_running(isolated_cwd) -> None:
    del isolated_cwd
    event = threading.Event()
    runtime = _blocking_runtime(event)
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(runtime), url_prefix="/api/v1")
    client = app.test_client()

    try:
        view = upload_fixture_image(client)
        project_id = view["id"]

        depth_response = client.post(
            f"/api/v1/projects/{project_id}/depth", json={"model": "midas"}
        )
        assert depth_response.status_code == 202
        job_id = depth_response.get_json()["job"]["id"]

        # The busy slot is reserved synchronously before the 202 response is
        # returned, so a concurrent mutation must be rejected immediately -
        # the depth model is still blocked on `event` at this point.
        busy_response = client.put(
            f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 4}
        )
        assert busy_response.status_code == 409
        assert busy_response.get_json()["error"]["code"] == "busy"

        second_depth = client.post(
            f"/api/v1/projects/{project_id}/depth", json={"model": "midas"}
        )
        assert second_depth.status_code == 409
        assert second_depth.get_json()["error"]["code"] == "busy"

        project_view = client.get(f"/api/v1/projects/{project_id}").get_json()
        assert project_view["busy"] == {"jobId": job_id, "kind": "depth"}
    finally:
        event.set()

    job = poll_job(client, job_id)
    assert job["status"] == "succeeded"

    # The project is usable again once the job completes.
    ok_response = client.put(
        f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 4}
    )
    assert ok_response.status_code == 200


def _failing_runtime():
    def failing_depth_factory(model="midas"):
        fake = FakeDepthEstimationModel(model)

        def boom(image, progress_callback=None):
            raise RuntimeError("synthetic depth failure")

        fake.depth_map = boom
        return fake

    return build_runtime(
        depth_model_factory=failing_depth_factory,
        segmentation_model_factory=FakeSegmentationModel,
        inpainting_model_factory=FakeInpaintingModel,
        upscaler_factory=FakeUpscaler,
    )


def test_failed_job_reports_a_sanitized_error_and_project_stays_usable(
    isolated_cwd,
) -> None:
    del isolated_cwd
    runtime = _failing_runtime()
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(runtime), url_prefix="/api/v1")
    client = app.test_client()

    view = upload_fixture_image(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/depth", json={"model": "midas"}
    )
    job = poll_job(client, response.get_json()["job"]["id"])

    assert job["status"] == "failed"
    assert "RuntimeError" in job["error"]
    assert "synthetic depth failure" in job["error"]

    # The project remains usable: it's no longer busy and can be read/mutated.
    project_response = client.get(f"/api/v1/projects/{project_id}")
    assert project_response.status_code == 200
    assert project_response.get_json()["busy"] is None

    retry_response = client.put(
        f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 4}
    )
    assert retry_response.status_code == 200


def test_job_view_has_detail_and_cancellable_fields(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/depth", json={"model": "dinov2"}
    )
    queued_job = response.get_json()["job"]
    assert queued_job["detail"] is None
    # Any queued job can be cancelled, regardless of its kind.
    assert queued_job["cancellable"] is True

    job = poll_job(client, queued_job["id"])
    assert job["status"] == "succeeded"
    assert job["detail"] is None
    # A finished job can no longer be cancelled.
    assert job["cancellable"] is False


def test_delete_unknown_job_is_404(client) -> None:
    response = client.delete("/api/v1/jobs/does-not-exist")

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


def test_delete_running_non_cancellable_job_is_409(isolated_cwd) -> None:
    del isolated_cwd
    event = threading.Event()
    runtime = _blocking_runtime(event)
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(runtime), url_prefix="/api/v1")
    client = app.test_client()

    try:
        view = upload_fixture_image(client)
        project_id = view["id"]

        depth_response = client.post(
            f"/api/v1/projects/{project_id}/depth", json={"model": "midas"}
        )
        job_id = depth_response.get_json()["job"]["id"]

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.get(f"/api/v1/jobs/{job_id}").get_json()["status"]
            if status == "running":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("depth job never reached running")

        delete_response = client.delete(f"/api/v1/jobs/{job_id}")
        assert delete_response.status_code == 409
        assert delete_response.get_json()["error"]["code"] == "not_cancellable"
    finally:
        event.set()

    job = poll_job(client, job_id)
    assert job["status"] == "succeeded"


def test_delete_queued_job_cancels_it_and_releases_the_busy_slot(
    isolated_cwd,
) -> None:
    del isolated_cwd
    event = threading.Event()
    runtime = _blocking_runtime(event)
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(runtime), url_prefix="/api/v1")
    client = app.test_client()

    try:
        view_a = upload_fixture_image(client)
        project_a = view_a["id"]
        view_b = upload_fixture_image(client)
        project_b = view_b["id"]

        # Occupies the single worker thread; blocked on `event`.
        response_a = client.post(
            f"/api/v1/projects/{project_a}/depth", json={"model": "midas"}
        )
        assert response_a.status_code == 202
        job_a_id = response_a.get_json()["job"]["id"]

        # project_b's own busy slot is free, but the worker thread is still
        # occupied running job A, so this job is left queued behind it.
        response_b = client.post(
            f"/api/v1/projects/{project_b}/depth", json={"model": "midas"}
        )
        assert response_b.status_code == 202
        job_b = response_b.get_json()["job"]
        assert job_b["status"] == "queued"
        job_b_id = job_b["id"]

        # project_b's busy slot was reserved synchronously (before the
        # worker thread even sees the job), so a concurrent mutation is
        # already rejected.
        busy_response = client.put(
            f"/api/v1/projects/{project_b}/slice-count", json={"numSlices": 4}
        )
        assert busy_response.status_code == 409
        assert busy_response.get_json()["error"]["code"] == "busy"

        delete_response = client.delete(f"/api/v1/jobs/{job_b_id}")
        assert delete_response.status_code == 200
        assert delete_response.get_json()["id"] == job_b_id
    finally:
        event.set()

    job_a = poll_job(client, job_a_id)
    assert job_a["status"] == "succeeded"

    job_b_final = poll_job(client, job_b_id)
    assert job_b_final["status"] == "cancelled"
    assert job_b_final["project"] is not None  # like succeeded/failed jobs

    # project_b's busy slot was released once its cancelled job finished
    # unwinding through `_begin_job`'s wrapper.
    ok_response = client.put(
        f"/api/v1/projects/{project_b}/slice-count", json={"numSlices": 4}
    )
    assert ok_response.status_code == 200
