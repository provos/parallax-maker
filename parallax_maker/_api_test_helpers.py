"""Small helpers shared by ``test_api_*.py``.

Named without a ``test_`` prefix so pytest does not try to collect it as a
test module.
"""

from __future__ import annotations

import io
import time

from .e2e_support import create_input_image


def upload_fixture_image(client, image=None) -> dict:
    """Upload ``image`` (or the standard e2e fixture image) and return the ProjectView."""

    image = image if image is not None else create_input_image()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    response = client.post(
        "/api/v1/projects",
        data={"image": (buffer, "input.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def poll_job(client, job_id: str, timeout: float = 60.0, interval: float = 0.02) -> dict:
    """Poll ``GET /jobs/{id}`` until it reaches a terminal status."""

    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200, response.get_json()
        last = response.get_json()
        if last["status"] in ("succeeded", "failed"):
            return last
        time.sleep(interval)
    raise AssertionError(f"job {job_id} did not reach a terminal status: {last}")
