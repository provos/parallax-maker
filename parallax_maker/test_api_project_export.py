"""HTTP tests for project-lifecycle, export/render, and configuration routes.

Mirrors ``e2e/project-export.spec.ts``'s scenarios over the ``/api/v1``
contract, using the same fixture state and fake models as the browser tests
(``parallax_maker.e2e_support.create_fixture_state``/``create_fake_runtime``)
but exercised directly through the Flask test client.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from ._api_test_helpers import poll_job
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


def _export_gltf(client, project_id, *, dof=False) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/export/gltf", json={"dof": dof}
    )
    assert response.status_code == 202, response.get_json()
    job = poll_job(client, response.get_json()["job"]["id"])
    assert job["status"] == "succeeded", job
    return job


# --- Project lifecycle: save / state-file / restore round trip -----------------


def test_save_then_download_state_file_round_trips_through_restore(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    save_response = client.post(f"/api/v1/projects/{project_id}/save")
    assert save_response.status_code == 200, save_response.get_json()

    file_response = client.get(f"/api/v1/projects/{project_id}/state-file")
    assert file_response.status_code == 200
    assert 'filename="appstate.json"' in file_response.headers["Content-Disposition"]
    raw = file_response.data
    payload = json.loads(raw)
    assert payload["filename"] == project_id

    restore_response = client.post(
        "/api/v1/projects/restore",
        data={"state": (io.BytesIO(raw), "appstate.json")},
        content_type="multipart/form-data",
    )
    assert restore_response.status_code == 200, restore_response.get_json()
    restored = restore_response.get_json()

    assert restored["id"] == project_id
    assert restored["image"] == view["image"]
    assert restored["numSlices"] == view["numSlices"]
    assert restored["thresholds"] == view["thresholds"]
    assert len(restored["slices"]) == len(view["slices"])
    for before, after in zip(view["slices"], restored["slices"]):
        assert before["depth"] == after["depth"]
        assert before["positivePrompt"] == after["positivePrompt"]
        assert before["negativePrompt"] == after["negativePrompt"]
    assert restored["settings"] == view["settings"]

    # Pixel-exact round trip: the raw slice bytes on disk are unchanged.
    for index in range(len(view["slices"])):
        original = client.get(view["slices"][index]["image"]["url"]).data
        after = client.get(restored["slices"][index]["image"]["url"]).data
        assert hashlib.sha256(original).digest() == hashlib.sha256(after).digest()


def test_state_file_reflects_in_memory_state_without_requiring_save(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    settings_response = client.put(
        f"/api/v1/projects/{project_id}/settings", json={"darkMode": False}
    )
    assert settings_response.status_code == 200
    assert settings_response.get_json()["changed"] is True

    file_response = client.get(f"/api/v1/projects/{project_id}/state-file")
    payload = json.loads(file_response.data)
    assert payload["dark_mode"] is False


# --- Configuration: settings persistence ----------------------------------------


def test_update_settings_persists_depth_model_camera_and_dark_mode(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    assert view["settings"]["depthModel"] == "dinov2"

    response = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={
            "depthModel": "midas",
            "camera": {"distance": 200.0, "focalLength": 300.0, "maxDistance": 400.0},
            "meshDisplacement": 33.0,
            "darkMode": False,
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    camera_view = body["settings"]["camera"]
    assert camera_view.pop("groundNear") == 0.0
    assert camera_view.pop("horizonRow") == pytest.approx(body["image"]["height"] / 2)
    assert body["settings"] == {
        "depthModel": "midas",
        "camera": {
            "distance": 200.0,
            "focalLength": 300.0,
            "maxDistance": 400.0,
            "pitch": 0.0,
        },
        "meshDisplacement": 33.0,
        "darkMode": False,
    }

    # A second identical request is a no-op (matches every underlying Dash
    # persist-callback's own PreventUpdate-on-unchanged guard).
    noop = client.put(
        f"/api/v1/projects/{project_id}/settings", json={"depthModel": "midas"}
    )
    assert noop.status_code == 200
    assert noop.get_json()["changed"] is False

    # Settings survive a save/restore round trip.
    client.post(f"/api/v1/projects/{project_id}/save")
    raw = client.get(f"/api/v1/projects/{project_id}/state-file").data
    restored = client.post(
        "/api/v1/projects/restore",
        data={"state": (io.BytesIO(raw), "appstate.json")},
        content_type="multipart/form-data",
    ).get_json()
    restored["settings"]["camera"].pop("groundNear")
    restored["settings"]["camera"].pop("horizonRow")
    assert restored["settings"]["camera"] == {
        "distance": 200.0,
        "focalLength": 300.0,
        "maxDistance": 400.0,
        "pitch": 0.0,
    }
    assert restored["settings"]["meshDisplacement"] == 33.0
    assert restored["settings"]["darkMode"] is False


def test_update_settings_rejects_invalid_camera_values(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {"distance": 10.0, "focalLength": 0.0, "maxDistance": 10.0}},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_camera_pitch_is_optional_persisted_and_validated(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    camera = {"distance": 100.0, "focalLength": 50.0, "maxDistance": 500.0}

    response = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "pitch": 9.5}},
    )
    assert response.status_code == 200
    assert response.get_json()["settings"]["camera"]["pitch"] == 9.5

    # Omitting pitch (older clients) leaves it unchanged.
    response = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "focalLength": 60.0}},
    )
    assert response.get_json()["settings"]["camera"]["pitch"] == 9.5

    client.post(f"/api/v1/projects/{project_id}/save")
    raw = client.get(f"/api/v1/projects/{project_id}/state-file").data
    restored = client.post(
        "/api/v1/projects/restore",
        data={"state": (io.BytesIO(raw), "appstate.json")},
        content_type="multipart/form-data",
    ).get_json()
    assert restored["settings"]["camera"]["pitch"] == 9.5

    too_steep = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "pitch": 61.0}},
    )
    assert too_steep.status_code == 400
    assert too_steep.get_json()["error"]["code"] == "invalid_request"

    # Within +-60 but too steep for a very wide lens: a client error, and
    # nothing is changed.
    wide_and_steep = client.put(
        f"/api/v1/projects/{project_id}/settings",
        json={"camera": {**camera, "focalLength": 5.0, "pitch": 60.0}},
    )
    assert wide_and_steep.status_code == 400
    assert wide_and_steep.get_json()["error"]["code"] == "invalid_request"
    current = client.get(f"/api/v1/projects/{project_id}").get_json()
    assert current["settings"]["camera"]["pitch"] == 9.5
    assert current["settings"]["camera"]["focalLength"] == 60.0


# --- Export: glTF (displacement / DOF / upscaled) -------------------------------


def test_gltf_export_without_displacement_is_a_flat_quad(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    # The fixture project starts with mesh_displacement=15 (create_fixture_state);
    # zero it first so this export exercises the flat-quad (no depth-map) path.
    settings = client.put(
        f"/api/v1/projects/{project_id}/settings", json={"meshDisplacement": 0}
    )
    assert settings.status_code == 200

    job = _export_gltf(client, project_id)
    assert job["project"]["exports"]["gltf"] is not None

    download = client.get(f"/api/v1/projects/{project_id}/export/gltf")
    assert download.status_code == 200
    assert "filename=scene.gltf" in download.headers["Content-Disposition"]
    scene = json.loads(download.data)
    assert scene["asset"]["version"] == "2.0"
    assert len(scene["meshes"]) == 3
    assert scene["accessors"][1]["count"] == 4
    assert scene["materials"][0]["alphaMode"] == "BLEND"


def test_gltf_export_with_displacement_produces_a_subdivided_mesh(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    settings = client.put(
        f"/api/v1/projects/{project_id}/settings", json={"meshDisplacement": 10.0}
    )
    assert settings.status_code == 200

    _export_gltf(client, project_id)
    scene = json.loads(client.get(f"/api/v1/projects/{project_id}/export/gltf").data)

    # displacement_scale > 0 with a generated per-slice depth map subdivides
    # each card into a (subdivisions + 1)^2 grid instead of a flat 4-vertex
    # quad - see test_export_services.py for the underlying service-level
    # proof; this asserts the same contract end to end over HTTP.
    assert scene["accessors"][1]["count"] > 4


def test_gltf_export_with_dof_uses_mask_alpha_mode(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    _export_gltf(client, project_id, dof=True)
    scene = json.loads(client.get(f"/api/v1/projects/{project_id}/export/gltf").data)

    assert scene["materials"][0]["alphaMode"] == "MASK"
    assert scene["materials"][0]["alphaCutoff"] == 0.5


class _FakeUpscaler:
    """Deterministic 2x nearest-neighbour upscaler (mirrors
    ``e2e_support.fakes.FakeUpscaler``, reimplemented here since
    ``create_fake_runtime()`` does not patch ``controller.Upscaler`` - only
    ``e2e_support.fakes.install_fakes()`` does, which this lightweight API
    test doesn't want to invoke just for this one class)."""

    def __init__(self, model_name="swin2sr", external_model=None) -> None:
        del model_name, external_model

    def upscale_image_tiled(self, image, overlap=64, prompt="", negative_prompt=""):
        del overlap, prompt, negative_prompt
        source = image if isinstance(image, Image.Image) else Image.fromarray(image)
        return source.resize((source.width * 2, source.height * 2))


class _FakeInpaintingPipeline:
    """Stand-in for ``inpainting.InpaintingModel``: just enough surface for
    ``create_inpainting_pipeline`` (``inpainting.py``)."""

    def __init__(self, model, server_address=None, workflow_path=None, api_key=None):
        self.model = model

    def __eq__(self, other):
        return isinstance(other, _FakeInpaintingPipeline) and self.model == other.model

    def load_model(self):
        return None


def test_upscale_then_gltf_export_uses_upscaled_textures(client, monkeypatch) -> None:
    import parallax_maker.controller as controller
    import parallax_maker.inpainting as inpainting

    monkeypatch.setattr(controller, "Upscaler", _FakeUpscaler)
    monkeypatch.setattr(inpainting, "InpaintingModel", _FakeInpaintingPipeline)

    view = _restore_fixture(client)
    project_id = view["id"]
    original_sizes = []
    for slice_view in view["slices"]:
        data = client.get(slice_view["image"]["url"]).data
        original_sizes.append(Image.open(io.BytesIO(data)).size)

    upscale_response = client.post(f"/api/v1/projects/{project_id}/export/upscale")
    assert upscale_response.status_code == 202
    upscale_job = poll_job(client, upscale_response.get_json()["job"]["id"])
    assert upscale_job["status"] == "succeeded", upscale_job
    assert upscale_job["project"]["exports"]["upscaled"] is True

    _export_gltf(client, project_id)
    scene = json.loads(client.get(f"/api/v1/projects/{project_id}/export/gltf").data)

    assert len(scene["images"]) == 3
    for index, image in enumerate(scene["images"]):
        assert image["uri"].startswith("data:image/png;base64,")
        decoded = base64.b64decode(image["uri"].split(",", 1)[1])
        width, height = Image.open(io.BytesIO(decoded)).size
        # FakeUpscaler (e2e_support.fakes) doubles both dimensions.
        assert (width, height) == (
            original_sizes[index][0] * 2,
            original_sizes[index][1] * 2,
        )


def test_gltf_export_requires_at_least_one_slice(client) -> None:
    from ._api_test_helpers import upload_fixture_image

    view = upload_fixture_image(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/export/gltf", json={"dof": False}
    )
    assert response.status_code == 202
    job = poll_job(client, response.get_json()["job"]["id"])
    assert job["status"] == "failed"
    assert "at least one slice" in job["error"]


def test_gltf_download_before_any_export_is_404(client) -> None:
    from ._api_test_helpers import upload_fixture_image

    view = upload_fixture_image(client)
    project_id = view["id"]

    response = client.get(f"/api/v1/projects/{project_id}/export/gltf")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


# --- Export: animation frames -----------------------------------------------------


def test_animation_export_writes_distinct_frames_with_no_download_field(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/export/animation", json={"frames": 4}
    )
    assert response.status_code == 202
    job = poll_job(client, response.get_json()["job"]["id"])
    assert job["status"] == "succeeded", job

    frame_paths = sorted(Path(project_id).glob("rendered_image_*.png"))
    assert [path.name for path in frame_paths] == [
        "rendered_image_000.png",
        "rendered_image_001.png",
        "rendered_image_002.png",
        "rendered_image_003.png",
    ]

    hashes = set()
    for path in frame_paths:
        with Image.open(path) as frame:
            assert frame.size == (view["image"]["width"], view["image"]["height"])
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(hashes) == 4  # every frame is visually distinct


def test_animation_export_rejects_non_positive_frame_counts(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/export/animation", json={"frames": 0}
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- Export: raw slice download ----------------------------------------------------


def test_slice_download_returns_the_exact_slice_file_bytes(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.get(f"/api/v1/projects/{project_id}/slices/1/download")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
    assert "attachment" in response.headers["Content-Disposition"]

    on_disk = (Path.cwd() / project_id / "image_slice_1.png").read_bytes()
    assert response.data == on_disk


def test_slice_download_rejects_an_out_of_range_index(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.get(f"/api/v1/projects/{project_id}/slices/99/download")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- Configuration probes -----------------------------------------------------------
#
# api/configuration.py's routes call configuration_services.probe_server/
# validate_api_key with no injected probes, so they resolve the real
# automatic1111/comfyui/stabilityai/falai module attributes at call time
# (see that module's docstring). The `client`/`runtime` fixtures build a fake
# Runtime (create_fake_runtime()) but deliberately do not call
# e2e_support.fakes.install_fakes() (these lightweight API tests don't need
# every provider patched, just the one each test exercises), so each probe
# test patches only the one provider entry point it exercises - offline
# determinism without a real network dependency, mirroring what
# install_fakes() does for the browser suite.


def test_probe_server_automatic1111_success(client, monkeypatch) -> None:
    import parallax_maker.automatic1111 as automatic1111

    monkeypatch.setattr(
        automatic1111, "make_models_request", lambda server_address: ["e2e-model"]
    )

    response = client.post(
        "/api/v1/config/probe-server",
        json={"model": "automatic1111", "serverAddress": "localhost:7860"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body == {
        "ok": True,
        "message": "Connection to automatic1111 successful: ['e2e-model']",
    }


def test_probe_server_comfyui_success(client, monkeypatch) -> None:
    import parallax_maker.comfyui as comfyui

    monkeypatch.setattr(
        comfyui, "get_history", lambda server_address, prompt_id: {"e2e": "ready"}
    )

    response = client.post(
        "/api/v1/config/probe-server",
        json={"model": "comfyui", "serverAddress": "localhost:8188"},
    )
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_probe_server_failure_is_ok_false_not_5xx(client, monkeypatch) -> None:
    import parallax_maker.automatic1111 as automatic1111

    def boom(server_address):
        raise ConnectionRefusedError("no server listening")

    monkeypatch.setattr(automatic1111, "make_models_request", boom)

    response = client.post(
        "/api/v1/config/probe-server",
        json={"model": "automatic1111", "serverAddress": "localhost:1"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is False
    assert "no server listening" in body["message"]


def test_probe_server_unknown_model_is_ok_false_not_5xx(client) -> None:
    response = client.post(
        "/api/v1/config/probe-server",
        json={"model": "not-a-real-model", "serverAddress": "localhost:1"},
    )
    assert response.status_code == 200
    assert response.get_json()["ok"] is False


def test_validate_key_stabilityai_success_never_echoes_key(client, monkeypatch) -> None:
    import parallax_maker.stabilityai as stabilityai

    class FakeStabilityAI:
        def __init__(self, api_key):
            self.api_key = api_key

        def validate_key(self):
            return True, 42.5

    monkeypatch.setattr(stabilityai, "StabilityAI", FakeStabilityAI)

    secret = "sk-1234567890abcdef"
    response = client.post(
        "/api/v1/config/validate-key",
        json={"model": "stabilityai", "apiKey": secret},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert secret not in json.dumps(body)


def test_validate_key_falai_success_never_echoes_key(client, monkeypatch) -> None:
    import parallax_maker.falai as falai

    class FakeFalAI:
        def __init__(self, api_key):
            self.api_key = api_key

        def validate_key(self):
            return True, None

    monkeypatch.setattr(falai, "FalAI", FakeFalAI)

    secret = "a-long-enough-fal-key"
    response = client.post(
        "/api/v1/config/validate-key",
        json={"model": "falai-sdxl", "apiKey": secret},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body == {"ok": True, "message": "Connection to falai-sdxl successful"}
    assert secret not in json.dumps(body)


def test_validate_key_falai_bad_format_is_ok_false(client) -> None:
    response = client.post(
        "/api/v1/config/validate-key",
        json={"model": "falai-sdxl", "apiKey": "short"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is False
    assert body["message"] == "Invalid fal.ai API key format"
