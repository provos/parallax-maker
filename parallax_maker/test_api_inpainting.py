"""HTTP tests for the canvas-mask and inpainting routes.

Reproduces e2e/parallax-maker.spec.ts scenarios 6-9 (painted-mask
candidates/apply/version-change, fill, enhance, erase) over the ``/api/v1``
contract, using the same fixture state and fake models as the browser tests
(``parallax_maker.e2e_support.create_fixture_state``/``create_fake_runtime``)
but exercised directly through the Flask test client instead of a browser.
"""

from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask
from PIL import Image, ImageDraw

from . import controller
from ._api_test_helpers import poll_job
from .api import create_api_blueprint
from .controller import AppState
from .e2e_support import create_fixture_state
from .e2e_support.fakes import (
    INPAINT_PALETTES,
    FakeDepthEstimationModel,
    FakeSegmentationModel,
    FakeUpscaler,
)
from .runtime import build_runtime

SOURCE_SIZE = (320, 240)
# A solid 40x40 mask painted well inside the fixture's opaque region, scaled
# from the same relative position as e2e's drawMaskStroke() (~40%-60% of the
# canvas), with a center far enough from its edges to survive the service's
# GaussianBlur(5, 5)/mask-blur composition without diluting the tolerance
# check below.
MASK_BOX = (140, 100, 180, 140)
MASK_INSIDE = (160, 120)
MASK_OUTSIDE = (10, 10)


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


def _select_slice(client, project_id: str, index: int | None) -> dict:
    response = client.put(
        f"/api/v1/projects/{project_id}/selection", json={"slice": index}
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _mask_image(size=SOURCE_SIZE, box=MASK_BOX) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(box, fill=(255, 255, 255, 255))
    return image


def _save_mask(client, project_id: str, index: int, mask: Image.Image | None = None) -> dict:
    mask = mask if mask is not None else _mask_image()
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    buffer.seek(0)
    response = client.put(
        f"/api/v1/projects/{project_id}/slices/{index}/mask",
        data={"mask": (buffer, "mask.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _generate(
    client, project_id: str, index: int, mode: str, positive: str = "", negative: str = ""
) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/slices/{index}/inpainting/generate",
        json={"mode": mode, "positivePrompt": positive, "negativePrompt": negative},
    )
    assert response.status_code == 202, response.get_json()
    return poll_job(client, response.get_json()["job"]["id"])


def _asset(client, url: str) -> Image.Image:
    response = client.get(url)
    assert response.status_code == 200, response.status_code
    return Image.open(io.BytesIO(response.data)).convert("RGBA")


def _slice_asset(client, project_id: str, index: int) -> Image.Image:
    view = client.get(f"/api/v1/projects/{project_id}").get_json()
    return _asset(client, view["slices"][index]["image"]["url"])


def _distance(rgb, palette_color) -> int:
    return max(abs(a - b) for a, b in zip(rgb, palette_color))


def _deep_interior_point(mask_255: np.ndarray, radius: int) -> tuple[int, int] | None:
    """A point whose ``radius``-neighborhood is entirely inside ``mask_255``.

    Erodes the boolean mask (0/255 uint8) by a ``(2*radius+1)``-square
    structuring element so the returned point (if any) is far enough from any
    mask edge to be unaffected by the service's mask-blur feathering.
    """

    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    eroded = cv2.erode(mask_255, kernel)
    ys, xs = np.where(eroded == 255)
    if xs.size == 0:
        return None
    mid = len(xs) // 2
    return int(xs[mid]), int(ys[mid])


def _failing_inpainting_runtime():
    """A Runtime whose inpainting pipeline always fails to load."""

    class _Failing:
        def load_model(self):
            raise RuntimeError("simulated model load failure")

        def inpaint(self, *args, **kwargs):
            raise RuntimeError("simulated model load failure")

    return build_runtime(
        depth_model_factory=FakeDepthEstimationModel,
        segmentation_model_factory=FakeSegmentationModel,
        inpainting_model_factory=lambda *a, **k: _Failing(),
        upscaler_factory=FakeUpscaler,
    )


# --- Masks -------------------------------------------------------------


def test_mask_save_renders_rgba_and_resizes_to_source_dims(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)

    project_dir = Path.cwd() / project_id
    json_path = project_dir / AppState.STATE_FILE
    json_before = json_path.read_bytes()

    # A half-size canvas mask; the service must BICUBIC-resize it to the
    # source (320x240) dimensions - (70,50)-(90,70) at half scale is the
    # same box as MASK_BOX at full scale.
    mask = _mask_image(size=(160, 120), box=(70, 50, 90, 70))
    result = _save_mask(client, project_id, 1, mask)
    assert result["changed"] is True
    assert result["slices"][1]["mask"] is not None

    # save_mask must not save the project JSON (InpaintingService contract).
    assert json_path.read_bytes() == json_before

    rendered = _asset(client, result["slices"][1]["mask"]["url"])
    assert rendered.size == SOURCE_SIZE

    inside = rendered.getpixel(MASK_INSIDE)
    assert inside[0] > 0
    assert inside == (inside[0], 0, 0, inside[0])  # RGBA (r, 0, 0, r)

    outside = rendered.getpixel(MASK_OUTSIDE)
    assert outside == (0, 0, 0, 0)


def test_mask_save_requires_matching_selected_slice(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)

    buffer = io.BytesIO()
    _mask_image().save(buffer, format="PNG")
    buffer.seek(0)
    response = client.put(
        f"/api/v1/projects/{project_id}/slices/0/mask",
        data={"mask": (buffer, "mask.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


def test_mask_save_requires_the_multipart_field(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/1/mask",
        data={},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_mask_delete_removes_asset_and_is_idempotent(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)

    response = client.delete(f"/api/v1/projects/{project_id}/slices/1/mask")
    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    assert body["slices"][1]["mask"] is None

    missing = client.get(f"/api/v1/projects/{project_id}/assets/mask-1")
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"] == "not_found"

    again = client.delete(f"/api/v1/projects/{project_id}/slices/1/mask")
    assert again.status_code == 200
    assert again.get_json()["changed"] is False


# --- Prompts/model settings ---------------------------------------------


def test_prompts_persist_and_are_a_no_op_when_unchanged(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)

    response = client.put(
        f"/api/v1/projects/{project_id}/slices/1/prompts",
        json={"positivePrompt": "a new prompt", "negativePrompt": "a new exclusion"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    assert body["slices"][1]["positivePrompt"] == "a new prompt"
    assert body["slices"][1]["negativePrompt"] == "a new exclusion"

    unchanged = client.put(
        f"/api/v1/projects/{project_id}/slices/1/prompts",
        json={"positivePrompt": "a new prompt", "negativePrompt": "a new exclusion"},
    )
    assert unchanged.status_code == 200
    assert unchanged.get_json()["changed"] is False


def test_settings_model_change_clears_candidates_unchanged_model_does_not(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)
    job = _generate(client, project_id, 1, "paint")
    assert job["project"]["inpainting"]["candidates"] is not None
    current_model = job["project"]["inpainting"]["model"]

    unchanged = client.put(
        f"/api/v1/projects/{project_id}/inpainting/settings",
        json={"model": current_model},
    )
    assert unchanged.status_code == 200
    assert unchanged.get_json()["changed"] is False
    assert unchanged.get_json()["inpainting"]["candidates"] is not None

    changed = client.put(
        f"/api/v1/projects/{project_id}/inpainting/settings",
        json={"model": "runwayml/stable-diffusion-v1-5", "strength": 0.5, "padding": 10},
    )
    assert changed.status_code == 200
    body = changed.get_json()
    assert body["changed"] is True
    assert body["inpainting"]["candidates"] is None
    assert body["inpainting"]["model"] == "runwayml/stable-diffusion-v1-5"
    assert body["inpainting"]["strength"] == 0.5
    assert body["inpainting"]["padding"] == 10


def test_settings_rejects_out_of_range_values(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.put(
        f"/api/v1/projects/{project_id}/inpainting/settings",
        json={"strength": 5.0},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_api_key_is_never_echoed_back(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    secret = "sk-super-secret-test-value-0000"
    settings = client.put(
        f"/api/v1/projects/{project_id}/inpainting/settings",
        json={"apiKey": secret, "externalServer": "example.invalid:1234"},
    )
    assert settings.status_code == 200

    for response in (
        settings,
        client.get(f"/api/v1/projects/{project_id}"),
        client.get(f"/api/v1/projects/{project_id}/logs"),
    ):
        assert secret not in response.get_data(as_text=True)

    state = AppState.from_cache(project_id)
    assert state.api_key == secret  # persisted server-side, just never echoed
    assert state.server_address == "example.invalid:1234"


def test_workflow_upload_is_stored_and_reported_non_secretly(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]

    response = client.put(
        f"/api/v1/projects/{project_id}/inpainting/workflow",
        data={"workflow": (io.BytesIO(b'{"prompt": {}}'), "workflow.json")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["inpainting"]["hasWorkflow"] is True


# --- Candidates: paint ----------------------------------------------------


def test_paint_generate_three_candidates_and_composition(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)

    raw_before = np.asarray(_slice_asset(client, project_id, 1))
    version_before = client.get(f"/api/v1/projects/{project_id}").get_json()["slices"][1][
        "version"
    ]

    job = _generate(client, project_id, 1, "paint", "deterministic browser test", "deterministic exclusion")
    assert job["status"] == "succeeded", job

    inpainting = job["project"]["inpainting"]
    candidates = inpainting["candidates"]
    assert candidates is not None
    assert candidates["sliceIndex"] == 1
    assert len(candidates["images"]) == 3

    slice_view = job["project"]["slices"][1]
    assert slice_view["positivePrompt"] == "deterministic browser test"
    assert slice_view["negativePrompt"] == "deterministic exclusion"
    assert slice_view["version"] == version_before  # generation must not mutate the slice

    raw_after = np.asarray(_slice_asset(client, project_id, 1))
    assert np.array_equal(raw_before, raw_after)

    candidate0 = _asset(client, candidates["images"][0]["url"])
    colors0 = {tuple(px) for px in np.asarray(candidate0.convert("RGB")).reshape(-1, 3)}
    assert INPAINT_PALETTES[0][0] in colors0
    assert INPAINT_PALETTES[0][1] in colors0

    candidate1 = _asset(client, candidates["images"][1]["url"])
    inside_rgb = candidate1.getpixel(MASK_INSIDE)[:3]
    palette1 = INPAINT_PALETTES[1]
    # The real mask blur/composition intentionally leaves a tiny contribution
    # from the source even at the selected maximum-mask pixel.
    assert min(_distance(inside_rgb, palette1[0]), _distance(inside_rgb, palette1[1])) <= 5

    outside = candidate1.getpixel(MASK_OUTSIDE)
    assert outside == tuple(int(c) for c in raw_before[MASK_OUTSIDE[1], MASK_OUTSIDE[0]])


def test_generation_failure_persists_prompts_and_keeps_state_unchanged(isolated_cwd) -> None:
    del isolated_cwd
    runtime = _failing_inpainting_runtime()
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(runtime), url_prefix="/api/v1")
    client = app.test_client()

    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)

    job = _generate(client, project_id, 1, "paint", "will still be saved", "also saved")
    assert job["status"] == "failed"
    assert "InpaintingModelFailed" in job["error"]

    state = AppState.from_cache(project_id)
    assert state.image_slices[1].positive_prompt == "will still be saved"
    assert state.image_slices[1].negative_prompt == "also saved"

    current = client.get(f"/api/v1/projects/{project_id}").get_json()
    assert current["inpainting"]["candidates"] is None


def test_failed_regeneration_keeps_previous_candidates_and_selection(
    client, monkeypatch
) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)

    first = _generate(client, project_id, 1, "paint", "first", "first-neg")
    assert first["status"] == "succeeded"
    generation_id = first["project"]["inpainting"]["candidates"]["generationId"]

    select = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": generation_id, "candidate": 1},
    )
    assert select.get_json()["inpainting"]["selectedCandidate"] == 1

    state = AppState.from_cache(project_id)
    pipeline = state.pipeline_spec
    assert pipeline is not None

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-generation failure")

    monkeypatch.setattr(pipeline, "inpaint_diffusers", boom)

    second = _generate(client, project_id, 1, "paint", "second", "second-neg")
    assert second["status"] == "failed"
    assert "InpaintingModelFailed" in second["error"]

    current = client.get(f"/api/v1/projects/{project_id}").get_json()
    assert current["inpainting"]["candidates"]["generationId"] == generation_id
    assert current["inpainting"]["selectedCandidate"] == 1
    # Prompts persist even for the failed call, per generate_candidates's contract.
    assert current["slices"][1]["positivePrompt"] == "second"
    assert current["slices"][1]["negativePrompt"] == "second-neg"


# --- Selection -------------------------------------------------------------


def test_select_candidate_toggles_off_and_stale_generation_is_409(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)
    job = _generate(client, project_id, 1, "paint")
    generation_id = job["project"]["inpainting"]["candidates"]["generationId"]

    select = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": generation_id, "candidate": 0},
    )
    assert select.status_code == 200
    assert select.get_json()["inpainting"]["selectedCandidate"] == 0

    toggle_off = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": generation_id, "candidate": 0},
    )
    assert toggle_off.status_code == 200
    assert toggle_off.get_json()["inpainting"]["selectedCandidate"] is None

    stale = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": "0" * 32, "candidate": 0},
    )
    assert stale.status_code == 409
    assert stale.get_json()["error"]["code"] == "stale_revision"


def test_selected_candidate_is_previewed_in_the_main_image(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    selected_view = _select_slice(client, project_id, 1)
    slice_preview = np.asarray(_asset(client, selected_view["mainImage"]["url"]))
    _save_mask(client, project_id, 1)
    job = _generate(client, project_id, 1, "paint")
    candidates = job["project"]["inpainting"]["candidates"]

    selected = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": candidates["generationId"], "candidate": 1},
    ).get_json()
    main = np.asarray(_asset(client, selected["mainImage"]["url"]))
    candidate = np.asarray(_asset(client, candidates["images"][1]["url"]))
    assert np.array_equal(main, candidate)

    cleared = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": candidates["generationId"], "candidate": 1},
    ).get_json()
    assert cleared["inpainting"]["selectedCandidate"] is None
    assert np.array_equal(np.asarray(_asset(client, cleared["mainImage"]["url"])), slice_preview)


def test_select_candidate_out_of_range_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)
    job = _generate(client, project_id, 1, "paint")
    generation_id = job["project"]["inpainting"]["candidates"]["generationId"]

    response = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": generation_id, "candidate": 99},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_slice_selection_change_clears_candidates(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)
    job = _generate(client, project_id, 1, "paint")
    assert job["project"]["inpainting"]["candidates"] is not None

    _select_slice(client, project_id, 0)

    current = client.get(f"/api/v1/projects/{project_id}").get_json()
    assert current["inpainting"]["candidates"] is None


# --- Apply -------------------------------------------------------------


def test_apply_with_stale_generation_id_is_409(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)
    _generate(client, project_id, 1, "paint")

    response = client.post(
        f"/api/v1/projects/{project_id}/slices/1/inpainting/apply",
        json={"generationId": "0" * 32},
    )
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "stale_revision"


def test_apply_writes_new_version_saves_json_and_clears_candidates(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)
    job = _generate(client, project_id, 1, "paint")
    generation_id = job["project"]["inpainting"]["candidates"]["generationId"]

    select = client.put(
        f"/api/v1/projects/{project_id}/inpainting/selection",
        json={"generationId": generation_id, "candidate": 1},
    )
    assert select.get_json()["inpainting"]["selectedCandidate"] == 1

    project_dir = Path.cwd() / project_id
    json_before = (project_dir / AppState.STATE_FILE).read_bytes()

    apply = client.post(
        f"/api/v1/projects/{project_id}/slices/1/inpainting/apply",
        json={"generationId": generation_id},
    )
    assert apply.status_code == 200
    body = apply.get_json()
    assert body["slices"][1]["version"] == 2
    assert body["inpainting"]["candidates"] is None
    assert body["inpainting"]["selectedCandidate"] is None

    assert (project_dir / AppState.STATE_FILE).read_bytes() != json_before

    state = AppState.from_cache(project_id)
    assert Path(state.image_slices[1].filename).name == "image_slice_1_v2.png"

    log = client.get(f"/api/v1/projects/{project_id}/logs").get_json()
    assert any("Inpainting applied to slice 1" in e["message"] for e in log["entries"])


def test_apply_without_a_valid_selection_is_400(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)
    job = _generate(client, project_id, 1, "paint")
    generation_id = job["project"]["inpainting"]["candidates"]["generationId"]

    response = client.post(
        f"/api/v1/projects/{project_id}/slices/1/inpainting/apply",
        json={"generationId": generation_id},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


# --- Candidates: fill -------------------------------------------------------


def test_fill_generate_favors_transparent_regions(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)

    raw_slice = _slice_asset(client, project_id, 1)
    alpha = np.asarray(raw_slice)[:, :, 3]
    transparent_ys, transparent_xs = np.where(alpha == 0)
    assert transparent_ys.size
    transparent_point = (int(transparent_xs[0]), int(transparent_ys[0]))

    # Pick an opaque point deep inside a solid region (well beyond the
    # default mask-blur radius) so the fill composite there is unaffected by
    # edge feathering, mirroring the tolerance e2e scenario 7 relies on.
    opaque_mask = (alpha == 255).astype(np.uint8) * 255
    opaque_point = None
    for radius in (60, 40, 20, 5):
        opaque_point = _deep_interior_point(opaque_mask, radius)
        if opaque_point is not None:
            break
    assert opaque_point is not None, "no opaque region found for slice 1"

    job = _generate(client, project_id, 1, "fill")
    assert job["status"] == "succeeded"
    candidates = job["project"]["inpainting"]["candidates"]
    assert len(candidates["images"]) == 3

    candidate0 = _asset(client, candidates["images"][0]["url"])
    assert candidate0.size == SOURCE_SIZE

    filled = candidate0.getpixel(transparent_point)
    assert filled[3] == 255
    palette0 = INPAINT_PALETTES[0]
    filled_distance = min(
        _distance(filled[:3], palette0[0]), _distance(filled[:3], palette0[1])
    )
    assert filled_distance <= 5

    candidate_opaque = candidate0.getpixel(opaque_point)
    raw_opaque = raw_slice.getpixel(opaque_point)
    assert candidate_opaque[3] == raw_opaque[3] == 255
    opaque_distance = min(
        _distance(candidate_opaque[:3], palette0[0]),
        _distance(candidate_opaque[:3], palette0[1]),
    )
    assert opaque_distance > filled_distance + 50


# --- Candidates: enhance -----------------------------------------------


def test_enhance_two_candidates_same_size_alpha_preserved(client, monkeypatch) -> None:
    # AppState._create_upscaler resolves the *module-level* `Upscaler` symbol
    # in controller.py directly (not runtime.py's upscaler_factory), so only
    # install_fakes() (Dash-only, e2e browser tests) patches it normally; the
    # API's fake runtime doesn't run install_fakes(), so this test patches it
    # the same way that module does, matching this file's other tests' own
    # precedent (see test_api_segmentation.py's selected-slice test).
    monkeypatch.setattr(controller, "Upscaler", FakeUpscaler)

    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)

    raw_slice = _slice_asset(client, project_id, 1)
    alpha = np.asarray(raw_slice)[:, :, 3]
    transparent_ys, transparent_xs = np.where(alpha == 0)
    assert transparent_ys.size
    transparent_point = (int(transparent_xs[0]), int(transparent_ys[0]))

    # FakeUpscaler paints a 4px yellow outline around the doubled image's
    # edge, so pick an opaque pixel on the slice's own top row (y=0), which
    # survives the enhance resize back down to the original dimensions.
    border_row = alpha[0, :]
    border_xs = np.where(border_row == 255)[0]
    assert border_xs.size, "expected an opaque pixel on the slice's top row"
    opaque_point = (int(border_xs[len(border_xs) // 2]), 0)

    job = _generate(client, project_id, 1, "enhance")
    assert job["status"] == "succeeded"
    candidates = job["project"]["inpainting"]["candidates"]
    assert len(candidates["images"]) == 2

    for image_ref in candidates["images"]:
        candidate = _asset(client, image_ref["url"])
        assert candidate.size == SOURCE_SIZE
        assert candidate.getpixel(transparent_point)[3] == 0

        border = candidate.getpixel(opaque_point)
        assert border[3] == 255
        assert border[0] > 200
        assert border[1] > 200
        assert border[2] < 50


# --- Erase ---------------------------------------------------------------


def test_erase_removes_painted_alpha_and_versions_slice(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)
    _save_mask(client, project_id, 1)

    raw_before = np.asarray(_slice_asset(client, project_id, 1))

    response = client.post(f"/api/v1/projects/{project_id}/slices/1/inpainting/erase")
    assert response.status_code == 200
    body = response.get_json()
    assert body["slices"][1]["version"] == 2

    state = AppState.from_cache(project_id)
    assert Path(state.image_slices[1].filename).name == "image_slice_1_v2.png"

    raw_after = np.asarray(_slice_asset(client, project_id, 1))
    assert raw_after[MASK_INSIDE[1], MASK_INSIDE[0], 3] == 0
    assert np.array_equal(
        raw_before[MASK_OUTSIDE[1], MASK_OUTSIDE[0]],
        raw_after[MASK_OUTSIDE[1], MASK_OUTSIDE[0]],
    )

    log = client.get(f"/api/v1/projects/{project_id}/logs").get_json()
    assert any("Inpainting erased for slice 1" in e["message"] for e in log["entries"])


def test_erase_requires_a_saved_mask(client) -> None:
    view = _restore_fixture(client)
    project_id = view["id"]
    _select_slice(client, project_id, 1)

    response = client.post(f"/api/v1/projects/{project_id}/slices/1/inpainting/erase")
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"
