"""Canvas-mask and inpainting HTTP routes.

Every route here delegates the actual mutation to the existing
``InpaintingService`` (see ``inpainting_services.py``); this module is only
responsible for HTTP transport (multipart mask/workflow decoding, asset
serving, candidate-image storage) reproducing the Dash adapters'
byte-for-byte behavior (``components.make_canvas_callbacks``,
``make_inpainting_container_callbacks``, ``make_configuration_callbacks``,
and ``webui.update_prompt_text``/``remember_inpaint_model``) over the
``/api/v1`` contract instead of Dash callbacks.

Note on imports: this module imports several private helpers from
``api/projects.py`` at module scope (mirroring ``api/segmentation.py``'s
existing pattern), so ``api/projects.py`` only ever imports back from here
with a *deferred* (function-body) import inside ``get_asset`` to avoid a
circular import; see that function's comment.

Candidate storage
------------------

``InpaintingService.generate_candidates`` returns PIL images but has no
notion of a server-held asset - Dash encodes them straight into browser data
URLs. For the API, a successful generation is stored as an
``InpaintingCandidateSet`` (``runtime.py``) on the project's
``ProjectRecord``, keyed by a fresh ``generationId`` and bound to the slice
index/version it was generated from. ``ProjectView.inpainting.candidates``
exposes it as ``candidate-{generationId}-{k}`` assets; selecting/clearing a
candidate still goes through ``InpaintingService.select_candidate``/
``clear_selection`` exactly as Dash does (selection lives on ``AppState``,
not in the candidate set). Applying a candidate set requires the same
``generationId`` *and* that the slice index/version still match what it was
generated from, else ``409 stale_revision`` - this is what lets old
candidates safely outlive a failed regeneration (per
``InpaintingService.generate_candidates``'s own contract) while still
rejecting a stale apply after the slice has moved on (a new version, a
different selected slice, or a newer generation).
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from flask import Blueprint, jsonify, request
from PIL import Image, UnidentifiedImageError

from ..controller import AppState, CompositeMode
from ..inpainting_services import (
    ApplyInpaintingCandidate,
    CachedInpaintingStateRepository,
    ClearInpaintingSelection,
    DeleteInpaintingMask,
    EraseInpainting,
    GenerateInpaintingCandidates,
    InpaintingMode,
    InpaintingUnchanged,
    SaveInpaintingMask,
    SelectInpaintingCandidate,
    UpdateInpaintingModel,
    UpdateInpaintingPrompts,
)
from . import schemas
from .assets import _contained
from .errors import InvalidRequest, NotFound, NotReady, StaleRevision
from .jobs import Job
from .projects import (
    _begin_job,
    _build_project_view,
    _job_response,
    _load_state,
    _mutation_guard,
    _mutation_response,
    _parse_json_body,
    _slice_version,
    _view_json,
)

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import ProjectRecord, Runtime

_MASK_ASSET_RE = re.compile(r"^mask-(\d+)$")
_CANDIDATE_ASSET_RE = re.compile(r"^candidate-([0-9a-f]{32})-(\d+)$")

_GENERATE_MODES = {
    "paint": InpaintingMode.PAINT,
    "fill": InpaintingMode.FILL,
    "enhance": InpaintingMode.ENHANCE,
}


# --- Asset resolution (GET /projects/{id}/assets/{assetId}) ----------------
#
# Reads bypass InpaintingService entirely and go straight at the filesystem/
# in-memory record, the same way slice/depth/thumbnail assets do in
# api/assets.py: the service's mask commands always act on
# ``state.selected_slice`` (ignoring any index a caller passes), which is
# right for a mutation but wrong for asset serving, since a client must be
# able to fetch e.g. slice 0's mask while slice 1 is selected.


def mask_asset_index(asset_id: str) -> int | None:
    """Return the slice index for a ``mask-{i}`` asset id, else ``None``."""

    match = _MASK_ASSET_RE.match(asset_id)
    return int(match.group(1)) if match else None


def candidate_asset_ids(asset_id: str) -> tuple[str, int] | None:
    """Return ``(generationId, k)`` for a ``candidate-{gen}-{k}`` asset id."""

    match = _CANDIDATE_ASSET_RE.match(asset_id)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def mask_asset_bytes(project_dir: Path, state: AppState, index: int) -> bytes:
    """PNG bytes for the ``mask-{index}`` asset.

    Renders the saved L-mode mask as RGBA ``(r, 0, 0, r)``, exactly matching
    ``components.make_canvas_callbacks``'s ``load_canvas_mask`` adapter, so a
    browser can paint an existing mask back onto its canvas.
    """

    if index < 0 or index >= len(state.image_slices):
        raise NotFound(f"unknown slice index: {index}")
    path = _contained(project_dir, Path(state.mask_filename(index)))
    if not path.exists():
        raise NotFound(f"no mask found for slice {index}")
    with Image.open(path) as source:
        r = source.convert("L")
    zero_channel = Image.new("L", r.size)
    rendered = Image.merge("RGBA", (r, zero_channel, zero_channel, r))
    buffer = io.BytesIO()
    rendered.save(buffer, format="PNG")
    return buffer.getvalue()


def candidate_asset_bytes(
    record: "ProjectRecord", generation_id: str, index: int
) -> bytes:
    """PNG bytes for the ``candidate-{generationId}-{index}`` asset."""

    candidates = record.get_inpainting_candidates()
    if (
        candidates is None
        or candidates.generation_id != generation_id
        or index < 0
        or index >= len(candidates.images)
    ):
        raise NotFound(
            f"unknown inpainting candidate asset: candidate-{generation_id}-{index}"
        )
    buffer = io.BytesIO()
    candidates.images[index].save(buffer, format="PNG")
    return buffer.getvalue()


def _require_selected_slice(state: AppState, index: int) -> None:
    """Validate ``index`` before a mutation that (like every Dash adapter it
    replaces) always acts on ``state.selected_slice`` regardless of any index
    the caller passes to the underlying ``InpaintingService`` command.

    Raises ``InvalidRequest`` (400) for an out-of-range index and
    ``NotReady`` (409) when ``index`` is not the currently selected slice, so
    a client whose view of the selection has gone stale gets a clear error
    instead of silently mutating a different slice than the one named in the
    URL.
    """

    if index < 0 or index >= len(state.image_slices):
        raise InvalidRequest(f"unknown slice index: {index}")
    if state.selected_slice != index:
        raise NotReady(
            f"slice {index} is not the selected slice; select it before "
            "performing inpainting operations on it"
        )


def _refresh_display_image(record: "ProjectRecord", state: AppState, index: int) -> None:
    """Recompose the ``main`` asset for ``index``, mirroring Dash's
    ``update_slices`` re-render (webui.py:1013-1026) after apply/erase change
    the selected slice's pixels.
    """

    mode = (
        CompositeMode.CHECKERBOARD if state.use_checkerboard else CompositeMode.GRAYSCALE
    )
    record.set_display_image(state.slice_image_composed(index, mode=mode))


def register_inpainting_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    # Deferred: only needed to construct InpaintingCandidateSet instances at
    # request time. A real top-level ``from ..runtime import ...`` here would
    # cycle back through runtime.py's own ``from .api.jobs import ...`` while
    # api/__init__.py is still mid-import; by the time this function actually
    # runs (called from create_api_blueprint(runtime), which requires a
    # already-constructed Runtime), runtime.py is guaranteed fully loaded.
    from ..runtime import InpaintingCandidateSet

    # --- Masks ---------------------------------------------------------

    @blueprint.put("/projects/<project_id>/slices/<int:index>/mask")
    def save_inpainting_mask(project_id: str, index: int):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        _require_selected_slice(state, index)

        file_storage = request.files.get("mask")
        if file_storage is None:
            raise InvalidRequest(
                "multipart field 'mask' with the canvas mask PNG is required"
            )
        try:
            image = Image.open(io.BytesIO(file_storage.read())).convert("RGBA")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequest(f"uploaded mask is not a valid image: {exc}") from None

        # `cropToRegion` mirrors Dash's "Crop to region of interest" checkbox
        # (components.py's CHECKLIST_REGION_OF_INTEREST, CMP-24's
        # `show_crop_region`): when true (the default, matching that
        # checkbox's own `value=["crop"]` default), the response's
        # `boundingBox` is the mask's own padded/squared bounding box for
        # PreviewOverlay.svelte's ROI preview (Dash's CLI-07); when false, no
        # bounding box is computed at all, exactly like Dash.
        crop_to_region = request.form.get("cropToRegion", "true").strip().lower() != "false"

        with _mutation_guard(record):
            settings = record.get_inpainting_settings()
            result = runtime.inpainting_service.save_mask(
                SaveInpaintingMask(
                    state_id=project_id,
                    canvas_image=image,
                    padding=settings.padding,
                    show_crop_region=crop_to_region,
                )
            )
            record.log.append(
                f"Saved mask for slice {result.slice_index} to {result.mask_filename}"
            )
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        payload = _view_json(view)
        payload["changed"] = True
        # `find_square_bounding_box` returns numpy integer types (via
        # `np.nonzero(...).min()/.max()`), which the stdlib JSON encoder
        # cannot serialize directly - cast to plain `int`s.
        payload["boundingBox"] = (
            [int(v) for v in result.bounding_box] if result.bounding_box else None
        )
        return jsonify(payload), 200

    @blueprint.delete("/projects/<project_id>/slices/<int:index>/mask")
    def delete_inpainting_mask(project_id: str, index: int):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        _require_selected_slice(state, index)

        with _mutation_guard(record):
            result = runtime.inpainting_service.delete_mask(
                DeleteInpaintingMask(state_id=project_id)
            )
            if result.deleted:
                record.log.append(f"Deleted mask for slice {result.slice_index}")
                record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, result.deleted)

    # --- Prompts/model settings -----------------------------------------

    @blueprint.put("/projects/<project_id>/slices/<int:index>/prompts")
    def update_inpainting_prompts(project_id: str, index: int):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.InpaintingPromptsRequest)
        record = runtime.projects.ensure(project_id)
        _require_selected_slice(state, index)

        changed = True
        with _mutation_guard(record):
            try:
                result = runtime.inpainting_service.update_prompts(
                    UpdateInpaintingPrompts(
                        state_id=project_id,
                        positive_prompt=payload.positive_prompt,
                        negative_prompt=payload.negative_prompt,
                    )
                )
                record.log.append(
                    f"Updated inpainting prompts for slice {result.slice_index}"
                )
                record.bump_revision()
            except InpaintingUnchanged:
                changed = False

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)

    @blueprint.put("/projects/<project_id>/inpainting/settings")
    def update_inpainting_settings(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.InpaintingSettingsRequest)
        record = runtime.projects.ensure(project_id)

        # An explicit null means "leave unchanged", not "store None".
        provided = {
            key: value
            for key, value in payload.model_dump(exclude_unset=True).items()
            if value is not None
        }
        changed = False
        with _mutation_guard(record):
            if "model" in provided:
                try:
                    runtime.inpainting_service.update_model(
                        UpdateInpaintingModel(
                            state_id=project_id, model_name=provided["model"]
                        )
                    )
                    record.log.append(f"Inpainting model set to {provided['model']}")
                    # Mirrors Dash's remember_inpaint_model, which clears
                    # CTR_INPAINTING_DISPLAY whenever the model actually
                    # changes (but not on InpaintingUnchanged, per its own
                    # try/except PreventUpdate).
                    record.set_inpainting_candidates(None)
                    changed = True
                except InpaintingUnchanged:
                    pass

            if (
                "external_server" in provided
                and state.server_address != provided["external_server"]
            ) or ("api_key" in provided and state.api_key != provided["api_key"]):
                if "external_server" in provided:
                    state.server_address = provided["external_server"]
                if "api_key" in provided:
                    state.api_key = provided["api_key"]
                CachedInpaintingStateRepository().save(
                    project_id, state, runtime.inpainting_service.JSON_ONLY
                )
                changed = True

            settings_fields = {
                key: value
                for key, value in provided.items()
                if key
                in {
                    "model",
                    "strength",
                    "guidance_scale",
                    "padding",
                    "blur",
                    "external_server",
                    "api_key",
                }
            }
            if settings_fields:
                current_settings = record.get_inpainting_settings()
                # "model" is applied unconditionally below (so the record
                # always reflects the caller's last requested value even when
                # InpaintingService.update_model treated it as a no-op above),
                # but must not by itself mark the request as "changed" - only
                # a field whose value actually differs does.
                if any(
                    getattr(current_settings, key) != value
                    for key, value in settings_fields.items()
                ):
                    changed = True
                record.update_inpainting_settings(**settings_fields)

            if changed:
                record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)

    @blueprint.put("/projects/<project_id>/inpainting/workflow")
    def upload_inpainting_workflow(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        file_storage = request.files.get("workflow")
        if file_storage is None:
            raise InvalidRequest(
                "multipart field 'workflow' with the ComfyUI workflow JSON "
                "is required"
            )
        data = file_storage.read()
        if not data:
            raise InvalidRequest("the uploaded workflow is empty")

        with _mutation_guard(record):
            record.set_inpainting_workflow(data)
            record.log.append("Uploaded ComfyUI workflow")
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    # --- Candidates ------------------------------------------------------

    @blueprint.post("/projects/<project_id>/slices/<int:index>/inpainting/generate")
    def generate_inpainting_candidates(project_id: str, index: int):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.InpaintingGenerateRequest)
        record = runtime.projects.ensure(project_id)
        _require_selected_slice(state, index)

        mode = _GENERATE_MODES[payload.mode]
        settings = record.get_inpainting_settings()
        # The persisted project model wins (e.g. after a restore).
        model_name = state.inpainting_model_name or settings.model
        workflow = record.get_inpainting_workflow() if model_name == "comfyui" else None

        def run(job: Job) -> None:
            with record.lock:
                result = runtime.inpainting_service.generate_candidates(
                    GenerateInpaintingCandidates(
                        state_id=project_id,
                        mode=mode,
                        model_name=model_name,
                        workflow=workflow,
                        positive_prompt=payload.positive_prompt,
                        negative_prompt=payload.negative_prompt,
                        strength=settings.strength,
                        guidance_scale=settings.guidance_scale,
                        padding=settings.padding,
                        blur=settings.blur,
                        # Stop short of 1.0: the job only completes once
                        # the candidate set is stored below.
                        progress=lambda fraction: job.set_progress(0.99 * fraction),
                    )
                )
                # Only replace the stored candidate set once generation has
                # fully succeeded, so a failure leaves the old set (and its
                # selection) in place - InpaintingService.generate_candidates
                # already guarantees this for state.selected_inpainting; this
                # mirrors it for our own server-side image storage.
                candidate_set = InpaintingCandidateSet(
                    generation_id=uuid4().hex,
                    slice_index=result.slice_index,
                    slice_version=_slice_version(state.image_slices[result.slice_index]),
                    images=result.candidates,
                )
                record.set_inpainting_candidates(candidate_set)
                record.log.append(
                    f"Generated {len(result.candidates)} inpainting candidates "
                    f"for slice {result.slice_index}"
                )
            # No final set_progress(1.0): on a cancellable job it would raise for
            # a cancel that arrived after the new candidates were stored. The
            # job reaches 1.0 when it is marked succeeded.

        job = _begin_job(
            runtime, record, project_id, kind="inpainting", run=run, cancellable=True
        )
        return _job_response(runtime, job)

    @blueprint.put("/projects/<project_id>/inpainting/selection")
    def update_inpainting_selection(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.InpaintingSelectionRequest)
        record = runtime.projects.ensure(project_id)

        changed = False
        with _mutation_guard(record):
            candidates = record.get_inpainting_candidates()
            if candidates is None or candidates.generation_id != payload.generation_id:
                raise StaleRevision(
                    "the inpainting candidate generation is no longer current"
                )
            if payload.candidate is None:
                result = runtime.inpainting_service.clear_selection(
                    ClearInpaintingSelection(state_id=project_id)
                )
                changed = result.previously_selected is not None
            else:
                # Raises InvalidInpaintingCandidate (-> 400) for an
                # out-of-range index; toggles the same index off again, per
                # InpaintingService.select_candidate's own contract.
                runtime.inpainting_service.select_candidate(
                    SelectInpaintingCandidate(
                        state_id=project_id,
                        candidate_index=payload.candidate,
                        candidate_count=len(candidates.images),
                    )
                )
                changed = True
            if changed:
                # Like Dash's select_inpainting_image: the main image previews
                # the selected candidate, and the slice composite otherwise.
                selected = state.selected_inpainting
                if selected is not None:
                    record.set_display_image(candidates.images[selected])
                elif candidates.slice_index < len(state.image_slices):
                    _refresh_display_image(record, state, candidates.slice_index)
                record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)

    @blueprint.post("/projects/<project_id>/slices/<int:index>/inpainting/apply")
    def apply_inpainting_candidate(project_id: str, index: int):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.InpaintingApplyRequest)
        record = runtime.projects.ensure(project_id)
        _require_selected_slice(state, index)

        with _mutation_guard(record):
            candidates = record.get_inpainting_candidates()
            current_version = _slice_version(state.image_slices[index])
            if (
                candidates is None
                or candidates.generation_id != payload.generation_id
                or candidates.slice_index != index
                or candidates.slice_version != current_version
            ):
                raise StaleRevision(
                    "the inpainting candidate generation is no longer current"
                )
            # Raises InvalidInpaintingCandidate (-> 400) when nothing valid is
            # selected, per InpaintingService.apply_candidate's own contract.
            result = runtime.inpainting_service.apply_candidate(
                ApplyInpaintingCandidate(
                    state_id=project_id, candidates=candidates.images
                )
            )
            # Dash clears CTR_INPAINTING_DISPLAY after a successful apply
            # (apply_inpainting's STORE_INPAINTING output re-triggers
            # react_selected_slice_change, components.py:916-936).
            record.set_inpainting_candidates(None)
            record.log.append(
                f"Inpainting applied to slice {result.slice_index} with new "
                f"image {result.image_filename}"
            )
            _refresh_display_image(record, state, index)
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/slices/<int:index>/inpainting/erase")
    def erase_inpainting(project_id: str, index: int):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        _require_selected_slice(state, index)

        with _mutation_guard(record):
            result = runtime.inpainting_service.erase(
                EraseInpainting(state_id=project_id)
            )
            record.log.append(f"Inpainting erased for slice {result.slice_index}")
            # Unlike apply, Dash's erase_inpainting does not re-trigger
            # react_selected_slice_change (no STORE_INPAINTING output), so
            # any displayed candidates are deliberately left alone here too.
            _refresh_display_image(record, state, index)
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)
