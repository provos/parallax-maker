"""Slice editing and mask tool HTTP routes.

Every route here delegates the actual mutation logic to the existing
``SliceEditingService`` (see ``slice_editing_services.py``) or, for undo/redo,
the existing ``InpaintingService.move_slice_version``; this module is only
responsible for HTTP transport, index/selection validation, and applying each
command's ``preview_image`` to the project's display asset exactly like
Dash's chained ``update_slices`` re-render (see
``SliceEditingService._refresh_selection``).

None of these operations call a slow model/provider, so - like the
selection/multi-point routes in ``segmentation.py`` - every route here runs
synchronously under ``_mutation_guard`` rather than as a background job.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from flask import Blueprint, request
from PIL import Image, UnidentifiedImageError

from ..inpainting_services import (
    InpaintingServiceError,
    MoveSliceVersion,
    SliceVersionDirection,
)
from ..slice_editing_services import (
    AddMaskToSlice,
    BalanceSlices,
    CopyToClipboard,
    CreateSlice,
    DeleteSlice,
    DEFAULT_FEATHER_AMOUNT,
    FeatherMask,
    InvertMask,
    PasteClipboard,
    RemoveMaskFromSlice,
    ReplaceSliceImage,
    SetCheckerboard,
    SetSliceDepth,
    SliceEditingUnchanged,
    refresh_selection_preview,
)
from . import schemas
from .errors import InvalidRequest
from .projects import (
    _build_project_view,
    _load_state,
    _mutation_guard,
    _mutation_response,
    _parse_json_body,
)

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def _apply_preview(record, preview_image) -> None:
    """Set the project's display asset from a command's ``preview_image``.

    ``None`` means "no change" (mirrors Dash's ``update_slices`` returning
    ``no_update`` for ``IMAGE.src`` when nothing is selected) - the display
    asset is left exactly as it was, not reset to the input image.
    """

    if preview_image is not None:
        record.set_display_image(preview_image)


def _validate_slice_index(state, index: int) -> None:
    if index < 0 or index >= len(state.image_slices):
        raise InvalidRequest(f"invalid slice index: {index}")


def register_slice_editing_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    @blueprint.post("/projects/<project_id>/slices/create")
    def create_slice(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            result = runtime.slice_editing_service.create_slice(
                CreateSlice(state_id=project_id)
            )
            _apply_preview(record, result.preview_image)
            record.log.append(
                "Created an empty slice"
                if result.empty
                else "Created a slice from the mask"
            )
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.delete("/projects/<project_id>/slices/<int:index>")
    def delete_slice(project_id: str, index: int):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        _validate_slice_index(state, index)

        with _mutation_guard(record):
            result = runtime.slice_editing_service.delete_slice(
                DeleteSlice(state_id=project_id, slice_index=index)
            )
            record.set_display_image(result.preview_image)
            record.log.append(f"Deleted slice at index {result.slice_index}")
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/slices/<int:index>/add-mask")
    def add_mask_to_slice(project_id: str, index: int):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        _validate_slice_index(state, index)
        if state.selected_slice != index:
            raise InvalidRequest(
                f"slice {index} is not the currently selected slice"
            )

        with _mutation_guard(record):
            result = runtime.slice_editing_service.add_mask_to_slice(
                AddMaskToSlice(state_id=project_id)
            )
            _apply_preview(record, result.preview_image)
            record.log.append(f"Added mask to slice {result.slice_index}")
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/slices/<int:index>/remove-mask")
    def remove_mask_from_slice(project_id: str, index: int):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        _validate_slice_index(state, index)
        if state.selected_slice != index:
            raise InvalidRequest(
                f"slice {index} is not the currently selected slice"
            )

        with _mutation_guard(record):
            result = runtime.slice_editing_service.remove_mask_from_slice(
                RemoveMaskFromSlice(state_id=project_id)
            )
            _apply_preview(record, result.preview_image)
            record.log.append(f"Removed mask from slice {result.slice_index}")
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/clipboard/copy")
    def copy_to_clipboard(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            runtime.slice_editing_service.copy_to_clipboard(
                CopyToClipboard(state_id=project_id)
            )
            record.log.append("Copied mask to clipboard")
            # No display/save side effects in Dash's copy_to_clipboard, but the
            # ProjectView's `clipboard` flag changed, so bump the revision.
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/clipboard/paste")
    def paste_clipboard(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            result = runtime.slice_editing_service.paste_clipboard(
                PasteClipboard(state_id=project_id)
            )
            _apply_preview(record, result.preview_image)
            record.log.append(f"Pasted clipboard to slice {result.slice_index}")
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/slices/balance")
    def balance_slices(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        changed = True
        with _mutation_guard(record):
            try:
                result = runtime.slice_editing_service.balance_slices(
                    BalanceSlices(state_id=project_id)
                )
                _apply_preview(record, result.preview_image)
                record.log.append("Balanced slice depths")
                record.bump_revision()
            except SliceEditingUnchanged:
                changed = False

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)

    @blueprint.put("/projects/<project_id>/slices/<int:index>/depth")
    def set_slice_depth(project_id: str, index: int):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.SetSliceDepthRequest)
        record = runtime.projects.ensure(project_id)
        _validate_slice_index(state, index)

        with _mutation_guard(record):
            result = runtime.slice_editing_service.set_slice_depth(
                SetSliceDepth(state_id=project_id, slice_index=index, depth=payload.depth)
            )
            _apply_preview(record, result.preview_image)
            record.log.append(
                f"Set depth of slice {result.slice_index} to {payload.depth}"
            )
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.put("/projects/<project_id>/slices/<int:index>/image")
    def replace_slice_image(project_id: str, index: int):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        _validate_slice_index(state, index)

        file_storage = request.files.get("image")
        if file_storage is None:
            raise InvalidRequest(
                "multipart field 'image' with the replacement image is required"
            )
        try:
            image = Image.open(io.BytesIO(file_storage.read()))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequest(f"uploaded file is not a valid image: {exc}") from None

        with _mutation_guard(record):
            result = runtime.slice_editing_service.replace_slice_image(
                ReplaceSliceImage(state_id=project_id, slice_index=index, image=image)
            )
            record.bump_input_version()  # the input image was recomposed
            record.log.append(
                f"Received image slice upload for slice {result.slice_index} "
                f"at {result.image_filename}"
            )
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/mask/invert")
    def invert_mask(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            result = runtime.slice_editing_service.invert_mask(
                InvertMask(state_id=project_id)
            )
            record.set_display_image(result.preview_image)
            record.log.append("Inverted mask")
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/mask/feather")
    def feather_mask(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            result = runtime.slice_editing_service.feather_mask(
                FeatherMask(state_id=project_id)
            )
            record.set_display_image(result.preview_image)
            record.log.append(f"Feathered mask by {DEFAULT_FEATHER_AMOUNT} pixels")
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.put("/projects/<project_id>/display")
    def set_checkerboard(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.SetCheckerboardRequest)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            result = runtime.slice_editing_service.set_checkerboard(
                SetCheckerboard(state_id=project_id, enabled=payload.use_checkerboard)
            )
            _apply_preview(record, result.preview_image)
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)

    @blueprint.post("/projects/<project_id>/slices/<int:index>/undo")
    def undo_slice(project_id: str, index: int):
        return _move_slice_version(runtime, project_id, index, SliceVersionDirection.BACKWARD)

    @blueprint.post("/projects/<project_id>/slices/<int:index>/redo")
    def redo_slice(project_id: str, index: int):
        return _move_slice_version(runtime, project_id, index, SliceVersionDirection.FORWARD)


def _move_slice_version(
    runtime: "Runtime", project_id: str, index: int, direction: SliceVersionDirection
):
    state = _load_state(project_id)
    record = runtime.projects.ensure(project_id)
    _validate_slice_index(state, index)

    with _mutation_guard(record):
        try:
            result = runtime.inpainting_service.move_slice_version(
                MoveSliceVersion(
                    state_id=project_id, slice_index=index, direction=direction
                )
            )
        except InpaintingServiceError:
            raise
        verb = "Undid" if direction is SliceVersionDirection.BACKWARD else "Redid"
        record.log.append(f"{verb} change to slice {result.slice_index}")
        # Mirrors undo_slice_request's own STORE_UPDATE_SLICE=True chain: a
        # slice-list mutation, so recompose the preview and clear the mask
        # like every other slice-editing command's downstream update_slices.
        _apply_preview(record, refresh_selection_preview(state))
        record.bump_revision()

    view = _build_project_view(runtime, project_id, state)
    return _mutation_response(view, True)
