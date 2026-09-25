"""Segmentation and slice-selection HTTP routes.

Every route here delegates the actual interaction logic to the existing
``SegmentationService``/``InpaintingService`` (see ``segmentation_services.py``
and ``inpainting_services.py``); this module is only responsible for HTTP
transport, reproducing Dash's ``click_event``/``display_slice``/
``toggle_multi_point`` behavior byte-for-byte (log lines, modifier-key
mapping, display-image updates) over the ``/api/v1`` contract instead of Dash
callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, request

from ..controller import AppState, CompositeMode
from ..inpainting_services import ClearInpaintingSelection
from ..segmentation_services import (
    AppliedMaskResult,
    CommitMultiPoint,
    MaskOperation,
    PointPolarity,
    QueuedPointResult,
    SelectDepthPoint,
    SelectInstancePoint,
    SetMultiPointMode,
)
from . import schemas
from .errors import InvalidRequest
from .jobs import Job
from .projects import (
    _begin_job,
    _build_project_view,
    _load_state,
    _mutation_guard,
    _mutation_response,
    _parse_json_body,
    _job_response,
)

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def modifiers_to_operation(shift_key: bool, ctrl_key: bool) -> MaskOperation:
    """Map click modifier keys to a ``MaskOperation`` exactly like Dash's
    ``click_event`` (webui.py:493): Shift wins over Ctrl when both are held.
    """

    if shift_key:
        return MaskOperation.ADD
    if ctrl_key:
        return MaskOperation.SUBTRACT
    return MaskOperation.REPLACE


def modifiers_to_polarity(ctrl_key: bool) -> PointPolarity:
    """Map Ctrl to a ``PointPolarity`` exactly like Dash's ``click_event``
    (webui.py:511-513). Only meaningful for instance-mode/multi-point clicks;
    in single-point instance mode it only affects a *queued* point's polarity,
    since an immediately-applied click always sends a positive point to the
    model and lets ``operation`` (SUBTRACT for a lone Ctrl) remove it from the
    mask instead.
    """

    return PointPolarity.NEGATIVE if ctrl_key else PointPolarity.POSITIVE


def _validate_click_bounds(state: AppState, x: int, y: int) -> None:
    """Reject an out-of-bounds click with 400 before a job is even queued.

    ``SegmentationService`` also validates this (raising
    ``InvalidSegmentationPoint``), but that only surfaces once the background
    job runs and fails - checking synchronously here, using the same bounds,
    gives the client an immediate ``400`` instead of a job to poll.
    """

    if state.imgData is None:
        return
    width, height = state.imgData.size
    if x < 0 or y < 0 or x >= width or y >= height:
        raise InvalidRequest(f"the selected point ({x}, {y}) is outside the image")


def _apply_segmentation_result(record, result, *, mode: str) -> None:
    """Apply a segmentation command's result exactly like Dash's ``click_event``.

    ``QueuedPointResult`` (multi-point queueing) changes neither the display
    image nor the log, matching webui.py:527-528. An ``AppliedMaskResult``
    updates the display image and appends the same log lines Dash does
    (webui.py:533-542): the pixel/depth line whenever a point was clicked
    (both depth- and instance-mode single clicks), plus the committed-points
    line for instance-mode clicks and multi-point commits.
    """

    if isinstance(result, QueuedPointResult):
        return
    assert isinstance(result, AppliedMaskResult)
    record.set_display_image(result.preview_image)
    if result.point is not None:
        record.log.append(
            f"Click event at pixel coordinates ({result.point[0]}, {result.point[1]}) "
            f"at depth {result.depth}"
        )
    if mode == "instance":
        record.log.append(
            f"Committed points {list(result.positive_points)} and "
            f"{list(result.negative_points)} for Segment Anything"
        )


def register_segmentation_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    @blueprint.put("/projects/<project_id>/selection")
    def update_selection(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.SelectionRequest)
        record = runtime.projects.ensure(project_id)

        slice_index = payload.slice
        if slice_index is not None and (
            slice_index < 0 or slice_index >= len(state.image_slices)
        ):
            raise InvalidRequest(f"invalid slice index: {slice_index}")

        changed = True
        with _mutation_guard(record):
            if state.selected_slice == slice_index:
                # Matches the design's click-again-to-deselect note: selecting
                # the already-selected slice is a no-op here (the frontend
                # sends `slice: null` to deselect, unlike Dash's click toggle).
                changed = False
            else:
                state.selected_slice = slice_index
                if slice_index is not None:
                    mode = (
                        CompositeMode.CHECKERBOARD
                        if state.use_checkerboard
                        else CompositeMode.GRAYSCALE
                    )
                    composed = state.slice_image_composed(slice_index, mode=mode)
                    record.set_display_image(composed)
                else:
                    record.set_display_image(None)
                runtime.inpainting_service.clear_selection(
                    ClearInpaintingSelection(state_id=project_id)
                )
                # Dash's display_slice/react_selected_slice_change also clears
                # CTR_INPAINTING_DISPLAY whenever the selected slice changes;
                # our candidate images are new server-side state the service
                # itself doesn't know about, so drop them here too.
                record.set_inpainting_candidates(None)
                record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)

    @blueprint.post("/projects/<project_id>/segmentation/click")
    def segmentation_click(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.SegmentationClickRequest)
        record = runtime.projects.ensure(project_id)
        _validate_click_bounds(state, payload.x, payload.y)

        operation = modifiers_to_operation(payload.shift_key, payload.ctrl_key)
        polarity = modifiers_to_polarity(payload.ctrl_key)
        point = (payload.x, payload.y)
        mode = payload.mode

        def run(job: Job) -> None:
            with record.lock:
                if mode == "instance":
                    result = runtime.segmentation_service.select_instance_point(
                        SelectInstancePoint(
                            state_id=project_id,
                            point=point,
                            operation=operation,
                            polarity=polarity,
                        )
                    )
                else:
                    result = runtime.segmentation_service.select_depth_point(
                        SelectDepthPoint(
                            state_id=project_id, point=point, operation=operation
                        )
                    )
                _apply_segmentation_result(record, result, mode=mode)
            job.set_progress(1.0)

        job = _begin_job(runtime, record, project_id, kind="segmentation", run=run)
        return _job_response(runtime, job)

    @blueprint.post("/projects/<project_id>/segmentation/commit")
    def segmentation_commit(project_id: str):
        _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        def run(job: Job) -> None:
            with record.lock:
                result = runtime.segmentation_service.commit_multi_point(
                    CommitMultiPoint(state_id=project_id)
                )
                _apply_segmentation_result(record, result, mode="instance")
            job.set_progress(1.0)

        job = _begin_job(runtime, record, project_id, kind="segmentation", run=run)
        return _job_response(runtime, job)

    @blueprint.put("/projects/<project_id>/segmentation/multi-point")
    def update_multi_point(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.MultiPointRequest)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            runtime.segmentation_service.set_multi_point_mode(
                SetMultiPointMode(state_id=project_id, enabled=payload.enabled)
            )
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, True)
