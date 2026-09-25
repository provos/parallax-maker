"""Manual 3D-camera navigation route.

Reproduces the removed Dash ``navigate_image`` callback over HTTP; see
``camera_services.navigate_camera`` for the underlying (framework-neutral,
unsaved-to-disk) command. This module is only responsible for HTTP
transport: request validation, the busy/mutation guard, and applying the
command's result to the project's display asset and inpainting-selection
state exactly like the ``selection`` route's own null-deselect handling
(``api/segmentation.py``'s ``update_selection``, ``slice: null`` branch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, request

from ..camera_services import navigate_camera
from ..inpainting_services import ClearInpaintingSelection
from . import schemas
from .projects import (
    _build_project_view,
    _load_state,
    _mutation_guard,
    _mutation_response,
    _parse_json_body,
)

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def register_camera_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    @blueprint.post("/projects/<project_id>/camera/navigate")
    def navigate_camera_route(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.CameraNavigateRequest)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            result = navigate_camera(state, payload.direction)
            if result.preview_image is None:
                # Mirrors navigate_image's early return with zero slices:
                # the selection was already None (a slice can't stay selected
                # once it no longer exists), so there is nothing else to
                # clear or re-render.
                record.log.append("No image slices to navigate")
                changed = False
            else:
                # Mirrors navigate_image's unconditional `state.selected_slice
                # = None` up front, applying the same deselect side effect
                # the selection route's own `slice: null` branch does.
                runtime.inpainting_service.clear_selection(
                    ClearInpaintingSelection(state_id=project_id)
                )
                record.set_inpainting_candidates(None)
                record.set_display_image(result.preview_image)
                record.log.append(
                    f"Navigated to new camera position {result.camera_position}"
                )
                record.bump_revision()
                changed = True

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)
