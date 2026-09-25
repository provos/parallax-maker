"""Project lifecycle HTTP routes: full save, legacy state-file download, and
persisted settings (depth model, camera, mesh displacement, dark mode).

Every route here delegates the actual mutation to ``ProjectService``
(``project_services.py``); this module is only responsible for HTTP
transport, following the same ``_mutation_guard``/``_build_project_view``
pattern as ``api/slice_editing.py``. None of these operations call a slow
model/provider, so - like slice-editing/mask-tool routes - every mutating
route here runs synchronously rather than as a background job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, Response, jsonify, request

from ..project_services import SaveProject, UpdateSettings
from . import schemas
from .projects import (
    _build_project_view,
    _load_state,
    _mutation_guard,
    _mutation_response,
    _parse_json_body,
    _view_json,
)

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def register_project_lifecycle_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    @blueprint.post("/projects/<project_id>/save")
    def save_project(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        with _mutation_guard(record):
            runtime.project_service.save_project(SaveProject(state_id=project_id))
            record.log.append(f"Saved state to {project_id}")

        view = _build_project_view(runtime, project_id, state)
        return jsonify(_view_json(view)), 200

    @blueprint.get("/projects/<project_id>/state-file")
    def download_state_file(project_id: str):
        """The exact JSON payload ``POST /projects/restore`` accepts (Dash's
        Load State control), served as a download - not the on-disk file, so
        this always reflects the project's current in-memory state even if
        "Save" hasn't been clicked since the last mutation."""

        state = _load_state(project_id)
        payload = state.to_json().encode("utf-8")
        response = Response(payload, mimetype="application/json")
        response.headers["Content-Disposition"] = 'attachment; filename="appstate.json"'
        return response

    @blueprint.put("/projects/<project_id>/settings")
    def update_settings(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.ProjectSettingsRequest)
        record = runtime.projects.ensure(project_id)

        changed = False
        with _mutation_guard(record):
            result = runtime.project_service.update_settings(
                UpdateSettings(
                    state_id=project_id,
                    depth_model=payload.depth_model,
                    camera_distance=(
                        payload.camera.distance if payload.camera is not None else None
                    ),
                    focal_length=(
                        payload.camera.focal_length
                        if payload.camera is not None
                        else None
                    ),
                    max_distance=(
                        payload.camera.max_distance
                        if payload.camera is not None
                        else None
                    ),
                    pitch=payload.camera.pitch if payload.camera is not None else None,
                    mesh_displacement=payload.mesh_displacement,
                    dark_mode=payload.dark_mode,
                )
            )
            changed = result.changed
            if changed:
                record.log.append("Updated project settings")
                record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)
