"""Export/render HTTP routes: glTF export (per-slice depth displacement,
optional DOF, upscaled-texture preference), texture upscaling, animation-frame
rendering, and raw slice downloads.

Every mutating/slow route here delegates to ``ExportService``
(``export_services.py``) and runs as a background job through the same
``_begin_job``/``JobManager`` machinery ``api/projects.py``/``api/inpainting.py``
already use for depth/slice/candidate generation - none of glTF export, texture
upscaling, or animation rendering are fast. The slice-download route is a plain
synchronous file response, like ``api/inpainting.py``'s mask/candidate assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flask import Blueprint, request, send_file

from ..controller import AppState
from ..export_services import (
    ExportGltf,
    ExportServiceError,
    RenderAnimation,
    UpscaleTextures,
)
from . import schemas
from .errors import InvalidRequest, NotFound
from .jobs import Job
from .projects import (
    _begin_job,
    _job_response,
    _load_state,
    _parse_json_body,
)

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def register_export_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    @blueprint.post("/projects/<project_id>/export/gltf")
    def start_gltf_export(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.GltfExportRequest)
        record = runtime.projects.ensure(project_id)
        # Mirrors webui.py's gltf_export: the displacement slider value is
        # whatever was last persisted through PUT .../settings (SLIDER_DISPLACEMENT
        # is saved by the same "persist camera parameters" callback, WEB-30).
        displacement_scale = state.mesh_displacement

        def run(job: Job) -> None:
            with record.lock:
                result = runtime.export_service.export_gltf(
                    ExportGltf(
                        state_id=project_id,
                        displacement_scale=displacement_scale,
                        support_dof=payload.dof,
                    )
                )
                message = f"Exported glTF scene with {result.slice_count} slices"
                if result.used_upscaled:
                    message += " (upscaled textures)"
                if result.generated_depth_maps:
                    message += " (generated per-slice depth maps)"
                record.log.append(message)
            job.set_progress(1.0)

        job = _begin_job(runtime, record, project_id, kind="export-gltf", run=run)
        return _job_response(runtime, job)

    @blueprint.get("/projects/<project_id>/export/gltf")
    def download_gltf(project_id: str):
        _load_state(project_id)
        path = Path.cwd() / project_id / AppState.MODEL_FILE
        if not path.exists():
            raise NotFound("no glTF export is available for this project yet")
        return send_file(
            path,
            mimetype="model/gltf+json",
            as_attachment=True,
            download_name="scene.gltf",
        )

    @blueprint.post("/projects/<project_id>/export/upscale")
    def start_upscale_export(project_id: str):
        _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        settings = record.get_inpainting_settings()
        workflow = (
            record.get_inpainting_workflow() if settings.model == "comfyui" else None
        )

        def run(job: Job) -> None:
            with record.lock:
                result = runtime.export_service.upscale_textures(
                    UpscaleTextures(
                        state_id=project_id,
                        model_name=settings.model,
                        server_address=settings.external_server,
                        api_key=settings.api_key or None,
                        workflow=workflow,
                    )
                )
                record.log.append(
                    f"Upscaled textures for {result.slice_count} slices"
                )
            job.set_progress(1.0)

        job = _begin_job(runtime, record, project_id, kind="upscale", run=run)
        return _job_response(runtime, job)

    @blueprint.post("/projects/<project_id>/export/animation")
    def start_animation_export(project_id: str):
        _load_state(project_id)
        payload = _parse_json_body(request, schemas.AnimationExportRequest)
        record = runtime.projects.ensure(project_id)

        def run(job: Job) -> None:
            with record.lock:
                result = runtime.export_service.render_animation(
                    RenderAnimation(state_id=project_id, num_frames=payload.frames)
                )
                # Matches webui.py's export_animation log line exactly; there is
                # deliberately no browser download here (see PARITY.md "Known
                # quirks" - ANIMATION_OUTPUT exists but no dcc.Download fires).
                record.log.append(
                    f"Exported {result.frame_count} frames to animation"
                )
            job.set_progress(1.0)

        job = _begin_job(runtime, record, project_id, kind="animation", run=run)
        return _job_response(runtime, job)

    @blueprint.get("/projects/<project_id>/slices/<int:index>/download")
    def download_slice(project_id: str, index: int):
        state = _load_state(project_id)
        if index < 0 or index >= len(state.image_slices):
            raise InvalidRequest(f"invalid slice index: {index}")

        try:
            path, name = runtime.export_service.slice_download_path(project_id, index)
        except ExportServiceError as exc:
            raise InvalidRequest(str(exc)) from None
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise NotFound(f"slice {index} has no image on disk")

        return send_file(path, mimetype="image/png", as_attachment=True, download_name=name)
