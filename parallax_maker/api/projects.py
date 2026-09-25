"""Project/asset/threshold/slice routes: the first Svelte vertical slice.

Every route here delegates the actual workflow logic to the existing
``WorkflowService`` (see ``workflow_services.py``); this module is only
responsible for HTTP transport (multipart/JSON decoding, concurrency control,
asset serving) and building the public ``ProjectView`` projection.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from uuid import uuid4

from flask import Blueprint, Request, jsonify, request
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from ..controller import AppState, CompositeMode
from ..project_services import (
    InvalidProjectFile,
    ProjectDirectoryNotFound,
    RestoreLegacyState,
)
from ..workflow_services import (
    ConfigureThresholds,
    GenerateDepth,
    GenerateSlices,
    UpdateThresholdValues,
    UploadImage,
    WorkflowUnchanged,
)
from . import schemas
from .assets import (
    file_version,
    main_asset_bytes,
    resolve_asset_path,
    send_asset,
    send_bytes,
    slice_thumbnail,
    thumbnail_index,
)
from .errors import Busy, InvalidRequest, NotFound, StaleRevision
from .jobs import Job

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import ProjectRecord, Runtime

#: Matches components.py's SLIDER_NUM_SLICES default; used the first time a
#: project's threshold count is configured (right after upload/depth), the
#: same way Dash's slider default flows into ``update_thresholds`` on the
#: client the first time the depth-map container updates.
DEFAULT_NUM_SLICES = 3


def _validate_project_id(project_id: str) -> None:
    if Path(project_id).name != project_id or not project_id.startswith("appstate-"):
        raise NotFound(f"unknown project: {project_id}")


def _load_state(project_id: str) -> AppState:
    _validate_project_id(project_id)
    try:
        return AppState.from_cache(project_id)
    except (FileNotFoundError, NotADirectoryError):
        raise NotFound(f"unknown project: {project_id}") from None


def _slice_version(image_slice) -> int:
    stem = Path(image_slice.filename).stem
    last = stem.split("_")[-1]
    if last.startswith("v") and last[1:].isdigit():
        return int(last[1:])
    return 1


def _asset_ref(project_id: str, asset_id: str, version: str) -> schemas.AssetRef:
    """URL for ``asset_id`` whose query changes only when its content does."""

    return schemas.AssetRef(
        url=f"/api/v1/projects/{project_id}/assets/{asset_id}?v={version}"
    )


def _slice_mask_ref(
    project_id: str, state: AppState, index: int
) -> schemas.AssetRef | None:
    """The ``mask-{index}`` asset ref, or ``None`` when no mask is saved."""

    path = Path(state.mask_filename(index))
    if not path.exists():
        return None
    return _asset_ref(project_id, f"mask-{index}", file_version(path))


def _build_inpainting_view(
    record: ProjectRecord, state: AppState
) -> schemas.InpaintingView:
    """Project the record's inpainting settings/candidates onto the wire.

    Reads only ``ProjectRecord``'s own in-memory fields (settings/candidates/
    workflow) plus ``state.selected_inpainting``, so this stays a one-way
    dependency from ``api/inpainting.py`` onto this module rather than a
    circular one (see that module's docstring).
    """

    settings = record.get_inpainting_settings()
    candidate_set = record.get_inpainting_candidates()
    candidates_view = None
    if candidate_set is not None:
        candidates_view = schemas.InpaintingCandidatesView(
            generation_id=candidate_set.generation_id,
            slice_index=candidate_set.slice_index,
            images=[
                _asset_ref(
                    record.project_id, f"candidate-{candidate_set.generation_id}-{k}", "1"
                )
                for k in range(len(candidate_set.images))
            ],
        )
    return schemas.InpaintingView(
        # Persisted project values win over per-process defaults (restore).
        model=state.inpainting_model_name or settings.model,
        strength=settings.strength,
        guidance_scale=settings.guidance_scale,
        padding=settings.padding,
        blur=settings.blur,
        external_server=state.server_address or settings.external_server,
        has_workflow=record.get_inpainting_workflow() is not None,
        candidates=candidates_view,
        selected_candidate=state.selected_inpainting,
    )


def _build_project_view(
    runtime: Runtime, project_id: str, state: AppState
) -> schemas.ProjectView:
    record = runtime.projects.ensure(project_id)
    revision = record.revision

    image = None
    if state.imgData is not None:
        image = schemas.ImageSize(
            width=state.imgData.width, height=state.imgData.height
        )

    assets = schemas.ProjectAssets(
        input=(
            _asset_ref(project_id, "input", f"i{record.input_version}")
            if state.imgData is not None
            else None
        ),
        depth=(
            _asset_ref(
                project_id,
                "depth",
                file_version(Path(project_id) / AppState.DEPTH_MAP_FILE),
            )
            if state.depthMapData is not None
            else None
        ),
    )

    main_image = (
        _asset_ref(
            project_id,
            "main",
            f"{record.input_version}.{record.display_version}",
        )
        if state.imgData is not None
        else None
    )

    segmentation = schemas.SegmentationView(
        multi_point_mode=state.multi_point_mode,
        queued_points=[
            schemas.SegmentationPoint(x=point[0], y=point[1], negative=bool(negative))
            for point, negative in state.points_selected
        ],
        slice_pixel=(
            (int(state.slice_pixel[0]), int(state.slice_pixel[1]))
            if state.slice_pixel is not None
            else None
        ),
        slice_pixel_depth=(
            int(state.slice_pixel_depth)
            if state.slice_pixel_depth is not None
            else None
        ),
        has_mask=state.slice_mask is not None,
    )

    slices = [
        schemas.SliceView(
            index=index,
            depth=int(image_slice.depth),
            version=_slice_version(image_slice),
            can_undo=image_slice.can_undo(forward=False),
            can_redo=image_slice.can_undo(forward=True),
            positive_prompt=image_slice.positive_prompt,
            negative_prompt=image_slice.negative_prompt,
            is_ground=image_slice.is_ground_plane,
            image=_asset_ref(
                project_id, f"slice-{index}", file_version(Path(image_slice.filename))
            ),
            thumbnail=_asset_ref(
                project_id,
                f"slice-{index}-thumb",
                file_version(Path(image_slice.filename)),
            ),
            mask=_slice_mask_ref(project_id, state, index),
        )
        for index, image_slice in enumerate(state.image_slices)
    ]

    busy = None
    active_job_id = record.active_job_id
    if active_job_id is not None:
        job = runtime.jobs.get(active_job_id)
        if job is not None:
            busy = schemas.BusyView(job_id=job.id, kind=job.kind)

    depth_model = ""
    if state.depth_estimation_model is not None:
        depth_model = state.depth_estimation_model.model_name

    settings = schemas.ProjectSettingsView(
        dark_mode=state.dark_mode,
        camera=schemas.CameraSettingsView(
            distance=state.camera.camera_distance,
            focal_length=state.camera.focal_length,
            max_distance=state.camera.max_distance,
            pitch=state.camera.pitch,
            ground_near=state.camera.ground_near,
            horizon_row=(
                state.camera.horizon_row(*state.imgData.size)
                if state.imgData is not None
                else None
            ),
        ),
        mesh_displacement=state.mesh_displacement,
        depth_model=state.depth_model_name or "",
    )

    # Not served through the generic /assets/{assetId} route: the client needs
    # a real download (Content-Disposition attachment named "scene.gltf"),
    # which api/export.py's dedicated GET .../export/gltf route provides.
    gltf_path = Path(project_id) / AppState.MODEL_FILE
    exports = schemas.ProjectExportsView(
        gltf=(
            schemas.AssetRef(
                url=f"/api/v1/projects/{project_id}/export/gltf"
                f"?v={file_version(gltf_path)}"
            )
            if gltf_path.exists()
            else None
        ),
        upscaled=any(
            Path(state.upscaled_filename(index)).exists()
            for index in range(len(state.image_slices))
        ),
    )

    return schemas.ProjectView(
        id=project_id,
        revision=revision,
        image=image,
        assets=assets,
        main_image=main_image,
        use_checkerboard=state.use_checkerboard,
        clipboard=state.clipboard_image is not None,
        depth_model=depth_model,
        num_slices=state.num_slices,
        thresholds=list(state.imgThresholds or []),
        slices=slices,
        selected_slice=state.selected_slice,
        segmentation=segmentation,
        inpainting=_build_inpainting_view(record, state),
        busy=busy,
        settings=settings,
        exports=exports,
    )


def _view_json(view: schemas.ProjectView) -> dict:
    return view.model_dump(mode="json", by_alias=True)


def _mutation_response(view: schemas.ProjectView, changed: bool):
    payload = _view_json(view)
    payload["changed"] = changed
    return jsonify(payload), 200


def _job_view(runtime: Runtime, job: Job) -> schemas.JobView:
    from .jobs import JobStatus

    job = job.snapshot()
    project = None
    if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
        try:
            state = AppState.from_cache(job.project_id)
        except (FileNotFoundError, NotADirectoryError):
            state = None
        if state is not None:
            project = _build_project_view(runtime, job.project_id, state)

    return schemas.JobView(
        id=job.id,
        kind=job.kind,
        status=job.status.value,
        progress=job.progress,
        error=job.error,
        project=project,
    )


def _job_response(runtime: Runtime, job: Job):
    payload = schemas.JobRef(job=_job_view(runtime, job)).model_dump(
        mode="json", by_alias=True
    )
    return jsonify(payload), 202


@contextmanager
def _mutation_guard(record: ProjectRecord) -> Iterator[None]:
    """Reject a synchronous mutation with 409 busy if a job is active.

    Otherwise reserves the busy slot for the duration of the mutation (using
    the same bookkeeping a background job would) and holds the project lock
    while it runs, per "every mutating request takes the project lock".
    """

    reservation_id = f"sync-{uuid4().hex}"
    if not record.try_begin_job(reservation_id):
        raise Busy("the project is busy with another operation")
    try:
        with record.lock:
            yield
    finally:
        record.end_job()


def _begin_job(
    runtime: Runtime, record: ProjectRecord, project_id: str, *, kind: str, run
) -> Job:
    reserved_id = uuid4().hex
    if not record.try_begin_job(reserved_id):
        raise Busy("the project is busy with another operation")

    def wrapped(job: Job) -> None:
        try:
            run(job)
        except Exception as exc:
            record.log.append(f"{kind} failed: {exc}", level="error")
            raise
        finally:
            # Bump even on failure: a job may have mutated state before failing.
            record.bump_revision()
            record.end_job()

    return runtime.jobs.submit(kind, project_id, wrapped, job_id=reserved_id)


def _parse_json_body(req: Request, model_cls):
    payload = req.get_json(silent=True)
    if payload is None:
        raise InvalidRequest("request body must be JSON")
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise InvalidRequest(f"invalid request body: {exc}") from None


def register_project_routes(blueprint: Blueprint, runtime: Runtime) -> None:
    @blueprint.post("/projects")
    def upload_project():
        file_storage = request.files.get("image")
        if file_storage is None:
            raise InvalidRequest(
                "multipart field 'image' with the source image is required"
            )

        try:
            image = Image.open(io.BytesIO(file_storage.read()))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequest(f"uploaded file is not a valid image: {exc}") from None

        result = runtime.workflow_service.upload_image(UploadImage(image=image))
        state = AppState.from_cache(result.state_id)
        state.num_slices = DEFAULT_NUM_SLICES

        record = runtime.projects.ensure(result.state_id)
        record.bump_input_version()
        record.set_display_image(None)
        record.log.append(
            f"Uploaded image ({state.imgData.width}x{state.imgData.height})"
        )

        view = _build_project_view(runtime, result.state_id, state)
        return jsonify(_view_json(view)), 201

    @blueprint.post("/projects/restore")
    def restore_project():
        file_storage = request.files.get("state")
        if file_storage is None:
            raise InvalidRequest(
                "multipart field 'state' with the legacy appstate.json is required"
            )

        try:
            raw = file_storage.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidRequest(f"state file is not valid JSON: {exc}") from None

        try:
            result = runtime.project_service.restore_legacy_state(
                RestoreLegacyState(raw_json=raw)
            )
        except ProjectDirectoryNotFound as exc:
            raise NotFound(str(exc)) from None
        except InvalidProjectFile as exc:
            raise InvalidRequest(str(exc)) from None
        state = result.state

        record = runtime.projects.ensure(state.filename)
        record.bump_input_version()
        record.set_display_image(None)
        record.bump_revision()
        record.log.append(f"Restored state from {state.filename}")

        view = _build_project_view(runtime, state.filename, state)
        return jsonify(_view_json(view)), 200

    @blueprint.get("/projects/<project_id>")
    def get_project(project_id: str):
        state = _load_state(project_id)
        view = _build_project_view(runtime, project_id, state)
        return jsonify(_view_json(view)), 200

    @blueprint.post("/projects/<project_id>/depth")
    def start_depth_job(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.DepthRequest)
        record = runtime.projects.ensure(project_id)

        def run(job: Job) -> None:
            runtime.progress_reporter.bind(job)
            try:
                with record.lock:
                    runtime.workflow_service.generate_depth(
                        GenerateDepth(state_id=project_id, model_name=payload.model)
                    )
                    state.depth_model_name = payload.model
                    try:
                        result = runtime.workflow_service.configure_thresholds(
                            ConfigureThresholds(
                                state_id=project_id, num_slices=state.num_slices
                            )
                        )
                        record.log.append(f"Thresholds: {result.thresholds}")
                    except WorkflowUnchanged:
                        pass
                    record.log.append(f"Generated depth map using {payload.model}")
                    record.set_display_image(None)
            finally:
                runtime.progress_reporter.clear()

        job = _begin_job(runtime, record, project_id, kind="depth", run=run)
        return _job_response(runtime, job)

    @blueprint.put("/projects/<project_id>/slice-count")
    def update_slice_count(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.SliceCountRequest)
        record = runtime.projects.ensure(project_id)

        changed = True
        with _mutation_guard(record):
            try:
                result = runtime.workflow_service.configure_thresholds(
                    ConfigureThresholds(
                        state_id=project_id, num_slices=payload.num_slices
                    )
                )
                record.log.append(f"Thresholds: {result.thresholds}")
                record.bump_revision()
            except WorkflowUnchanged:
                changed = False

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)

    @blueprint.put("/projects/<project_id>/thresholds")
    def update_thresholds_route(project_id: str):
        state = _load_state(project_id)
        payload = _parse_json_body(request, schemas.ThresholdsRequest)
        record = runtime.projects.ensure(project_id)

        if payload.base_revision != record.revision:
            raise StaleRevision(
                f"expected revision {record.revision}, got {payload.base_revision}"
            )

        changed = True
        with _mutation_guard(record):
            if payload.base_revision != record.revision:
                raise StaleRevision(
                    f"expected revision {record.revision}, got {payload.base_revision}"
                )
            try:
                result = runtime.workflow_service.update_threshold_values(
                    UpdateThresholdValues(
                        state_id=project_id,
                        values=payload.values,
                        num_slices=state.num_slices,
                    )
                )
                record.log.append(f"Thresholds: {result.values}")
                record.bump_revision()
            except WorkflowUnchanged:
                changed = False

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, changed)

    @blueprint.post("/projects/<project_id>/slices")
    def start_slices_job(project_id: str):
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)

        def run(job: Job) -> None:
            with record.lock:
                result = runtime.workflow_service.generate_slices(
                    GenerateSlices(state_id=project_id)
                )
                record.log.append(f"Generated {result.slice_count} image slices")
                # Mirrors Dash's update_slices: only touch the display/selection
                # interaction state when a slice is (still) selected; otherwise
                # leave whatever is currently displayed alone.
                if state.selected_slice is not None:
                    mode = (
                        CompositeMode.CHECKERBOARD
                        if state.use_checkerboard
                        else CompositeMode.GRAYSCALE
                    )
                    composed = state.slice_image_composed(
                        state.selected_slice, mode=mode
                    )
                    record.set_display_image(composed)
                    state.slice_pixel = None
                    state.slice_pixel_depth = None
                    state.slice_mask = None
            job.set_progress(1.0)

        job = _begin_job(runtime, record, project_id, kind="slices", run=run)
        return _job_response(runtime, job)

    @blueprint.get("/jobs/<job_id>")
    def get_job(job_id: str):
        job = runtime.jobs.get(job_id)
        if job is None:
            raise NotFound(f"unknown job: {job_id}")
        payload = _job_view(runtime, job).model_dump(mode="json", by_alias=True)
        return jsonify(payload), 200

    @blueprint.get("/projects/<project_id>/assets/<asset_id>")
    def get_asset(project_id: str, asset_id: str):
        state = _load_state(project_id)
        project_dir = Path.cwd() / project_id
        if asset_id == "main":
            record = runtime.projects.ensure(project_id)
            data = main_asset_bytes(record, project_dir, state)
            return send_bytes(data, "image/png", request)
        index = thumbnail_index(asset_id)
        if index is not None:
            data = slice_thumbnail(project_dir, state, index)
            return send_bytes(data, "image/png", request)
        # Deferred import: api.inpainting imports several helpers from this
        # module at its own top level (mirroring api.segmentation's existing
        # pattern), so importing it back here at module scope would cycle;
        # by request time (long after both modules have finished loading)
        # that is no longer a concern.
        from .inpainting import (
            candidate_asset_bytes,
            candidate_asset_ids,
            mask_asset_bytes,
            mask_asset_index,
        )

        mask_index = mask_asset_index(asset_id)
        if mask_index is not None:
            data = mask_asset_bytes(project_dir, state, mask_index)
            return send_bytes(data, "image/png", request)
        candidate_ids = candidate_asset_ids(asset_id)
        if candidate_ids is not None:
            record = runtime.projects.ensure(project_id)
            data = candidate_asset_bytes(record, *candidate_ids)
            return send_bytes(data, "image/png", request)
        path = resolve_asset_path(project_dir, state, asset_id)
        return send_asset(path, request)

    @blueprint.get("/projects/<project_id>/logs")
    def get_logs(project_id: str):
        _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        after = request.args.get("after", default=0, type=int) or 0
        entries = record.log.after(after)
        next_seq = entries[-1].seq if entries else after
        payload = schemas.LogsView(
            entries=[
                schemas.LogEntryView(seq=e.seq, level=e.level, message=e.message)
                for e in entries
            ],
            next=next_seq,
        )
        return jsonify(payload.model_dump(mode="json", by_alias=True)), 200
