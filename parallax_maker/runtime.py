"""Composition root for the HTTP API and, eventually, the Svelte frontend.

``Runtime`` bundles everything the API blueprint needs that used to live as
module globals in :mod:`parallax_maker.webui`: the model/pipeline factories,
the framework-neutral workflow/segmentation/inpainting services built from
those factories, the per-project :class:`ProjectRegistry` (locks, revisions,
active jobs, log ring buffers) and the background :class:`~parallax_maker.
api.jobs.JobManager`.

:func:`create_runtime` builds the production ``Runtime``; the deterministic
browser-test double lives in :func:`parallax_maker.e2e_support.fakes.
create_fake_runtime`, which calls :func:`build_runtime` with fake factories so
both entry points share the exact same wiring code.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque

from PIL import Image

from .api.jobs import Job, JobManager
from .depth import DepthEstimationModel
from .export_services import ExportService
from .inpainting import InpaintingModel
from .inpainting_services import InpaintingService
from .instance import SegmentationModel
from .project_services import ProjectService
from .segmentation_services import SegmentationService
from .slice_editing_services import SliceEditingService
from .upscaler import Upscaler
from .workflow_services import WorkflowService

#: Number of log entries retained per project before older ones are dropped.
PROJECT_LOG_CAPACITY = 200


@dataclass(frozen=True)
class LogEntry:
    """One entry in a project's log ring buffer."""

    seq: int
    level: str
    message: str


class ProjectLog:
    """A small, thread-safe, monotonically-numbered ring buffer of log lines.

    Replaces the Dash log pane (``C.LOGS_DATA``) for the API: each project
    keeps its own bounded history so ``GET /projects/{id}/logs?after={seq}``
    can page through recent activity without growing without bound.
    """

    def __init__(self, maxlen: int = PROJECT_LOG_CAPACITY) -> None:
        self._entries: Deque[LogEntry] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()

    def append(self, message: str, level: str = "info") -> LogEntry:
        """Record ``message`` and return the entry that was appended."""

        with self._lock:
            self._seq += 1
            entry = LogEntry(seq=self._seq, level=level, message=message)
            self._entries.append(entry)
            return entry

    def after(self, seq: int) -> list[LogEntry]:
        """Return entries with ``seq`` strictly greater than ``seq``."""

        with self._lock:
            return [entry for entry in self._entries if entry.seq > seq]

    def latest_seq(self) -> int:
        with self._lock:
            return self._seq


@dataclass(frozen=True)
class InpaintingSettings:
    """Model/parameter settings for inpainting generation (in-memory only).

    Mirrors the Dash sliders'/dropdown's live values (components.py's
    ``DROPDOWN_INPAINT_MODEL``/``SLIDER_INPAINT_STRENGTH``/
    ``SLIDER_INPAINT_GUIDANCE``/``SLIDER_MASK_PADDING``/``SLIDER_MASK_BLUR``/
    ``INPUT_EXTERNAL_SERVER`` state). Dash never persists most of these beyond
    the browser widget's own state; only ``model`` (via
    ``InpaintingService.update_model``) and ``external_server``/``api_key``
    (written straight onto ``AppState`` by Dash's own ``reset_external_*``
    callbacks) are ever saved to the project JSON, which ``api/inpainting.py``
    mirrors. ``api_key`` is deliberately never echoed back by the API.
    """

    model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    strength: float = 0.8
    guidance_scale: float = 7.5
    padding: int = 50
    blur: int = 50
    external_server: str = "localhost:7860"
    api_key: str = ""


@dataclass(frozen=True)
class InpaintingCandidateSet:
    """One successful ``generate_candidates`` result, kept server-side.

    Bound to the slice index/version it was generated from so
    ``POST .../inpainting/apply`` can reject a stale set (see the
    "Candidates" section of ``docs/svelte-migration/ARCHITECTURE.md``).
    """

    generation_id: str
    slice_index: int
    slice_version: int
    images: tuple[Image.Image, ...]


class ProjectRecord:
    """Per-project concurrency and observability state.

    ``lock`` guards the ``AppState`` mutation itself and is always acquired
    and released from a single call frame in one thread: either the request
    thread for a synchronous mutation (slice-count/threshold updates), or the
    ``JobManager`` worker thread for the duration of a job's ``run`` callback.
    It is therefore always safe to use a plain :class:`threading.RLock`.

    Whether the project is "busy" is tracked separately via
    ``try_begin_job``/``end_job``, guarded by a short-lived internal mutex, so
    a request thread can atomically test-and-set the active job before handing
    the real work to the worker thread without racing it.
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.lock = threading.RLock()
        self.log = ProjectLog()
        self._meta_lock = threading.Lock()
        self._revision = 1
        self._active_job_id: str | None = None

        #: The image currently served at the ``main`` asset id, or ``None`` to
        #: mean "show the input image" (matches ``AppState.serve_main_image``
        #: vs. ``serve_input_image`` in Dash). In-memory only: never persisted
        #: to the project JSON, mirroring Dash's own transient main-image src.
        self.display_image: Image.Image | None = None
        #: Content versions used in asset URLs, so a mutation only reloads the
        #: images it actually changed (the project revision changes on every
        #: mutation, including ones that leave every image untouched).
        self.display_version = 0
        self.input_version = 0
        self._main_asset_lock = threading.Lock()
        #: ``((input_version, display_version), encoded_png_bytes)``.
        self._main_asset_cache: tuple[tuple[int, int], bytes] | None = None

        #: Inpainting model/parameter settings, the last successful candidate
        #: generation (if any), and any uploaded ComfyUI workflow bytes; see
        #: InpaintingSettings/InpaintingCandidateSet. In-memory only, guarded
        #: separately from the main-asset fields above since api/inpainting.py
        #: reads/writes them independently from request threads.
        self._inpainting_lock = threading.Lock()
        self.inpainting_settings = InpaintingSettings()
        self.inpainting_candidates: InpaintingCandidateSet | None = None
        self.inpainting_workflow: bytes | None = None

    def set_display_image(self, image: Image.Image | None) -> None:
        """Set the in-memory display image (``None`` means "show the input")."""

        with self._main_asset_lock:
            if image is None and self.display_image is None:
                return  # already showing the input; keep the URL stable
            self.display_image = image
            self.display_version += 1
            self._main_asset_cache = None

    def bump_input_version(self) -> None:
        """Record that ``AppState.imgData`` was replaced (upload/restore)."""

        with self._main_asset_lock:
            self.input_version += 1
            self._main_asset_cache = None

    def main_asset_key(self) -> tuple[tuple[int, int], Image.Image | None]:
        """The main asset's content key and display image, read atomically."""

        with self._main_asset_lock:
            return (self.input_version, self.display_version), self.display_image

    def main_asset_cache(self) -> tuple[tuple[int, int], bytes] | None:
        with self._main_asset_lock:
            return self._main_asset_cache

    def cache_main_asset(self, key: tuple[int, int], data: bytes) -> None:
        with self._main_asset_lock:
            if key == (self.input_version, self.display_version):
                self._main_asset_cache = (key, data)

    @property
    def revision(self) -> int:
        with self._meta_lock:
            return self._revision

    def bump_revision(self) -> int:
        with self._meta_lock:
            self._revision += 1
            return self._revision

    @property
    def active_job_id(self) -> str | None:
        with self._meta_lock:
            return self._active_job_id

    def try_begin_job(self, job_id: str) -> bool:
        """Atomically claim the busy slot for ``job_id``.

        Returns False (without side effects) if another job is already
        active, which callers turn into a ``409 busy`` response.
        """

        with self._meta_lock:
            if self._active_job_id is not None:
                return False
            self._active_job_id = job_id
            return True

    def end_job(self) -> None:
        with self._meta_lock:
            self._active_job_id = None

    def get_inpainting_settings(self) -> InpaintingSettings:
        with self._inpainting_lock:
            return self.inpainting_settings

    def update_inpainting_settings(self, **fields: object) -> InpaintingSettings:
        """Merge ``fields`` onto the current settings and return the result."""

        with self._inpainting_lock:
            self.inpainting_settings = replace(self.inpainting_settings, **fields)
            return self.inpainting_settings

    def get_inpainting_candidates(self) -> InpaintingCandidateSet | None:
        with self._inpainting_lock:
            return self.inpainting_candidates

    def set_inpainting_candidates(
        self, candidates: InpaintingCandidateSet | None
    ) -> None:
        with self._inpainting_lock:
            self.inpainting_candidates = candidates

    def get_inpainting_workflow(self) -> bytes | None:
        with self._inpainting_lock:
            return self.inpainting_workflow

    def set_inpainting_workflow(self, workflow: bytes | None) -> None:
        with self._inpainting_lock:
            self.inpainting_workflow = workflow


class ProjectRegistry:
    """Track per-project locks, revisions, active jobs and logs.

    Works with the existing ``AppState.cache``/``CachedAppStateRepository``:
    the project id is the ``appstate-*`` directory name already used as the
    cache key, so this registry needs no filesystem knowledge of its own.
    """

    def __init__(self) -> None:
        self._records: dict[str, ProjectRecord] = {}
        self._registry_lock = threading.Lock()

    def ensure(self, project_id: str) -> ProjectRecord:
        """Return the record for ``project_id``, creating it at revision 1."""

        with self._registry_lock:
            record = self._records.get(project_id)
            if record is None:
                record = ProjectRecord(project_id)
                self._records[project_id] = record
            return record

    def get(self, project_id: str) -> ProjectRecord | None:
        with self._registry_lock:
            return self._records.get(project_id)

    def bump_revision(self, project_id: str) -> int:
        return self.ensure(project_id).bump_revision()

    def revision(self, project_id: str) -> int:
        return self.ensure(project_id).revision

    def forget(self, project_id: str) -> None:
        """Drop bookkeeping for a project; used by tests to avoid leaking state."""

        with self._registry_lock:
            self._records.pop(project_id, None)


class ProgressReporter:
    """Forward depth-generation progress callbacks to the currently active job.

    ``WorkflowService`` is constructed once with a single progress callback.
    ``JobManager`` runs at most one job at a time on its single worker thread,
    so a simple "currently bound job" slot is enough to route progress without
    per-call synchronization beyond what the worker thread already provides.
    """

    def __init__(self) -> None:
        self._current: Job | None = None

    def bind(self, job: Job) -> None:
        self._current = job

    def clear(self) -> None:
        self._current = None

    def __call__(self, current: int, total: int) -> None:
        job = self._current
        if job is not None and total > 0:
            job.set_progress(current / total)


@dataclass
class Runtime:
    """Everything the API blueprint (and, later, other transports) needs.

    ``*_factory`` callables resolve model/pipeline classes the same way the
    frozen Dash callbacks do (``webui.DepthEstimationModel`` and friends): a
    plain callable so tests can substitute deterministic fakes.
    """

    depth_model_factory: Callable[..., object]
    segmentation_model_factory: Callable[..., object]
    inpainting_model_factory: Callable[..., object]
    upscaler_factory: Callable[..., object]
    workflow_service: WorkflowService
    segmentation_service: SegmentationService
    inpainting_service: InpaintingService
    slice_editing_service: SliceEditingService = field(
        default_factory=SliceEditingService
    )
    project_service: ProjectService = field(default_factory=ProjectService)
    export_service: ExportService = field(default_factory=ExportService)
    projects: ProjectRegistry = field(default_factory=ProjectRegistry)
    jobs: JobManager = field(default_factory=JobManager)
    progress_reporter: ProgressReporter = field(default_factory=ProgressReporter)


#: Mask-generation expansion used by slice generation; mirrors webui.EXPAND_MASK.
DEFAULT_SLICE_EXPAND = 5


def build_runtime(
    *,
    depth_model_factory: Callable[..., object],
    segmentation_model_factory: Callable[..., object],
    inpainting_model_factory: Callable[..., object],
    upscaler_factory: Callable[..., object],
) -> Runtime:
    """Build a ``Runtime`` from a set of model/pipeline factories.

    Shared by :func:`create_runtime` (production classes) and
    :func:`parallax_maker.e2e_support.fakes.create_fake_runtime` (fakes), so
    the wiring of services to factories only needs to be correct once.
    """

    progress_reporter = ProgressReporter()
    workflow_service = WorkflowService(
        depth_model_factory=depth_model_factory,
        progress_reporter=progress_reporter,
        slice_expand=DEFAULT_SLICE_EXPAND,
    )
    segmentation_service = SegmentationService(model_factory=segmentation_model_factory)
    inpainting_service = InpaintingService(pipeline_factory=inpainting_model_factory)
    # export_gltf's per-slice depth-map regeneration must use the same
    # (possibly fake) depth model factory as depth generation/thresholds, so
    # e2e_support.fakes.create_fake_runtime() stays fully deterministic.
    export_service = ExportService(depth_model_factory=depth_model_factory)

    return Runtime(
        depth_model_factory=depth_model_factory,
        segmentation_model_factory=segmentation_model_factory,
        inpainting_model_factory=inpainting_model_factory,
        upscaler_factory=upscaler_factory,
        workflow_service=workflow_service,
        segmentation_service=segmentation_service,
        inpainting_service=inpainting_service,
        export_service=export_service,
        progress_reporter=progress_reporter,
    )


def create_runtime() -> Runtime:
    """Build the production ``Runtime`` (real models, real filesystem state)."""

    return build_runtime(
        depth_model_factory=DepthEstimationModel,
        segmentation_model_factory=SegmentationModel,
        inpainting_model_factory=InpaintingModel,
        upscaler_factory=Upscaler,
    )
