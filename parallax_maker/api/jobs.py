"""Single-worker background job execution for mutating project operations.

Depth generation and slice generation can take a while, so the HTTP layer
kicks them off with ``202 {job}`` and lets the client poll ``GET /jobs/{id}``
until the job reaches a terminal status. Running everything through one
worker thread keeps the model/pipeline caches on ``AppState`` (and the fake
provider globals patched by ``install_fakes``) free of cross-thread races
without needing per-model locking.

Some jobs (currently only ``inpainting`` generation) are cancellable: a
client can ``DELETE /jobs/{id}`` while the job is queued, or while it is
running and opted into cancellation. A running cancellable job only notices
the request the next time it reports progress (``Job.set_progress``), since
that is the only point the worker thread checks back in with the job record;
see ``cancellation.OperationCancelled``.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from uuid import uuid4

from ..cancellation import OperationCancelled

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A tracked unit of work; mutated in place as it progresses."""

    id: str
    kind: str
    project_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    error: str | None = None
    #: Set at submit time; whether this job can be cancelled while running
    #: (a queued job can always be cancelled, regardless of this flag).
    cancellable: bool = False
    #: A short human-readable status line (e.g. "Loading the depth model
    #: (the first run downloads its weights)"); cleared on every terminal
    #: transition.
    detail: str | None = None
    _cancel_requested: bool = field(default=False, repr=False, compare=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def set_progress(self, value: float) -> None:
        with self._lock:
            self.progress = max(0.0, min(1.0, value))
            cancellable = self.cancellable
        # Checked outside the lock: raising while holding it would leave the
        # job locked if a caller ever caught the exception and retried.
        if cancellable:
            self.raise_if_cancelled()

    def set_detail(self, text: str | None) -> None:
        with self._lock:
            self.detail = text

    def request_cancel(self) -> bool:
        """Ask this job to stop; returns whether the request was accepted.

        Accepted for any ``QUEUED`` job, or a ``RUNNING`` job that opted into
        ``cancellable``. A ``RUNNING`` non-cancellable job, or any job that
        has already reached a terminal status, rejects the request.
        """

        with self._lock:
            if self.status == JobStatus.QUEUED or (
                self.status == JobStatus.RUNNING and self.cancellable
            ):
                self._cancel_requested = True
                return True
            return False

    def raise_if_cancelled(self) -> None:
        with self._lock:
            requested = self._cancel_requested
        if requested:
            raise OperationCancelled(f"job {self.id} ({self.kind}) was cancelled")

    def mark_running(self) -> None:
        with self._lock:
            self.status = JobStatus.RUNNING

    def mark_succeeded(self) -> None:
        with self._lock:
            self.status = JobStatus.SUCCEEDED
            self.progress = 1.0
            self.detail = None

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = JobStatus.FAILED
            self.error = error
            self.detail = None

    def mark_cancelled(self) -> None:
        with self._lock:
            self.status = JobStatus.CANCELLED
            self.detail = None

    def snapshot(self) -> "JobSnapshot":
        """A consistent copy of the mutable fields for request threads."""

        with self._lock:
            cancellable = not self._cancel_requested and (
                self.status == JobStatus.QUEUED
                or (self.status == JobStatus.RUNNING and self.cancellable)
            )
            return JobSnapshot(
                id=self.id,
                kind=self.kind,
                project_id=self.project_id,
                status=self.status,
                progress=self.progress,
                error=self.error,
                detail=self.detail,
                cancellable=cancellable,
                cancel_requested=self._cancel_requested,
            )


@dataclass(frozen=True)
class JobSnapshot:
    id: str
    kind: str
    project_id: str
    status: JobStatus
    progress: float
    error: str | None
    detail: str | None
    #: Whether ``DELETE /jobs/{id}`` would currently succeed for this job.
    cancellable: bool
    cancel_requested: bool


#: A job's unit of work; receives the ``Job`` so it can report progress.
RunFn = Callable[[Job], None]


class JobManager:
    """Run submitted jobs one at a time on a dedicated worker thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._queue: "queue.Queue[tuple[Job, RunFn]]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._run_forever, name="parallax-job-worker", daemon=True
        )
        self._worker.start()

    def submit(
        self,
        kind: str,
        project_id: str,
        run: RunFn,
        *,
        job_id: str | None = None,
        cancellable: bool = False,
    ) -> Job:
        """Queue ``run`` for execution and return its (queued) ``Job`` record.

        ``job_id`` lets a caller reserve the id up front (e.g. to atomically
        claim a project's busy slot before the worker thread picks the job
        up); a fresh id is generated when omitted. ``cancellable`` opts the
        job into ``DELETE /jobs/{id}`` while it is running (a queued job can
        always be cancelled).
        """

        job = Job(
            id=job_id or uuid4().hex,
            kind=kind,
            project_id=project_id,
            cancellable=cancellable,
        )
        with self._jobs_lock:
            self._jobs[job.id] = job
        self._queue.put((job, run))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Request cancellation of ``job_id``; returns the job, or ``None``
        if unknown. Callers still need ``Job.request_cancel()``'s own return
        value (or a fresh snapshot) to know whether the request was accepted.
        """

        job = self.get(job_id)
        if job is None:
            return None
        job.request_cancel()
        return job

    def _run_forever(self) -> None:
        while True:
            job, run = self._queue.get()
            job.mark_running()
            try:
                run(job)
                job.mark_succeeded()
            except OperationCancelled:
                logger.info(
                    "job %s (%s) for project %s cancelled",
                    job.id,
                    job.kind,
                    job.project_id,
                )
                job.mark_cancelled()
            except Exception as exc:  # noqa: BLE001 - convert to a sanitized job state
                logger.exception(
                    "job %s (%s) for project %s failed",
                    job.id,
                    job.kind,
                    job.project_id,
                )
                job.mark_failed(_sanitize_error(exc))
            finally:
                self._queue.task_done()


def _sanitize_error(exc: Exception) -> str:
    """A short, client-safe message; full tracebacks are logged server-side only."""

    message = f"{type(exc).__name__}: {exc}"
    return message[:500]
