"""Single-worker background job execution for mutating project operations.

Depth generation and slice generation can take a while, so the HTTP layer
kicks them off with ``202 {job}`` and lets the client poll ``GET /jobs/{id}``
until the job reaches a terminal status. Running everything through one
worker thread keeps the model/pipeline caches on ``AppState`` (and the fake
provider globals patched by ``install_fakes``) free of cross-thread races
without needing per-model locking.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    """A tracked unit of work; mutated in place as it progresses."""

    id: str
    kind: str
    project_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    error: str | None = None
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def set_progress(self, value: float) -> None:
        with self._lock:
            self.progress = max(0.0, min(1.0, value))


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
    ) -> Job:
        """Queue ``run`` for execution and return its (queued) ``Job`` record.

        ``job_id`` lets a caller reserve the id up front (e.g. to atomically
        claim a project's busy slot before the worker thread picks the job
        up); a fresh id is generated when omitted.
        """

        job = Job(id=job_id or uuid4().hex, kind=kind, project_id=project_id)
        with self._jobs_lock:
            self._jobs[job.id] = job
        self._queue.put((job, run))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def _run_forever(self) -> None:
        while True:
            job, run = self._queue.get()
            job.status = JobStatus.RUNNING
            try:
                run(job)
                job.status = JobStatus.SUCCEEDED
                job.progress = 1.0
            except Exception as exc:  # noqa: BLE001 - convert to a sanitized job state
                logger.exception(
                    "job %s (%s) for project %s failed",
                    job.id,
                    job.kind,
                    job.project_id,
                )
                job.status = JobStatus.FAILED
                job.error = _sanitize_error(exc)
            finally:
                self._queue.task_done()


def _sanitize_error(exc: Exception) -> str:
    """A short, client-safe message; full tracebacks are logged server-side only."""

    message = f"{type(exc).__name__}: {exc}"
    return message[:500]
