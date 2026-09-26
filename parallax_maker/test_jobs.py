"""Unit tests for the ``Job``/``JobManager`` state machine, independent of
the Flask blueprint (see ``test_api_jobs.py`` for HTTP-level coverage,
including the ``DELETE /jobs/{id}`` route).
"""

from __future__ import annotations

import threading
import time

import pytest

from .api.jobs import Job, JobManager, JobStatus
from .cancellation import OperationCancelled


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "condition was not met within the timeout"


def test_request_cancel_accepts_any_queued_job() -> None:
    job = Job(id="j1", kind="depth", project_id="p1", cancellable=False)
    assert job.status == JobStatus.QUEUED
    assert job.request_cancel() is True


def test_request_cancel_accepts_running_cancellable_job() -> None:
    job = Job(id="j1", kind="inpainting", project_id="p1", cancellable=True)
    job.mark_running()
    assert job.request_cancel() is True
    with pytest.raises(OperationCancelled):
        job.raise_if_cancelled()


def test_request_cancel_rejects_running_non_cancellable_job() -> None:
    job = Job(id="j1", kind="depth", project_id="p1", cancellable=False)
    job.mark_running()
    assert job.request_cancel() is False
    job.raise_if_cancelled()  # never armed -> does not raise


@pytest.mark.parametrize("terminal", ["mark_succeeded", "mark_cancelled"])
def test_request_cancel_rejects_a_job_that_already_succeeded_or_was_cancelled(
    terminal: str,
) -> None:
    job = Job(id="j1", kind="inpainting", project_id="p1", cancellable=True)
    job.mark_running()
    getattr(job, terminal)()
    assert job.request_cancel() is False


def test_request_cancel_rejects_a_failed_job() -> None:
    job = Job(id="j1", kind="depth", project_id="p1", cancellable=True)
    job.mark_running()
    job.mark_failed("boom")
    assert job.request_cancel() is False


def test_set_progress_raises_for_a_cancellable_job_once_cancel_is_requested() -> None:
    job = Job(id="j1", kind="inpainting", project_id="p1", cancellable=True)
    job.mark_running()
    job.set_progress(0.1)  # no cancellation requested yet -> no raise
    assert job.progress == 0.1

    job.request_cancel()
    with pytest.raises(OperationCancelled):
        job.set_progress(0.2)


def test_set_progress_never_raises_for_a_non_cancellable_job() -> None:
    job = Job(id="j1", kind="depth", project_id="p1", cancellable=False)
    job.mark_running()
    # A running non-cancellable job rejects the request outright, so there
    # is nothing for a later set_progress to raise on.
    assert job.request_cancel() is False
    job.set_progress(0.5)
    assert job.progress == 0.5


def test_snapshot_cancellable_reflects_queued_running_and_cancel_requested_states() -> (
    None
):
    job = Job(id="j1", kind="inpainting", project_id="p1", cancellable=True)
    job.set_detail("loading")

    snapshot = job.snapshot()
    assert snapshot.status == JobStatus.QUEUED
    assert snapshot.detail == "loading"
    assert snapshot.cancellable is True
    assert snapshot.cancel_requested is False

    job.mark_running()
    snapshot = job.snapshot()
    assert snapshot.status == JobStatus.RUNNING
    assert snapshot.cancellable is True

    job.request_cancel()
    snapshot = job.snapshot()
    assert snapshot.cancel_requested is True
    # Already cancel-requested: DELETE would no longer accept a fresh request.
    assert snapshot.cancellable is False


def test_snapshot_cancellable_is_false_for_a_running_non_cancellable_job() -> None:
    job = Job(id="j1", kind="depth", project_id="p1", cancellable=False)
    job.mark_running()
    assert job.snapshot().cancellable is False


def test_mark_terminal_transitions_clear_detail() -> None:
    succeeded = Job(id="j1", kind="depth", project_id="p1")
    succeeded.set_detail("loading")
    succeeded.mark_running()
    succeeded.mark_succeeded()
    assert succeeded.detail is None

    failed = Job(id="j2", kind="depth", project_id="p1")
    failed.set_detail("loading")
    failed.mark_running()
    failed.mark_failed("boom")
    assert failed.detail is None

    cancelled = Job(id="j3", kind="inpainting", project_id="p1", cancellable=True)
    cancelled.set_detail("loading")
    cancelled.mark_running()
    cancelled.mark_cancelled()
    assert cancelled.detail is None


def test_job_manager_marks_the_job_cancelled_when_run_raises_operation_cancelled() -> (
    None
):
    manager = JobManager()

    def run(job: Job) -> None:
        raise OperationCancelled("cancel requested")

    job = manager.submit("depth", "p1", run)

    _wait_until(lambda: job.status != JobStatus.QUEUED)
    _wait_until(lambda: job.status != JobStatus.RUNNING)
    assert job.status == JobStatus.CANCELLED


def test_job_manager_cancel_requests_cancellation_of_a_running_job() -> None:
    manager = JobManager()
    release = threading.Event()

    def run(job: Job) -> None:
        release.wait(timeout=5)
        job.raise_if_cancelled()

    job = manager.submit("inpainting", "p1", run, cancellable=True)
    _wait_until(lambda: job.status == JobStatus.RUNNING)

    cancelled_job = manager.cancel(job.id)
    assert cancelled_job is job
    assert job.snapshot().cancel_requested is True

    release.set()
    _wait_until(lambda: job.status != JobStatus.RUNNING)
    assert job.status == JobStatus.CANCELLED


def test_job_manager_cancel_returns_none_for_an_unknown_job() -> None:
    manager = JobManager()
    assert manager.cancel("does-not-exist") is None


def test_request_cancel_rejects_a_repeated_request() -> None:
    job = Job(id="j", kind="inpainting", project_id="p")
    assert job.request_cancel() is True
    assert job.request_cancel() is False
    assert job.snapshot().cancellable is False
