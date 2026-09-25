"""Domain exception -> HTTP response mapping for the API blueprint.

All JSON error bodies have the shape ``{"error": {"code": str, "message":
str}}``. ``WorkflowUnchanged`` is deliberately *not* wired up here: it is not
an error, and routes that can raise it catch it explicitly so they can return
``200`` with the current ``ProjectView`` and ``"changed": false``.
"""

from __future__ import annotations

import logging

from flask import jsonify

from ..export_services import ExportServiceError, InvalidSliceIndex as InvalidExportIndex
from ..inpainting_services import (
    InpaintingModelFailed,
    InpaintingServiceError,
    InvalidInpaintingCandidate,
)
from ..project_services import ProjectServiceError
from ..segmentation_services import (
    InvalidSegmentationPoint,
    SegmentationModelFailed,
    SegmentationServiceError,
)
from ..slice_editing_services import InvalidSliceIndex, SliceEditingServiceError
from ..workflow_services import WorkflowNotReady

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Base class for errors that carry an HTTP status and a stable error code."""

    status_code = 500
    code = "internal"

    def __init__(
        self, message: str, *, status_code: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class InvalidRequest(ApiError):
    """Malformed request body or invalid parameter."""

    status_code = 400
    code = "invalid_request"


class NotFound(ApiError):
    """Unknown project/slice/job/asset."""

    status_code = 404
    code = "not_found"


class NotReady(ApiError):
    """The requested operation cannot run with the current project state."""

    status_code = 409
    code = "not_ready"


class Busy(ApiError):
    """Another mutation or job is already running for this project."""

    status_code = 409
    code = "busy"


class StaleRevision(ApiError):
    """The request's ``baseRevision`` no longer matches the project."""

    status_code = 409
    code = "stale_revision"


class ProviderError(ApiError):
    """A depth/segmentation/inpainting provider or model call failed."""

    status_code = 502
    code = "provider_error"


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def error_response(status_code: int, code: str, message: str):
    return jsonify(error_body(code, message)), status_code


def register_error_handlers(blueprint) -> None:
    """Register Flask error handlers implementing the doc's mapping table."""

    @blueprint.errorhandler(ApiError)
    def _handle_api_error(err: ApiError):
        return error_response(err.status_code, err.code, err.message)

    @blueprint.errorhandler(WorkflowNotReady)
    def _handle_workflow_not_ready(err: WorkflowNotReady):
        return error_response(409, "not_ready", str(err))

    @blueprint.errorhandler(InvalidSegmentationPoint)
    def _handle_invalid_segmentation_point(err: InvalidSegmentationPoint):
        # More specific than SegmentationServiceError below: a malformed/out-of-
        # bounds point is a client input error, not a state-readiness problem.
        return error_response(400, "invalid_request", str(err))

    @blueprint.errorhandler(SegmentationModelFailed)
    def _handle_segmentation_model_failed(err: SegmentationModelFailed):
        return error_response(502, "provider_error", str(err))

    @blueprint.errorhandler(SegmentationServiceError)
    def _handle_segmentation_error(err: SegmentationServiceError):
        # Covers SegmentationNotReady, InvalidMaskState, MultiPointModeRequired,
        # and NoPointsQueued: all "the current state can't do this" cases.
        return error_response(409, "not_ready", str(err))

    @blueprint.errorhandler(InvalidInpaintingCandidate)
    def _handle_invalid_inpainting_candidate(err: InvalidInpaintingCandidate):
        # More specific than InpaintingServiceError below: a malformed/out-of-
        # range candidate selection is a client input error, not a
        # state-readiness problem.
        return error_response(400, "invalid_request", str(err))

    @blueprint.errorhandler(InpaintingModelFailed)
    def _handle_inpainting_model_failed(err: InpaintingModelFailed):
        return error_response(502, "provider_error", str(err))

    @blueprint.errorhandler(InpaintingServiceError)
    def _handle_inpainting_error(err: InpaintingServiceError):
        # Covers InpaintingNotReady, InpaintingMaskNotFound, InpaintingUnchanged
        # (routes that can raise InpaintingUnchanged catch it explicitly to
        # return 200 changed:false instead) and SliceVersionUnavailable.
        return error_response(409, "not_ready", str(err))

    @blueprint.errorhandler(InvalidSliceIndex)
    def _handle_invalid_slice_index(err: InvalidSliceIndex):
        # More specific than SliceEditingServiceError below: an out-of-range
        # or malformed index is a client input error, not a readiness problem.
        return error_response(400, "invalid_request", str(err))

    @blueprint.errorhandler(SliceEditingServiceError)
    def _handle_slice_editing_error(err: SliceEditingServiceError):
        # Covers SliceEditingNotReady (no image/selection/mask/clipboard).
        # SliceEditingUnchanged is handled by the route itself, like
        # WorkflowUnchanged, so it never reaches here as an error.
        return error_response(409, "not_ready", str(err))

    @blueprint.errorhandler(InvalidExportIndex)
    def _handle_invalid_export_index(err: InvalidExportIndex):
        # api/export.py's own routes already convert this to InvalidRequest
        # themselves (an out-of-range slice index is validated before any
        # ExportService call), so this only guards a future direct caller.
        return error_response(400, "invalid_request", str(err))

    @blueprint.errorhandler(ExportServiceError)
    def _handle_export_error(err: ExportServiceError):
        # Covers ExportNotReady (no slices to export/upscale/render).
        return error_response(409, "not_ready", str(err))

    @blueprint.errorhandler(ProjectServiceError)
    def _handle_project_error(err: ProjectServiceError):
        # Covers InvalidProjectFile/ProjectDirectoryNotFound; api/projects.py's
        # own restore route already maps these to 400/404 itself, so this only
        # guards a future direct caller of ProjectService.
        return error_response(400, "invalid_request", str(err))

    @blueprint.errorhandler(404)
    def _handle_flask_404(err):
        return error_response(404, "not_found", "the requested resource was not found")

    @blueprint.errorhandler(405)
    def _handle_flask_405(err):
        return error_response(405, "invalid_request", "method not allowed")

    @blueprint.errorhandler(Exception)
    def _handle_unexpected(err: Exception):
        logger.exception("unhandled error while serving the API request")
        return error_response(500, "internal", "an internal error occurred")
