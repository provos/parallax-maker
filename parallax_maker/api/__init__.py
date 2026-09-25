"""HTTP API blueprint for the Svelte migration.

``create_api_blueprint(runtime)`` returns a ``flask.Blueprint`` implementing
the ``/api/v1`` contract from ``docs/svelte-migration/ARCHITECTURE.md``. It is
mounted onto the existing Dash ``Flask`` app by ``parallax_maker.server``, and
onto a bare Flask app directly in tests.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from flask import Blueprint, jsonify

from . import schemas
from .camera import register_camera_routes
from .configuration import register_configuration_routes
from .errors import NotFound, register_error_handlers
from .export import register_export_routes
from .inpainting import register_inpainting_routes
from .project import register_project_lifecycle_routes
from .projects import register_project_routes
from .segmentation import register_segmentation_routes
from .slice_editing import register_slice_editing_routes

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def _package_version() -> str:
    try:
        return version("parallax-maker")
    except PackageNotFoundError:
        return "0.0.0"


def create_api_blueprint(runtime: Runtime) -> Blueprint:
    """Build the ``/api/v1`` blueprint wired to ``runtime``."""

    blueprint = Blueprint("parallax_maker_api", __name__)
    register_error_handlers(blueprint)
    register_project_routes(blueprint, runtime)
    register_segmentation_routes(blueprint, runtime)
    register_slice_editing_routes(blueprint, runtime)
    register_camera_routes(blueprint, runtime)
    register_inpainting_routes(blueprint, runtime)
    register_project_lifecycle_routes(blueprint, runtime)
    register_export_routes(blueprint, runtime)
    register_configuration_routes(blueprint, runtime)

    @blueprint.get("/health")
    def health():
        payload = schemas.HealthView(ok=True, version=_package_version())
        return jsonify(payload.model_dump(mode="json", by_alias=True)), 200

    @blueprint.route(
        "/<path:unmatched>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
    )
    def unknown_endpoint(unmatched: str):
        # Keep unknown /api/v1 paths out of Dash's catch-all HTML page.
        raise NotFound(f"unknown API endpoint: {unmatched}")

    return blueprint
