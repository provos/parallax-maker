"""Ground-plane fitting route (see ``ground_services.fit_ground``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint

from ..ground_services import GroundNotReady, fit_ground
from ..project_services import UpdateSettings
from .errors import error_response
from .projects import (
    _build_project_view,
    _load_state,
    _mutation_guard,
    _mutation_response,
)

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def register_ground_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    @blueprint.post("/projects/<project_id>/ground/fit")
    def fit_ground_route(project_id: str):
        """Puts the horizon on the ground mask's top edge and lets the ground
        meet the frame's bottom at the nearest card touching it."""
        state = _load_state(project_id)
        record = runtime.projects.ensure(project_id)
        try:
            fit = fit_ground(state)
        except GroundNotReady as error:
            return error_response(409, "not_ready", str(error))

        with _mutation_guard(record):
            result = runtime.project_service.update_settings(
                UpdateSettings(
                    state_id=project_id, pitch=fit.pitch, ground_near=fit.ground_near
                )
            )
            record.log.append(
                f"Fitted the ground plane: horizon at row {fit.horizon_row:.0f}, "
                f"ground from depth {fit.ground_near:.0f}"
            )
            record.bump_revision()

        view = _build_project_view(runtime, project_id, state)
        return _mutation_response(view, result.changed)
