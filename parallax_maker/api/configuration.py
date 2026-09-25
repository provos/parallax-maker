"""Configuration probe HTTP routes: external-server connection tests and API
key validation for Automatic1111/ComfyUI/StabilityAI/fal.ai.

Delegates to ``configuration_services.probe_server``/``validate_api_key``.
Unlike every other route in this package, these two are not project-scoped -
they mirror ``components.py``'s ``test_external_connection``/``test_api_key``,
which probe a plain form value the user hasn't necessarily attached to a
project yet, not ``AppState``. Both always return ``200`` with
``{"ok": bool, "message": str}``: a failed probe is a normal response, never
a ``5xx`` (network/provider failures are caught by the service layer), and
the tested ``apiKey`` is never echoed back by either the request schema's
validation error path or the response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, jsonify, request

from ..configuration_services import probe_server, validate_api_key
from . import schemas
from .projects import _parse_json_body

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import Runtime


def register_configuration_routes(blueprint: Blueprint, runtime: "Runtime") -> None:
    del runtime  # these probes are stateless; no project/runtime state needed

    @blueprint.post("/config/probe-server")
    def probe_server_route():
        payload = _parse_json_body(request, schemas.ProbeServerRequest)
        result = probe_server(payload.model, payload.server_address)
        view = schemas.ProbeResultView(ok=result.ok, message=result.message)
        return jsonify(view.model_dump(mode="json", by_alias=True)), 200

    @blueprint.post("/config/validate-key")
    def validate_key_route():
        payload = _parse_json_body(request, schemas.ValidateKeyRequest)
        result = validate_api_key(payload.model, payload.api_key)
        view = schemas.ProbeResultView(ok=result.ok, message=result.message)
        return jsonify(view.model_dump(mode="json", by_alias=True)), 200
