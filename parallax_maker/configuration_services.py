"""Framework-neutral configuration probe services.

Mirrors ``components.py``'s ``test_external_connection``/``test_api_key``
callbacks (see ``PARITY.md`` "Configuration") without any Dash/Flask
dependency: given a model name and a server address/API key, probe the
provider and report success/failure with the exact log message Dash produces,
instead of raising for a failed probe. ``api/configuration.py`` maps
``ProbeResult`` straight onto ``{"ok": ..., "message": ...}`` - a failed probe
is a normal ``200`` response, never a ``5xx``, and the tested credential is
never echoed back (only the resulting message, which - like Dash's own log
line - never contains the key itself).

The default probe/validator callables resolve their target module lazily (by
importing the module object, not the function, and reading the attribute at
call time) so that ``e2e_support.fakes.install_fakes()`` - which patches
``automatic1111.make_models_request``, ``comfyui.get_history``,
``stabilityai.StabilityAI`` and ``falai.FalAI`` as module attributes - is
honored even though those patches are applied after this module is imported.
Every probe/validator is also independently injectable for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import automatic1111 as _automatic1111
from . import comfyui as _comfyui
from . import falai as _falai
from . import stabilityai as _stabilityai

#: (server_address) -> truthy data on success; raises on failure.
ServerProbe = Callable[[str], object]

#: (api_key) -> (success, credits_or_error).
KeyValidator = Callable[[str], tuple[bool, object]]


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of a connection/key probe; never raises, never carries a
    credential - only ``message``, which mirrors Dash's own log line."""

    ok: bool
    message: str


def _default_automatic1111_probe(server_address: str) -> object:
    return _automatic1111.make_models_request(server_address)


def _default_comfyui_probe(server_address: str) -> object:
    return _comfyui.get_history(server_address, "test")


def _default_stabilityai_validator(api_key: str) -> tuple[bool, object]:
    return _stabilityai.StabilityAI(api_key).validate_key()


def _default_falai_validator(api_key: str) -> tuple[bool, object]:
    return _falai.FalAI(api_key).validate_key()


def probe_server(
    model: str,
    server_address: str,
    *,
    automatic1111_probe: ServerProbe = _default_automatic1111_probe,
    comfyui_probe: ServerProbe = _default_comfyui_probe,
) -> ProbeResult:
    """Probe an Automatic1111/ComfyUI server; mirrors ``test_external_connection``."""

    try:
        if model == "automatic1111":
            data = automatic1111_probe(server_address)
        elif model == "comfyui":
            data = comfyui_probe(server_address)
        else:
            return ProbeResult(False, f"Unknown model type: {model}")
    except Exception as exc:  # noqa: BLE001 - mirrors Dash's own broad except
        return ProbeResult(False, f"Connection to {model} failed: {exc}")

    if data is not None:
        return ProbeResult(True, f"Connection to {model} successful: {data}")
    return ProbeResult(False, f"Connection to {model} failed")


def validate_api_key(
    model: str,
    api_key: str,
    *,
    stabilityai_validator: KeyValidator = _default_stabilityai_validator,
    falai_validator: KeyValidator = _default_falai_validator,
) -> ProbeResult:
    """Validate a StabilityAI/fal.ai API key; mirrors ``test_api_key``."""

    if model == "stabilityai":
        if not (api_key.startswith("sk-") and len(api_key) > 12):
            return ProbeResult(False, "Invalid StabilityAI API key format")
        try:
            success, credits = stabilityai_validator(api_key)
        except Exception as exc:  # noqa: BLE001 - mirrors Dash's own broad except
            return ProbeResult(False, f"Connection to {model} failed: {exc}")
        if success:
            return ProbeResult(
                True,
                f"Connection to {model} successful: you have "
                f"{credits:0.2f} remaining credits",
            )
        return ProbeResult(False, f"Connection to {model} failed")

    if model.startswith("falai-"):
        if not len(api_key) > 10:
            return ProbeResult(False, "Invalid fal.ai API key format")
        try:
            success, error = falai_validator(api_key)
        except Exception as exc:  # noqa: BLE001 - mirrors Dash's own broad except
            return ProbeResult(False, f"Connection to {model} failed: {exc}")
        if success:
            return ProbeResult(True, f"Connection to {model} successful")
        return ProbeResult(False, f"Connection to {model} failed: {error}")

    return ProbeResult(False, f"Unknown model type: {model}")
