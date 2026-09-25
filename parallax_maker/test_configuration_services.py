"""Contract tests for the framework-neutral configuration-probe service."""

from __future__ import annotations

from .configuration_services import probe_server, validate_api_key


# --- probe_server ----------------------------------------------------------------


def test_probe_server_automatic1111_success():
    result = probe_server(
        "automatic1111",
        "localhost:7860",
        automatic1111_probe=lambda addr: ["model-a", "model-b"],
    )

    assert result.ok is True
    assert "successful" in result.message
    assert "['model-a', 'model-b']" in result.message


def test_probe_server_comfyui_success():
    result = probe_server(
        "comfyui", "localhost:8188", comfyui_probe=lambda addr: {"e2e": "ready"}
    )

    assert result.ok is True
    assert "comfyui" in result.message


def test_probe_server_reports_none_response_as_failure():
    result = probe_server(
        "automatic1111", "localhost:7860", automatic1111_probe=lambda addr: None
    )

    assert result.ok is False
    assert result.message == "Connection to automatic1111 failed"


def test_probe_server_catches_probe_exceptions():
    def boom(addr):
        raise RuntimeError("connection refused")

    result = probe_server("automatic1111", "localhost:7860", automatic1111_probe=boom)

    assert result.ok is False
    assert "connection refused" in result.message


def test_probe_server_rejects_unknown_model():
    result = probe_server("unknown-model", "localhost:7860")

    assert result.ok is False
    assert "Unknown model type" in result.message


def test_probe_server_does_not_call_the_wrong_probe():
    calls = []
    result = probe_server(
        "comfyui",
        "localhost:8188",
        automatic1111_probe=lambda addr: calls.append("automatic1111") or ["x"],
        comfyui_probe=lambda addr: calls.append("comfyui") or {"ok": True},
    )

    assert result.ok is True
    assert calls == ["comfyui"]


# --- validate_api_key --------------------------------------------------------------


def test_validate_api_key_stabilityai_success():
    result = validate_api_key(
        "stabilityai",
        "sk-1234567890abcdef",
        stabilityai_validator=lambda key: (True, 42.5),
    )

    assert result.ok is True
    assert "42.50" in result.message


def test_validate_api_key_stabilityai_rejects_bad_format_without_probing():
    calls = []
    result = validate_api_key(
        "stabilityai",
        "not-a-key",
        stabilityai_validator=lambda key: calls.append(key) or (True, 1.0),
    )

    assert result.ok is False
    assert result.message == "Invalid StabilityAI API key format"
    assert calls == []


def test_validate_api_key_stabilityai_provider_failure():
    result = validate_api_key(
        "stabilityai",
        "sk-1234567890abcdef",
        stabilityai_validator=lambda key: (False, None),
    )

    assert result.ok is False
    assert result.message == "Connection to stabilityai failed"


def test_validate_api_key_falai_success():
    result = validate_api_key(
        "falai-sdxl", "a-long-enough-key", falai_validator=lambda key: (True, None)
    )

    assert result.ok is True
    assert result.message == "Connection to falai-sdxl successful"


def test_validate_api_key_falai_rejects_short_key_without_probing():
    calls = []
    result = validate_api_key(
        "falai-sdxl", "short", falai_validator=lambda key: calls.append(key) or (True, None)
    )

    assert result.ok is False
    assert result.message == "Invalid fal.ai API key format"
    assert calls == []


def test_validate_api_key_falai_provider_failure_carries_the_error():
    result = validate_api_key(
        "falai-sdxl",
        "a-long-enough-key",
        falai_validator=lambda key: (False, "quota exceeded"),
    )

    assert result.ok is False
    assert "quota exceeded" in result.message


def test_validate_api_key_catches_validator_exceptions():
    def boom(key):
        raise RuntimeError("network unreachable")

    result = validate_api_key(
        "stabilityai", "sk-1234567890abcdef", stabilityai_validator=boom
    )

    assert result.ok is False
    assert "network unreachable" in result.message


def test_validate_api_key_rejects_unknown_model():
    result = validate_api_key("unknown-model", "any-key")

    assert result.ok is False
    assert "Unknown model type" in result.message


def test_validate_api_key_never_echoes_the_key_in_the_message():
    secret = "sk-super-secret-value-xyz"
    result = validate_api_key(
        "stabilityai", secret, stabilityai_validator=lambda key: (True, 1.0)
    )

    assert secret not in result.message
