from __future__ import annotations

from .api.schemas import combined_json_schema, dump_schema


def test_health_reports_ok_and_a_version(client) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert isinstance(body["version"], str) and body["version"]


def test_unknown_route_returns_json_not_found(client) -> None:
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404


def test_combined_json_schema_includes_project_view() -> None:
    schema = combined_json_schema()

    assert "ProjectView" in schema["$defs"]
    assert "SliceView" in schema["$defs"]
    assert "properties" in schema["$defs"]["ProjectView"]
    # camelCase on the wire.
    assert "numSlices" in schema["$defs"]["ProjectView"]["properties"]


def test_dump_schema_writes_a_file(tmp_path) -> None:
    out = tmp_path / "schema.json"

    dump_schema(out)

    assert out.exists()
    assert "ProjectView" in out.read_text()


def test_unknown_api_paths_return_json_404(client) -> None:
    for path in (
        "/api/v1/nope",
        "/api/v1/projects/appstate-x/assets/..%2F..%2Fpyproject.toml",
    ):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.get_json()["error"]["code"] == "not_found"
