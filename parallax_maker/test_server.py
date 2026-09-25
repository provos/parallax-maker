from pathlib import Path

import pytest
from flask import Flask

from .server import _register_static_app


@pytest.fixture
def client(tmp_path: Path):
    (tmp_path / "index.html").write_text("<!doctype html><title>app</title>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-abc.js").write_text("console.log(1)")
    app = Flask(__name__)
    _register_static_app(app, tmp_path)
    return app.test_client()


def test_root_serves_index_without_caching(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<title>app</title>" in response.data
    assert response.headers["Cache-Control"] == "no-cache"


def test_hashed_asset_is_served_with_long_caching(client):
    response = client.get("/assets/index-abc.js")
    assert response.status_code == 200
    assert "immutable" in response.headers["Cache-Control"]


def test_unknown_route_falls_back_to_index_but_missing_asset_is_404(client):
    assert b"<title>app</title>" in client.get("/some/route").data
    assert client.get("/assets/missing.js").status_code == 404


def test_path_traversal_is_not_served(client):
    response = client.get("/..%2Fserver.py")
    assert b"import" not in response.data


@pytest.mark.parametrize(
    "path, location",
    [
        ("/next/", "/"),
        ("/next/assets/index-abc.js", "/assets/index-abc.js"),
        ("/next/%5C%5Cevil.com", "/evil.com"),
    ],
)
def test_legacy_next_redirects_stay_on_site(client, path, location):
    response = client.get(path)
    assert response.status_code == 308
    assert response.headers["Location"] == location


def test_legacy_next_double_slash_never_redirects_off_site(client):
    response = client.get("/next//evil.com")
    location = response.headers.get("Location", "")
    assert not location.startswith("//") and "evil.com" not in location.split("/")[:3]


def test_not_built_is_explained(tmp_path: Path):
    app = Flask(__name__)
    _register_static_app(app, tmp_path)
    response = app.test_client().get("/")
    assert response.status_code == 404
