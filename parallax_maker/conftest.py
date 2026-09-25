"""Shared pytest fixtures for the HTTP API test suite (``test_api_*.py``).

Kept separate from the existing per-module test helpers (e.g.
``RecordingStateRepository`` in ``test_workflow_services.py``) since these
fixtures are specific to exercising the Flask blueprint through its test
client with the fake runtime, and are useful across several API test modules.
"""

from __future__ import annotations

import pytest
from flask import Flask

from .api import create_api_blueprint
from .controller import AppState
from .e2e_support import create_fake_runtime


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run the test in an empty directory and undo any AppState.cache growth.

    ``AppState.cache`` is a class-level dict shared by the whole process, so
    without this a project created by one test could leak into another.
    """

    monkeypatch.chdir(tmp_path)
    cache_keys_before = set(AppState.cache)
    yield tmp_path
    for key in list(AppState.cache):
        if key not in cache_keys_before:
            del AppState.cache[key]


@pytest.fixture
def runtime():
    """A fresh fake Runtime; each test gets its own JobManager/ProjectRegistry."""

    return create_fake_runtime()


@pytest.fixture
def api_app(runtime):
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(runtime), url_prefix="/api/v1")
    return app


@pytest.fixture
def client(isolated_cwd, api_app):
    del isolated_cwd  # order-only dependency: chdir before the app is used
    return api_app.test_client()
