"""CLI-behavior tests for ``parallax-gltf-cli`` (parallax_maker/gltf_cli.py).

Characterizes the console-script entry point end to end: it loads an
``AppState`` saved with ``AppState.to_file`` from ``-i`` (a project
directory, despite the flag's name), exports a glTF scene into ``-o`` via
``ExportService`` (not the removed ``webui.export_state_as_gltf``), and
supports the ``--scale``/``--no-inline``/``--depth`` flags exactly as
before the Dash removal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from . import gltf_cli
from .controller import AppState
from .slice import ImageSlice


def make_project(name: str = "appstate-gltf-cli") -> Path:
    """Build and persist a minimal two-slice project, as ``AppState.to_file`` would.

    ``AppState.check_pathnames`` (run by ``from_json``/``from_file``) requires
    ``filename`` to resolve to ``<cwd>/appstate-*``, so the caller must
    ``monkeypatch.chdir`` into a temp directory first and pass a bare
    ``appstate-*`` name, exactly like the real save/restore workflow.
    """

    state = AppState()
    state.imgData = Image.new("RGB", (20, 10), (100, 110, 120))
    state.image_slices = [
        ImageSlice(
            np.dstack(
                [
                    np.full((10, 20, 3), (10, 20, 30), dtype=np.uint8),
                    np.full((10, 20), 255, dtype=np.uint8),
                ]
            ),
            depth=50,
        ),
        ImageSlice(
            np.dstack(
                [
                    np.full((10, 20, 3), (40, 50, 60), dtype=np.uint8),
                    np.full((10, 20), 255, dtype=np.uint8),
                ]
            ),
            depth=150,
        ),
    ]
    project_dir = Path(name)
    state.filename = str(project_dir)
    state.to_file(project_dir)
    return project_dir


def test_main_exports_gltf_without_displacement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project_dir = make_project()
    output_dir = tmp_path / "out"

    argv = [
        "parallax-gltf-cli",
        "-i",
        str(project_dir),
        "-o",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    gltf_cli.main()

    gltf_path = output_dir / AppState.MODEL_FILE
    assert gltf_path.exists()
    scene = json.loads(gltf_path.read_text())
    assert scene["asset"]["version"] == "2.0"
    assert len(scene["meshes"]) == 2
    # No displacement requested (default --scale 0.0): flat, un-subdivided quads.
    assert scene["accessors"][1]["count"] == 4
    # Inline images are the default (no --no-inline).
    assert all(image["uri"].startswith("data:image/png;base64,") for image in scene["images"])


def test_main_no_inline_writes_external_textures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project_dir = make_project("appstate-gltf-cli-noinline")
    output_dir = tmp_path / "out-noinline"

    argv = [
        "parallax-gltf-cli",
        "-i",
        str(project_dir),
        "-o",
        str(output_dir),
        "--no-inline",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    gltf_cli.main()

    scene = json.loads((output_dir / AppState.MODEL_FILE).read_text())
    assert not any(image["uri"].startswith("data:") for image in scene["images"])


def test_main_with_scale_subdivides_mesh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project_dir = make_project("appstate-gltf-cli-scale")
    output_dir = tmp_path / "out-scale"

    # Pre-populate each slice's depth map so ExportService.export_gltf's
    # "regenerate a missing depth map" branch (which would otherwise load a
    # real DepthEstimationModel) is skipped, keeping this a fast, offline
    # CLI-behavior test rather than a model-loading test.
    for index in range(2):
        depth_path = project_dir / f"image_slice_{index}_depth.png"
        Image.fromarray(np.linspace(0, 255, 10 * 20, dtype=np.uint8).reshape(10, 20)).save(
            depth_path
        )

    argv = [
        "parallax-gltf-cli",
        "-i",
        str(project_dir),
        "-o",
        str(output_dir),
        "--scale",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    gltf_cli.main()

    scene = json.loads((output_dir / AppState.MODEL_FILE).read_text())
    # A positive displacement scale with an on-disk depth map for every slice
    # subdivides each card instead of leaving a flat 4-vertex quad.
    assert scene["accessors"][1]["count"] > 4


def test_main_requires_state_file_argument(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["parallax-gltf-cli"])
    with pytest.raises((SystemExit, FileNotFoundError, TypeError)):
        gltf_cli.main()
