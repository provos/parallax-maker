"""Contract tests for the framework-neutral export/render service."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from .controller import AppState
from .export_services import (
    ExportGltf,
    ExportNotReady,
    ExportService,
    InvalidSliceIndex,
    RenderAnimation,
    UpscaleTextures,
)
from .slice import ImageSlice


class MemoryStateRepository:
    def __init__(self, state: AppState) -> None:
        self.state = state

    def load(self, state_id: str) -> AppState:
        return self.state


class FakeDepthModel:
    """Deterministic 0..255 ramp; equality only by ``model`` name (mirrors
    the real ``DepthEstimationModel``'s own ``__eq__`` contract closely
    enough for the "reuse a matching cached model" branch to be exercised)."""

    def __init__(self, model: str = "midas") -> None:
        self.model = model

    def __eq__(self, other):
        return isinstance(other, FakeDepthModel) and self.model == other.model


def fake_generate_depth_map(image, model=None, progress_callback=None):
    height, width = image.shape[:2]
    return np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))


def make_slice(tmp_path: Path, name: str, depth: float) -> ImageSlice:
    image = np.zeros((10, 20, 4), dtype=np.uint8)
    image[:, :, :3] = (10, 20, 30)
    image[:, :, 3] = 255
    filename = tmp_path / f"{name}.png"
    Image.fromarray(image, mode="RGBA").save(filename)
    return ImageSlice(image.copy(), depth=depth, filename=str(filename))


def make_state(tmp_path: Path, name: str = "appstate-export-svc") -> AppState:
    state = AppState()
    state.filename = name
    state.imgData = Image.new("RGB", (20, 10), (100, 110, 120))
    state.image_slices = [
        make_slice(tmp_path, "image_slice_0", depth=50),
        make_slice(tmp_path, "image_slice_1", depth=150),
    ]
    return state


def make_service(state: AppState, **kwargs) -> ExportService:
    return ExportService(state_repository=MemoryStateRepository(state), **kwargs)


# --- export_gltf ---------------------------------------------------------------


def test_export_gltf_without_displacement_writes_a_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path, "appstate-gltf-flat")
    Path(state.filename).mkdir()
    service = make_service(state)

    result = service.export_gltf(
        ExportGltf(state_id=state.filename, displacement_scale=0.0)
    )

    assert result.slice_count == 2
    assert result.used_upscaled is False
    assert result.generated_depth_maps is False
    assert result.gltf_path.exists()

    scene = json.loads(result.gltf_path.read_text())
    assert scene["asset"]["version"] == "2.0"
    assert len(scene["meshes"]) == 2
    assert len(scene["images"]) == 2
    assert all(image["uri"].startswith("data:image/png;base64,") for image in scene["images"])
    # No displacement: each card is an un-subdivided flat quad (4 corners).
    # accessors[1] is the first mesh's POSITION accessor (see create_card:
    # tex-coord accessor is appended before the vertex accessor).
    assert scene["accessors"][1]["count"] == 4


def test_export_gltf_with_displacement_generates_missing_depth_maps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path, "appstate-gltf-displaced")
    Path(state.filename).mkdir()
    service = make_service(state, depth_model_factory=FakeDepthModel)

    with patch(
        "parallax_maker.export_services.generate_depth_map",
        side_effect=fake_generate_depth_map,
    ):
        result = service.export_gltf(
            ExportGltf(state_id=state.filename, displacement_scale=10.0)
        )

    assert result.generated_depth_maps is True
    for index in range(len(state.image_slices)):
        assert state.depth_filename(index).exists()

    displaced_scene = json.loads(result.gltf_path.read_text())
    # displacement_scale > 0 with a depth map present subdivides each card
    # into a (subdivisions + 1)^2 vertex grid (create_card/gltf.py); with no
    # displacement the card stays an un-subdivided flat quad (4 vertices) -
    # see the sibling "without displacement" test above for that baseline.
    # accessors[1] is the first mesh's POSITION accessor.
    assert displaced_scene["accessors"][1]["count"] > 4
    assert displaced_scene["accessors"][1]["count"] == 501 * 501


def test_export_gltf_prefers_upscaled_slice_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path, "appstate-gltf-upscaled")
    Path(state.filename).mkdir()
    upscaled = Image.new("RGBA", (40, 20), (200, 0, 0, 255))
    upscaled.save(state.upscaled_filename(0))
    service = make_service(state)

    result = service.export_gltf(
        ExportGltf(state_id=state.filename, displacement_scale=0.0)
    )

    assert result.used_upscaled is True


def test_export_gltf_requires_at_least_one_slice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path)
    state.image_slices = []
    service = make_service(state)

    with pytest.raises(ExportNotReady):
        service.export_gltf(ExportGltf(state_id=state.filename))


# --- upscale_textures ------------------------------------------------------------


class FakeUpscaler:
    def __init__(self, model_name="swin2sr", external_model=None) -> None:
        self.model_name = model_name

    def upscale_image_tiled(self, image, overlap=64, prompt="", negative_prompt=""):
        source = image if isinstance(image, Image.Image) else Image.fromarray(image)
        return source.resize((source.width * 2, source.height * 2))


class FakePipeline:
    def __init__(self, model, server_address=None, workflow_path=None, api_key=None):
        self.model = model

    def __eq__(self, other):
        return isinstance(other, FakePipeline) and self.model == other.model

    def load_model(self):
        return None


@patch("parallax_maker.controller.Upscaler", FakeUpscaler)
@patch("parallax_maker.inpainting.InpaintingModel", FakePipeline)
def test_upscale_textures_writes_an_upscaled_file_per_slice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path, "appstate-upscale-svc")
    Path(state.filename).mkdir()
    service = make_service(state)

    result = service.upscale_textures(
        UpscaleTextures(state_id=state.filename, model_name="inpainting")
    )

    assert result.slice_count == 2
    for index in range(len(state.image_slices)):
        assert state.upscaled_filename(index).exists()
        with Image.open(state.upscaled_filename(index)) as upscaled:
            assert upscaled.size == (40, 20)


def test_upscale_textures_requires_at_least_one_slice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path)
    state.image_slices = []
    service = make_service(state)

    with pytest.raises(ExportNotReady):
        service.upscale_textures(
            UpscaleTextures(state_id=state.filename, model_name="inpainting")
        )


# --- render_animation ------------------------------------------------------------


def test_render_animation_writes_the_requested_frame_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path, "appstate-animation-svc")
    Path(state.filename).mkdir()
    service = make_service(state)

    result = service.render_animation(
        RenderAnimation(state_id=state.filename, num_frames=3)
    )

    assert result.frame_count == 3
    frames = sorted((tmp_path / state.filename).glob("rendered_image_*.png"))
    assert len(frames) == 3
    for frame in frames:
        assert frame.stat().st_size > 0


def test_render_animation_rejects_non_positive_frame_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path)
    service = make_service(state)

    with pytest.raises(ExportNotReady):
        service.render_animation(RenderAnimation(state_id=state.filename, num_frames=0))


# --- slice_download_path -----------------------------------------------------------


def test_slice_download_path_returns_the_slice_file_and_its_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path)
    service = make_service(state)

    path, name = service.slice_download_path(state.filename, 1)

    assert path == Path(state.image_slices[1].filename)
    assert name == Path(state.image_slices[1].filename).name


def test_slice_download_path_rejects_an_out_of_range_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path)
    service = make_service(state)

    with pytest.raises(InvalidSliceIndex):
        service.slice_download_path(state.filename, 5)
