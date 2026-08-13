import json
from pathlib import Path

import numpy as np
from PIL import Image

from .e2e_support.fakes import (
    INPAINT_PALETTES,
    FakeDepthEstimationModel,
    FakeInpaintingModel,
    FakeSegmentationModel,
    FakeUpscaler,
)
from .e2e_support.fixtures import create_fixture_state, create_input_image
from .e2e_server import _mask_metadata


def test_fake_depth_is_banded_and_spans_byte_range():
    depth = FakeDepthEstimationModel("dinov2").depth_map(
        np.zeros((32, 80, 3), dtype=np.uint8)
    )

    assert depth.shape == (32, 80)
    assert depth.dtype == np.uint8
    assert depth[0, 0] == 0
    assert depth[-1, -1] == 255
    assert depth[0, 10] - depth[0, 9] == 32
    assert depth[-1, 0] == 31


def test_fake_segmentation_uses_points_and_negative_holes():
    model = FakeSegmentationModel()
    model.segment_image(Image.new("RGB", (120, 100)))

    mask = model.mask_at_point_blended(
        {"positive_points": [(40, 50)], "negative_points": [(45, 50)]}
    )

    assert mask[50, 25] == 255
    assert mask[50, 45] == 0
    assert mask[10, 100] == 0


def test_fake_inpainting_cycles_exact_checkerboard_palettes():
    model = FakeInpaintingModel()
    model.load_model()
    image = Image.new("RGB", (32, 32), "black")
    mask = Image.new("L", image.size, 255)

    results = [
        model.inpaint("", "", image, mask, padding=0, blur_radius=0)
        for _ in range(3)
    ]

    for result, palette in zip(results, INPAINT_PALETTES):
        pixels = np.asarray(result.convert("RGB"))
        assert tuple(pixels[2, 2]) == palette[0]
        assert tuple(pixels[2, 10]) == palette[1]


def test_fake_upscaler_is_two_x_and_marks_border():
    result = FakeUpscaler().upscale_image_tiled(Image.new("RGB", (10, 6), "navy"))

    assert result.size == (20, 12)
    assert result.getpixel((0, 0)) == (255, 255, 0)


def test_restore_fixture_uses_relative_safe_state_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = create_fixture_state(Path.cwd())

    assert state_path.exists()
    assert create_input_image().size == (320, 240)
    state_json = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_json["mesh_displacement"] == 15
    assert state_json["image_slices_filenames"] == [
        f"appstate-e2e-restore/image_slice_{index}.png" for index in range(3)
    ]


def test_restore_fixture_supports_isolated_state_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    first = create_fixture_state(Path.cwd(), "appstate-e2e-first")
    second = create_fixture_state(Path.cwd(), "appstate-e2e-second")

    assert first.parent != second.parent
    assert json.loads(first.read_text(encoding="utf-8"))["filename"] == (
        "appstate-e2e-first"
    )
    assert json.loads(second.read_text(encoding="utf-8"))["filename"] == (
        "appstate-e2e-second"
    )


def test_mask_metadata_reports_exact_inside_outside_and_bounds():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[7:13, 9:18] = 255

    metadata = _mask_metadata(mask)

    assert metadata["bounds"] == [9, 7, 17, 12]
    assert metadata["max"] == 255
    assert metadata["nonzero"] == 54
    assert mask[metadata["inside"][1], metadata["inside"][0]] == 255
    assert mask[metadata["outside"][1], metadata["outside"][0]] == 0
