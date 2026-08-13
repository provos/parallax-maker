"""Small, deterministic artifacts shared by browser workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from parallax_maker.controller import AppState
from parallax_maker.segmentation import generate_image_slices

from .fakes import FakeDepthEstimationModel


FIXTURE_STATE_PREFIX = "appstate-e2e-restore"


def create_input_image(size: tuple[int, int] = (320, 240)) -> Image.Image:
    """Create a colorful input whose orientation and pixel positions are obvious."""

    width, height = size
    x = np.linspace(20, 220, width, dtype=np.uint8)
    y = np.linspace(10, 180, height, dtype=np.uint8)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = x[None, :]
    image[:, :, 1] = y[:, None]
    image[:, :, 2] = 70
    result = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(result)
    draw.rectangle((24, 24, 104, 96), fill=(240, 50, 45))
    draw.ellipse((205, 55, 292, 142), fill=(35, 210, 90))
    draw.polygon(((135, 205), (175, 125), (215, 205)), fill=(35, 90, 245))
    return result


def create_fixture_state(root: Path, state_name: str | None = None) -> Path:
    """Write a complete restorable state and return its JSON path."""

    root = root.resolve()
    if root != Path.cwd().resolve():
        raise ValueError("fixture root must be the current working directory")
    if state_name is None:
        state_name = FIXTURE_STATE_PREFIX
    if Path(state_name).name != state_name or not state_name.startswith("appstate-"):
        raise ValueError("state_name must be a single appstate-* directory name")
    state_dir = root / state_name
    state = AppState()
    state.filename = state_name
    state.set_img_data(create_input_image())
    state.depthMapData = FakeDepthEstimationModel("dinov2").depth_map(state.imgData)
    state.num_slices = 3
    state.imgThresholds = [0, 85, 170, 255]
    state.depth_model_name = "dinov2"
    state.inpainting_model_name = (
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    )
    state.dark_mode = True
    state.camera.camera_distance = 125.0
    state.camera.focal_length = 475.0
    state.camera.max_distance = 140.0
    state.mesh_displacement = 15
    state.image_slices = generate_image_slices(
        np.asarray(state.imgData),
        state.depthMapData,
        state.imgThresholds,
        num_expand=5,
    )
    for index, image_slice in enumerate(state.image_slices):
        image_slice.positive_prompt = f"fixture foreground {index}"
        image_slice.negative_prompt = f"fixture exclusion {index}"
    # AppState's serving URLs assume slice filenames remain cwd-relative.  The
    # E2E server has already changed into ``root``, so persist with that relative
    # directory while retaining an absolute path for Flask's fixture response.
    state.to_file(Path(state_name))
    return state_dir / AppState.STATE_FILE
