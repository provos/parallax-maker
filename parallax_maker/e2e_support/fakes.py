"""Deterministic inference substitutes used exclusively by browser tests.

The substitutes sit at model/provider boundaries.  Dash callbacks, ``AppState``,
mask feathering/compositing, versioning, exports, and filesystem persistence keep
using the production implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


INPAINT_PALETTES = (
    ((0, 255, 255), (255, 0, 255)),
    ((255, 128, 0), (0, 64, 255)),
    ((64, 255, 64), (128, 0, 192)),
)
CHECK_SIZE = 8


class FakeDepthEstimationModel:
    """Return an eight-band depth field with a ramp inside every band."""

    MODELS = ["midas", "zoedepth", "dinov2"]

    def __init__(self, model: str = "midas") -> None:
        if model not in self.MODELS:
            raise ValueError(f"Unsupported depth model: {model}")
        self._model_name = model
        self.model = "e2e-depth-model"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FakeDepthEstimationModel)
            and self._model_name == other._model_name
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_model(self, progress_callback=None) -> str:
        if progress_callback:
            progress_callback(100, 100)
        return self.model

    def depth_map(self, image: Any, progress_callback=None) -> np.ndarray:
        height, width = np.asarray(image).shape[:2]
        x = np.arange(width, dtype=np.uint16)
        y = np.arange(height, dtype=np.uint16)
        bands = np.minimum((x * 8) // max(width, 1), 7) * 32
        ramp = (y * 31) // max(height - 1, 1)
        result = np.minimum(bands[None, :] + ramp[:, None], 255).astype(np.uint8)
        if progress_callback:
            progress_callback(100, 100)
        return result


def _normalise_points(
    point_input: Any,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if isinstance(point_input, tuple):
        return [point_input], []
    if isinstance(point_input, list):
        return list(point_input), []
    if isinstance(point_input, dict):
        return (
            list(point_input.get("positive_points", [])),
            list(point_input.get("negative_points", [])),
        )
    raise ValueError("Invalid point input")


class FakeSegmentationModel:
    """Point-guided disk masks: positives add regions and negatives cut holes."""

    MODELS = ["mask2former", "sam"]

    def __init__(self, model: str = "sam") -> None:
        if model not in self.MODELS:
            raise ValueError(f"Unsupported segmentation model: {model}")
        self.model_name = model
        self.model = "e2e-segmentation-model"
        self.image_processor = None
        self.image: Image.Image | None = None
        self.mask: np.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FakeSegmentationModel)
            and self.model_name == other.model_name
        )

    def load_model(self) -> str:
        return self.model

    def segment_image(self, image: Any) -> None:
        self.image = (
            Image.fromarray(np.asarray(image))
            if isinstance(image, np.ndarray)
            else image.copy()
        ).convert("RGB")
        self.mask = None
        return None

    def mask_at_point(self, point_input: Any) -> np.ndarray:
        return self.mask_at_point_blended(point_input)

    def mask_at_point_blended(self, point_input: Any) -> np.ndarray:
        if self.image is None:
            raise RuntimeError("segment_image must be called before mask_at_point")

        positive_points, negative_points = _normalise_points(point_input)
        width, height = self.image.size
        radius = max(8, min(width, height) // 6)
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for x, y in positive_points:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=255
            )
        hole_radius = max(5, (radius * 2) // 3)
        for x, y in negative_points:
            draw.ellipse(
                (
                    x - hole_radius,
                    y - hole_radius,
                    x + hole_radius,
                    y + hole_radius,
                ),
                fill=0,
            )

        self.mask = np.asarray(mask, dtype=np.uint8)
        return self.mask


def checkerboard(size: tuple[int, int], variant: int) -> Image.Image:
    """Create an exact RGB checkerboard for one of the three candidates."""

    width, height = size
    first, second = INPAINT_PALETTES[variant % len(INPAINT_PALETTES)]
    yy, xx = np.indices((height, width))
    cells = ((xx // CHECK_SIZE) + (yy // CHECK_SIZE)) % 2
    result = np.empty((height, width, 3), dtype=np.uint8)
    result[cells == 0] = first
    result[cells == 1] = second
    return Image.fromarray(result, mode="RGB")


def _load_inpainting_base():
    from parallax_maker.inpainting import InpaintingModel

    return InpaintingModel


class FakeInpaintingModel(_load_inpainting_base()):
    """Use the real inpainting composition around a checkerboard generator."""

    def __init__(
        self,
        model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self._e2e_requested_model = model
        # Force every selectable backend through the same local production branch.
        self.model = self.MODELS[0]
        self._e2e_call_count = 0

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FakeInpaintingModel)
            and self._e2e_requested_model == other._e2e_requested_model
        )

    def load_model(self):
        self._dimension = None
        self.pipeline = "e2e-checkerboard-pipeline"
        return self.pipeline

    def inpaint_diffusers(
        self,
        resize_init_image,
        resize_mask_image,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        num_inference_steps,
        seed,
    ):
        del resize_mask_image, prompt, negative_prompt, strength
        del guidance_scale, num_inference_steps, seed
        variant = self._e2e_call_count % len(INPAINT_PALETTES)
        self._e2e_call_count += 1
        return checkerboard(resize_init_image.size, variant)


class FakeUpscaler:
    """Nearest-neighbour 2x enlargement with a conspicuous yellow border."""

    def __init__(self, model_name="swin2sr", external_model=None) -> None:
        del external_model
        self.model_name = model_name
        self.scale_factor = 2

    def upscale_image_tiled(self, image, overlap=64, prompt=None, negative_prompt=None):
        del overlap, prompt, negative_prompt
        source = image if isinstance(image, Image.Image) else Image.fromarray(image)
        result = source.resize(
            (source.width * self.scale_factor, source.height * self.scale_factor),
            Image.Resampling.NEAREST,
        )
        draw = ImageDraw.Draw(result)
        color = (255, 255, 0, 255) if result.mode == "RGBA" else (255, 255, 0)
        draw.rectangle(
            (0, 0, result.width - 1, result.height - 1), outline=color, width=4
        )
        return result


@dataclass
class FakeExternalProvider:
    """Provider-shaped object for configuration callbacks, without any I/O."""

    api_key: str | None = None

    def validate_key(self):
        return True, 123.0

    def upscale_image(self, image, prompt="", negative_prompt=""):
        del prompt, negative_prompt
        return FakeUpscaler().upscale_image_tiled(image)


def _blocked_network(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("External network access is disabled by the E2E test server")


def _offline_gltf_iframe(gltf_uri: str) -> str:
    return (
        "<html><body><p id='e2e-gltf-viewer' "
        f"data-model-uri='{gltf_uri}'>glTF created</p></body></html>"
    )


def install_fakes():
    """Install process-local fakes and return the already configured Dash module."""

    import requests.sessions

    from parallax_maker import automatic1111, comfyui, controller, falai, inpainting
    from parallax_maker import stabilityai, webui
    from parallax_maker import components

    # Classes captured as module globals by callback functions.
    webui.DepthEstimationModel = FakeDepthEstimationModel
    webui.SegmentationModel = FakeSegmentationModel
    webui.InpaintingModel = FakeInpaintingModel
    controller.Upscaler = FakeUpscaler
    controller.StabilityAI = FakeExternalProvider
    controller.FalAI = FakeExternalProvider

    # Production URLs use whole-second timestamps. Several legitimate callbacks
    # can complete inside one second, leaving React with an unchanged ``src`` and
    # a stale image. Give the test process a monotonic query value so each real
    # file write is observable without sleeps.
    original_serve_main_image = controller.AppState.serve_main_image
    image_versions = count()

    def serve_main_image_with_unique_url(state, image):
        url = original_serve_main_image(state, image)
        path = url.split("?", maxsplit=1)[0]
        return f"{path}?e2e-v={next(image_versions)}"

    controller.AppState.serve_main_image = serve_main_image_with_unique_url

    # The production factory resolves this module global when a callback invokes it.
    inpainting.InpaintingModel = FakeInpaintingModel
    inpainting.StabilityAI = FakeExternalProvider
    inpainting.FalAI = FakeExternalProvider
    stabilityai.StabilityAI = FakeExternalProvider
    falai.FalAI = FakeExternalProvider

    # Configuration probes are deterministic too.
    components.StabilityAI = FakeExternalProvider
    components.make_models_request = lambda server_address: ["e2e-model"]
    components.get_history = lambda server_address, prompt_id: {"e2e": "ready"}
    automatic1111.make_models_request = components.make_models_request
    automatic1111.make_img2img_request = _blocked_network
    comfyui.inpainting_comfyui = _blocked_network
    inpainting.inpainting_comfyui = _blocked_network

    # Browser tests run offline.  The normal application retains both scripts.
    webui.app.config.external_scripts = []
    webui.get_gltf_iframe = _offline_gltf_iframe
    # Font Awesome normally gives icon-only controls their glyph dimensions.
    # Preserve clickable geometry offline without changing production assets.
    webui.app.index_string = webui.app.index_string.replace(
        "</head>",
        "<style>.fa-solid,.fas{min-width:1rem;min-height:1rem}</style></head>",
    )
    requests.sessions.Session.request = _blocked_network
    return webui
