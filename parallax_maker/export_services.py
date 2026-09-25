"""Framework-neutral services for export/render.

Covers what ``PARITY.md`` lists under "Export/Render": glTF export (with
optional per-slice depth-displacement maps and upscaled textures), texture
upscaling, animation-frame rendering, and the raw-slice-download path
resolution used by ``webui.py``'s ``download_image``.

``webui.py``'s own ``export_state_as_gltf`` (webui.py:1346) is a plain helper
called independently from both ``gltf_export`` (download) and
``gltf_create`` (in-page viewer) - its own ``# XXX - this and the callback
above can be chained to avoid code duplication`` comment (webui.py:1314)
still applies there. ``ExportService.export_gltf`` below is the *one*
implementation both a future API download route and a future API "create for
viewing" route can share, reproducing that helper's exact behavior: it
regenerates a per-slice depth map only when one doesn't already exist on disk
and ``displacement_scale > 0``, and prefers an upscaled slice file over the
original whenever ``AppState.upscaled_filename`` exists for that slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .controller import AppState
from .depth import DepthEstimationModel
from .inpainting import create_inpainting_pipeline
from .scene import render_state_view
from .segmentation import export_gltf, generate_depth_map, render_image_sequence
from .utils import postprocess_depth_map

#: Matches webui.py's export_state_as_gltf's own postprocess_depth_map call.
DEPTH_MAP_FINAL_BLUR = 50

#: Matches webui.py's export_animation's own hardcoded push-distance factor
#: (webui.py:1497's ``push_distance=camera_distance * 0.75``).
DEFAULT_PUSH_DISTANCE_FACTOR = 0.75


class ExportServiceError(Exception):
    """Base class for export/render domain failures."""


class ExportNotReady(ExportServiceError):
    """Required image/slice state is missing for the requested export."""


class InvalidSliceIndex(ExportServiceError):
    """A slice index is missing, out of range, or otherwise invalid."""


class ExportStateRepository(Protocol):
    def load(self, state_id: str) -> AppState: ...


class CachedExportStateRepository:
    """Adapt the existing AppState cache."""

    def load(self, state_id: str) -> AppState:
        return AppState.from_cache(state_id)


@dataclass(frozen=True)
class ExportGltf:
    state_id: str
    displacement_scale: float = 0.0
    support_dof: bool = False
    #: Depth model used to regenerate a missing per-slice depth map; mirrors
    #: export_state_as_gltf's own ``modelname="midas"`` default, which is the
    #: value webui.py's download entry point (``gltf_export``) always uses -
    #: only the in-page viewer entry point (``gltf_create``) ever passes a
    #: different, dropdown-selected value.
    model_name: str = "midas"
    inline_images: bool = True


@dataclass(frozen=True)
class ExportedGltfResult:
    state_id: str
    gltf_path: Path
    slice_count: int
    used_upscaled: bool
    generated_depth_maps: bool


@dataclass(frozen=True)
class UpscaleTextures:
    state_id: str
    model_name: str
    server_address: str | None = None
    api_key: str | None = None
    #: Raw ComfyUI workflow JSON bytes (already decoded - unlike Dash's
    #: dcc.Upload contents, this is not a ``data:...;base64,`` URL); only
    #: consulted when ``model_name == "comfyui"``, mirroring
    #: ``create_inpainting_pipeline``.
    workflow: bytes | None = None


@dataclass(frozen=True)
class UpscaledTexturesResult:
    state_id: str
    slice_count: int


@dataclass(frozen=True)
class RenderAnimation:
    state_id: str
    num_frames: int
    push_distance_factor: float = DEFAULT_PUSH_DISTANCE_FACTOR


@dataclass(frozen=True)
class RenderedAnimationResult:
    state_id: str
    frame_count: int
    output_dir: Path


class ExportService:
    """glTF export, texture upscaling and animation rendering commands."""

    def __init__(
        self,
        state_repository: ExportStateRepository | None = None,
        *,
        depth_model_factory=DepthEstimationModel,
    ) -> None:
        self._states = state_repository or CachedExportStateRepository()
        self._depth_model_factory = depth_model_factory

    def export_gltf(self, command: ExportGltf) -> ExportedGltfResult:
        state = self._states.load(command.state_id)
        self._require_slices(state)

        generated_depth_maps = False
        depth_filenames: list[Path] = []
        if command.displacement_scale > 0:
            for index, image_slice in enumerate(state.image_slices):
                depth_filename = state.depth_filename(index)
                if not depth_filename.exists():
                    model = self._depth_model_factory(model=command.model_name)
                    if model != state.depth_estimation_model:
                        state.depth_estimation_model = model
                    depth_map = generate_depth_map(
                        image_slice.image[:, :, :3], model=state.depth_estimation_model
                    )
                    depth_map = postprocess_depth_map(
                        depth_map,
                        image_slice.image[:, :, 3],
                        final_blur=DEPTH_MAP_FINAL_BLUR,
                    )
                    Image.fromarray(depth_map).save(depth_filename, compress_level=1)
                    generated_depth_maps = True
                depth_filenames.append(depth_filename)

        used_upscaled = False
        slice_filenames: list[Path] = []
        for index, image_slice in enumerate(state.image_slices):
            upscaled_filename = state.upscaled_filename(index)
            if upscaled_filename.exists():
                slice_filenames.append(upscaled_filename)
                used_upscaled = True
            else:
                slice_filenames.append(Path(image_slice.filename))

        output_path = Path(command.state_id) / AppState.MODEL_FILE
        gltf_path = export_gltf(
            output_path,
            state.camera,
            state.image_slices,
            slice_filenames,
            depth_filenames,
            displacement_scale=command.displacement_scale,
            inline_images=command.inline_images,
            support_dof=command.support_dof,
        )

        return ExportedGltfResult(
            state_id=command.state_id,
            gltf_path=Path(gltf_path),
            slice_count=len(state.image_slices),
            used_upscaled=used_upscaled,
            generated_depth_maps=generated_depth_maps,
        )

    def upscale_textures(self, command: UpscaleTextures) -> UpscaledTexturesResult:
        state = self._states.load(command.state_id)
        self._require_slices(state)

        # This deliberately calls the pipeline-replacement helper exactly the
        # way webui.py's upscale_texture callback does: it invalidates and
        # rebuilds state.pipeline_spec/state.upscaler when the requested
        # model/server/workflow configuration differs from what is already
        # cached (see the handoff's "Cache reuse" bullet and
        # inpainting_services.py's own cache-identity discussion).
        if command.server_address is not None:
            state.server_address = command.server_address
        if command.api_key is not None:
            state.api_key = command.api_key
        create_inpainting_pipeline(command.model_name, command.workflow, state)

        state.upscale_slices()
        return UpscaledTexturesResult(
            state_id=command.state_id, slice_count=len(state.image_slices)
        )

    def render_animation(self, command: RenderAnimation) -> RenderedAnimationResult:
        state = self._states.load(command.state_id)
        self._require_slices(state)
        if command.num_frames <= 0:
            raise ExportNotReady("num_frames must be a positive integer")

        camera_distance = state.camera.camera_distance
        camera_matrix = state.camera_matrix()
        card_corners_3d_list = state.get_cards()
        camera_position = np.array([0, 0, -camera_distance], dtype=np.float32)

        output_dir = Path(command.state_id)
        render_image_sequence(
            output_dir,
            state.image_slices,
            card_corners_3d_list,
            camera_matrix,
            camera_position,
            push_distance=camera_distance * command.push_distance_factor,
            num_frames=command.num_frames,
            camera_rotation=state.camera.rotation_world_to_camera(),
            render=lambda position: render_state_view(
                state.image_slices, state.camera, position
            ),
        )

        return RenderedAnimationResult(
            state_id=command.state_id,
            frame_count=command.num_frames,
            output_dir=output_dir,
        )

    def slice_download_path(self, state_id: str, slice_index: int) -> tuple[Path, str]:
        """Resolve the raw slice PNG path/filename webui.py's ``download_image``
        sends verbatim (``dcc.send_file(image_path, Path(...).name)``)."""

        state = self._states.load(state_id)
        index = self._slice_index(state, slice_index)
        path = Path(state.image_slices[index].filename)
        return path, path.name

    @staticmethod
    def _require_slices(state: AppState) -> None:
        if not state.image_slices:
            raise ExportNotReady("at least one slice is required")

    @staticmethod
    def _slice_index(state: AppState, index: object) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(state.image_slices)
        ):
            raise InvalidSliceIndex(f"slice index {index!r} is invalid")
        return index
