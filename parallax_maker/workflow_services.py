"""Framework-neutral application services for the core image workflow.

This module owns workflow orchestration and persistence decisions.  UI adapters
remain responsible for decoding transport formats, rendering components, and
translating service outcomes into framework-specific control flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from PIL import Image

from .controller import AppState
from .depth import DepthEstimationModel
from .segmentation import (
    analyze_depth_histogram,
    generate_depth_map,
    generate_image_slices,
)


class WorkflowUnchanged(Exception):
    """The requested operation would not change workflow state."""


class WorkflowNotReady(Exception):
    """The requested operation cannot run with the current workflow state."""


@dataclass(frozen=True)
class StateSaveOptions:
    """Select which binary artifacts are persisted with state JSON."""

    save_image_slices: bool = True
    save_depth_map: bool = True
    save_input_image: bool = True


class StateRepository(Protocol):
    """Persistence boundary used by workflow services."""

    def create(self) -> tuple[AppState, str]: ...

    def load(self, state_id: str) -> AppState: ...

    def save(
        self, state_id: str, state: AppState, options: StateSaveOptions
    ) -> None: ...


class CachedAppStateRepository:
    """Adapt the existing AppState cache and filesystem persistence API."""

    def create(self) -> tuple[AppState, str]:
        return AppState.from_file_or_new(None)

    def load(self, state_id: str) -> AppState:
        return AppState.from_cache(state_id)

    def save(self, state_id: str, state: AppState, options: StateSaveOptions) -> None:
        state.to_file(
            state_id,
            save_image_slices=options.save_image_slices,
            save_depth_map=options.save_depth_map,
            save_input_image=options.save_input_image,
        )


@dataclass(frozen=True)
class UploadImage:
    image: Image.Image


@dataclass(frozen=True)
class UploadImageResult:
    state_id: str


@dataclass(frozen=True)
class GenerateDepth:
    state_id: str
    model_name: str


@dataclass(frozen=True)
class GenerateDepthResult:
    state_id: str


@dataclass(frozen=True)
class ConfigureThresholds:
    state_id: str
    num_slices: int


@dataclass(frozen=True)
class ConfigureThresholdsResult:
    state_id: str
    thresholds: list[int]
    missing_depth: bool


@dataclass(frozen=True)
class UpdateThresholdValues:
    state_id: str
    values: Sequence[int]
    num_slices: int


@dataclass(frozen=True)
class UpdateThresholdValuesResult:
    state_id: str
    values: list[int]
    preview_image: Image.Image | None


@dataclass(frozen=True)
class GenerateSlices:
    state_id: str


@dataclass(frozen=True)
class GenerateSlicesResult:
    state_id: str
    slice_count: int


class WorkflowService:
    """Perform core workflow commands independently of a UI framework."""

    DEPTH_SAVE_OPTIONS = StateSaveOptions(
        save_image_slices=False,
        save_depth_map=True,
        save_input_image=False,
    )
    SLICE_SAVE_OPTIONS = StateSaveOptions()

    def __init__(
        self,
        state_repository: StateRepository | None = None,
        *,
        depth_model_factory: Callable[..., object] = DepthEstimationModel,
        depth_generator: Callable[..., np.ndarray] = generate_depth_map,
        threshold_analyzer: Callable[..., list[int]] = analyze_depth_histogram,
        slice_generator: Callable[..., list] = generate_image_slices,
        progress_reporter: Callable[[int, int], None] | None = None,
        slice_expand: int = 5,
    ) -> None:
        self._states = state_repository or CachedAppStateRepository()
        self._depth_model_factory = depth_model_factory
        self._depth_generator = depth_generator
        self._threshold_analyzer = threshold_analyzer
        self._slice_generator = slice_generator
        self._progress_reporter = progress_reporter
        self._slice_expand = slice_expand

    def upload_image(self, command: UploadImage) -> UploadImageResult:
        state, state_id = self._states.create()
        state.set_img_data(command.image)
        return UploadImageResult(state_id=state_id)

    def generate_depth(self, command: GenerateDepth) -> GenerateDepthResult:
        state = self._states.load(command.state_id)
        image = state.imgData
        if image.mode == "RGBA":
            image = image.convert("RGB")

        depth_model = self._depth_model_factory(model=command.model_name)
        if depth_model != state.depth_estimation_model:
            state.depth_estimation_model = depth_model

        state.depthMapData = self._depth_generator(
            np.array(image),
            model=state.depth_estimation_model,
            progress_callback=self._progress_reporter,
        )
        state.imgThresholds = None
        self._states.save(command.state_id, state, self.DEPTH_SAVE_OPTIONS)
        return GenerateDepthResult(state_id=command.state_id)

    def configure_thresholds(
        self, command: ConfigureThresholds
    ) -> ConfigureThresholdsResult:
        state = self._states.load(command.state_id)
        if (
            state.num_slices == command.num_slices
            and state.imgThresholds is not None
            and len(state.imgThresholds) == command.num_slices + 1
        ):
            raise WorkflowUnchanged("number of slices and thresholds are unchanged")

        state.num_slices = command.num_slices
        missing_depth = state.depthMapData is None
        if missing_depth:
            state.imgThresholds = [0]
            state.imgThresholds.extend(
                [
                    i * (255 // (command.num_slices - 1))
                    for i in range(1, command.num_slices)
                ]
            )
        elif (
            state.imgThresholds is None
            or len(state.imgThresholds) != command.num_slices
        ):
            state.imgThresholds = self._threshold_analyzer(
                state.depthMapData, num_slices=command.num_slices
            )

        return ConfigureThresholdsResult(
            state_id=command.state_id,
            thresholds=list(state.imgThresholds),
            missing_depth=missing_depth,
        )

    def update_threshold_values(
        self, command: UpdateThresholdValues
    ) -> UpdateThresholdValuesResult:
        state = self._states.load(command.state_id)
        values = list(command.values)
        if state.imgThresholds[1:-1] == values:
            raise WorkflowUnchanged("threshold values are unchanged")

        if values[0] <= 0:
            values[0] = 1
        for index in range(1, command.num_slices - 1):
            if values[index] <= values[index - 1]:
                values[index] = values[index - 1] + 1

        if values[-1] >= 255:
            values[-1] = 254
        for index in range(command.num_slices - 3, -1, -1):
            if values[index] >= values[index + 1]:
                values[index] = values[index + 1] - 1

        state.imgThresholds[1:-1] = values

        preview_image = None
        if state.slice_pixel:
            state.slice_mask, _ = state.depth_slice_from_pixel(
                state.slice_pixel[0], state.slice_pixel[1]
            )
            if state.slice_mask is not None:
                preview_image = state.apply_mask(state.imgData, state.slice_mask)
            else:
                preview_image = state.imgData

        return UpdateThresholdValuesResult(
            state_id=command.state_id,
            values=values,
            preview_image=preview_image,
        )

    def generate_slices(self, command: GenerateSlices) -> GenerateSlicesResult:
        state = self._states.load(command.state_id)
        if state.depthMapData is None:
            raise WorkflowNotReady("a depth map is required to generate slices")

        state.image_slices = self._slice_generator(
            np.array(state.imgData),
            state.depthMapData,
            state.imgThresholds,
            num_expand=self._slice_expand,
        )
        self._states.save(command.state_id, state, self.SLICE_SAVE_OPTIONS)
        return GenerateSlicesResult(
            state_id=command.state_id,
            slice_count=len(state.image_slices),
        )
