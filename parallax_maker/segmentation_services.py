"""Framework-neutral services for point-guided mask selection.

Browser event translation, logging, image URL serving, and UI control flow stay
in the Dash adapter.  This module owns the segmentation state machine and keeps
its mutations cache-only, matching the existing application behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from .controller import AppState, CompositeMode
from .segmentation import mask_from_depth

Point = tuple[int, int]


class MaskOperation(Enum):
    REPLACE = "replace"
    ADD = "add"
    SUBTRACT = "subtract"


class PointPolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class SegmentationServiceError(Exception):
    """Base class for segmentation-domain failures."""


class SegmentationNotReady(SegmentationServiceError):
    """Required image, depth, threshold, or interaction state is missing."""


class InvalidSegmentationPoint(SegmentationServiceError):
    """A point cannot be mapped into the current image and depth map."""


class InvalidMaskState(SegmentationServiceError):
    """A generated or existing mask is incompatible with the source image."""


class MultiPointModeRequired(SegmentationServiceError):
    """A multi-point-only command was requested outside multi-point mode."""


class NoPointsQueued(SegmentationServiceError):
    """A multi-point commit was requested without any queued points."""


class SegmentationModelFailed(SegmentationServiceError):
    """The segmentation model failed or returned an invalid result."""


class SegmentationStateRepository(Protocol):
    def load(self, state_id: str) -> AppState: ...


class CachedSegmentationStateRepository:
    """Load mutable interaction state through the existing process cache."""

    def load(self, state_id: str) -> AppState:
        return AppState.from_cache(state_id)


class SegmentationModelLike(Protocol):
    def segment_image(self, image: Image.Image) -> object: ...

    def mask_at_point_blended(self, point_input: dict[str, list[Point]]) -> object: ...


@dataclass(frozen=True)
class SelectDepthPoint:
    state_id: str
    point: Point
    operation: MaskOperation


@dataclass(frozen=True)
class SelectInstancePoint:
    state_id: str
    point: Point
    operation: MaskOperation
    polarity: PointPolarity


@dataclass(frozen=True)
class CommitMultiPoint:
    state_id: str


@dataclass(frozen=True)
class SetMultiPointMode:
    state_id: str
    enabled: bool


@dataclass(frozen=True)
class AppliedMaskResult:
    state_id: str
    point: Point | None
    depth: int | None
    operation: MaskOperation
    preview_image: Image.Image
    positive_points: tuple[Point, ...]
    negative_points: tuple[Point, ...]
    selected_slice: int | None


@dataclass(frozen=True)
class QueuedPointResult:
    state_id: str
    point: Point
    depth: int
    polarity: PointPolarity
    queue_size: int


@dataclass(frozen=True)
class MultiPointModeResult:
    state_id: str
    enabled: bool
    cleared_points: int


class SegmentationService:
    """Apply depth and model-derived masks without depending on a UI framework."""

    def __init__(
        self,
        state_repository: SegmentationStateRepository | None = None,
        *,
        model_factory: Callable[[], SegmentationModelLike],
    ) -> None:
        self._states = state_repository or CachedSegmentationStateRepository()
        self._model_factory = model_factory

    def select_depth_point(self, command: SelectDepthPoint) -> AppliedMaskResult:
        state = self._states.load(command.state_id)
        operation = self._validate_operation(command.operation)
        point, depth = self._validate_point_and_depth(state, command.point)
        thresholds = self._validate_thresholds(state, depth)
        threshold_index = next(
            (index for index, threshold in enumerate(thresholds) if depth <= threshold),
            None,
        )
        if threshold_index is None:
            raise SegmentationNotReady("the selected depth is outside the thresholds")
        candidate = mask_from_depth(
            state.depthMapData,
            int(thresholds[threshold_index - 1]),
            int(thresholds[threshold_index]),
        )
        source = self._require_image(state)
        candidate = self._validate_mask(candidate, source.size, "generated")
        self._validate_existing_mask(state, operation, source.size)
        state.slice_pixel = point
        state.slice_pixel_depth = depth
        mask = self._apply_candidate(state, candidate, operation, source.size)
        return self._applied_result(
            command.state_id,
            state,
            source,
            mask,
            point=point,
            depth=depth,
            operation=operation,
        )

    def select_instance_point(
        self, command: SelectInstancePoint
    ) -> AppliedMaskResult | QueuedPointResult:
        state = self._states.load(command.state_id)
        operation = self._validate_operation(command.operation)
        polarity = self._validate_polarity(command.polarity)
        point, depth = self._validate_point_and_depth(state, command.point)
        self._validate_thresholds(state, depth)

        if state.multi_point_mode:
            is_negative = polarity is PointPolarity.NEGATIVE
            state.slice_pixel = point
            state.slice_pixel_depth = depth
            state.points_selected.append((point, is_negative))
            return QueuedPointResult(
                state_id=command.state_id,
                point=point,
                depth=depth,
                polarity=polarity,
                queue_size=len(state.points_selected),
            )

        positive_points = (point,)
        negative_points: tuple[Point, ...] = ()
        source = self._instance_source(state)
        self._validate_existing_mask(state, operation, source.size)
        candidate = self._instance_mask(state, source, positive_points, negative_points)
        state.slice_pixel = point
        state.slice_pixel_depth = depth
        mask = self._apply_candidate(state, candidate, operation, source.size)
        return self._applied_result(
            command.state_id,
            state,
            source,
            mask,
            point=point,
            depth=depth,
            operation=operation,
            positive_points=positive_points,
            negative_points=negative_points,
        )

    def commit_multi_point(self, command: CommitMultiPoint) -> AppliedMaskResult:
        state = self._states.load(command.state_id)
        self._require_image(state)
        if not state.multi_point_mode:
            raise MultiPointModeRequired(
                "multi-point mode is required to commit points"
            )
        if not state.points_selected:
            raise NoPointsQueued("at least one point is required to commit")

        source = self._instance_source(state)
        positive_points: list[Point] = []
        negative_points: list[Point] = []
        for queued_item in state.points_selected:
            try:
                queued_point, is_negative = queued_item
            except (TypeError, ValueError):
                raise InvalidSegmentationPoint("a queued point is invalid") from None
            if not isinstance(is_negative, bool):
                raise InvalidSegmentationPoint("a queued point polarity is invalid")
            point = self._validate_source_point(queued_point, source.size)
            if is_negative:
                negative_points.append(point)
            else:
                positive_points.append(point)

        positives = tuple(positive_points)
        negatives = tuple(negative_points)
        candidate = self._instance_mask(state, source, positives, negatives)
        mask = self._apply_candidate(
            state, candidate, MaskOperation.REPLACE, source.size
        )
        # Successful commits intentionally retain the point queue and mode.
        return self._applied_result(
            command.state_id,
            state,
            source,
            mask,
            point=None,
            depth=None,
            operation=MaskOperation.REPLACE,
            positive_points=positives,
            negative_points=negatives,
        )

    def set_multi_point_mode(self, command: SetMultiPointMode) -> MultiPointModeResult:
        state = self._states.load(command.state_id)
        cleared_points = len(state.points_selected)
        state.multi_point_mode = command.enabled
        state.points_selected = []
        return MultiPointModeResult(
            state_id=command.state_id,
            enabled=command.enabled,
            cleared_points=cleared_points,
        )

    def _validate_point_and_depth(
        self, state: AppState, point: Point
    ) -> tuple[Point, int]:
        image = self._require_image(state)
        depth_map = state.depthMapData
        if depth_map is None:
            raise SegmentationNotReady("a depth map is required to select a point")
        try:
            depth_map = np.asarray(depth_map)
        except (TypeError, ValueError):
            raise SegmentationNotReady("the depth map is invalid") from None
        if depth_map.ndim != 2:
            raise SegmentationNotReady("the depth map must be two-dimensional")
        if depth_map.shape != (image.height, image.width):
            raise SegmentationNotReady("the depth map must match the input image")
        normalized = self._validate_source_point(point, image.size)
        try:
            depth = int(depth_map[normalized[1], normalized[0]])
        except (TypeError, ValueError, OverflowError):
            raise SegmentationNotReady("the selected depth is invalid") from None
        return normalized, depth

    @staticmethod
    def _validate_source_point(point: Point, source_size: tuple[int, int]) -> Point:
        if not isinstance(point, tuple) or len(point) != 2:
            raise InvalidSegmentationPoint("a point must contain x and y coordinates")
        point_x, point_y = point
        if isinstance(point_x, bool) or isinstance(point_y, bool):
            raise InvalidSegmentationPoint("point coordinates must be integers")
        if not isinstance(point_x, (int, np.integer)) or not isinstance(
            point_y, (int, np.integer)
        ):
            raise InvalidSegmentationPoint("point coordinates must be integers")
        normalized = (int(point_x), int(point_y))
        if (
            normalized[0] < 0
            or normalized[1] < 0
            or normalized[0] >= source_size[0]
            or normalized[1] >= source_size[1]
        ):
            raise InvalidSegmentationPoint("the selected point is outside the image")
        return normalized

    @staticmethod
    def _require_image(state: AppState) -> Image.Image:
        if state.imgData is None:
            raise SegmentationNotReady("an input image is required for segmentation")
        if not isinstance(state.imgData, Image.Image):
            raise SegmentationNotReady("the input image must be a PIL image")
        return state.imgData

    @staticmethod
    def _validate_thresholds(state: AppState, depth: int) -> tuple[int, ...]:
        thresholds = state.imgThresholds
        if thresholds is None:
            raise SegmentationNotReady("complete threshold boundaries are required")
        try:
            has_complete_boundaries = len(thresholds) == state.num_slices + 1
        except TypeError:
            has_complete_boundaries = False
        if not has_complete_boundaries:
            raise SegmentationNotReady("complete threshold boundaries are required")
        try:
            normalized = tuple(int(value) for value in thresholds)
        except (TypeError, ValueError, OverflowError):
            raise SegmentationNotReady(
                "threshold boundaries must be integers"
            ) from None
        if not any(depth <= threshold for threshold in normalized):
            raise SegmentationNotReady("the selected depth is outside the thresholds")
        return normalized

    @staticmethod
    def _validate_operation(operation: MaskOperation) -> MaskOperation:
        if not isinstance(operation, MaskOperation):
            raise InvalidMaskState("the mask operation is invalid")
        return operation

    @staticmethod
    def _validate_polarity(polarity: PointPolarity) -> PointPolarity:
        if not isinstance(polarity, PointPolarity):
            raise InvalidSegmentationPoint("the point polarity is invalid")
        return polarity

    def _instance_source(self, state: AppState) -> Image.Image:
        source = self._require_image(state)
        if state.selected_slice is None:
            return source
        index = state.selected_slice
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(state.image_slices)
        ):
            raise SegmentationNotReady("the selected slice index is invalid")
        try:
            source = state.slice_image_composed(index, CompositeMode.NONE)
        except (AssertionError, IndexError, TypeError, ValueError) as error:
            raise SegmentationNotReady(
                "the selected slice could not be composed"
            ) from error
        if not isinstance(source, Image.Image):
            raise SegmentationNotReady("the selected slice must be a PIL image")
        return source

    def _instance_mask(
        self,
        state: AppState,
        source: Image.Image,
        positive_points: tuple[Point, ...],
        negative_points: tuple[Point, ...],
    ) -> np.ndarray:
        model = state.segmentation_model
        created_model = model is None
        if created_model:
            try:
                model = self._model_factory()
            except Exception as error:
                raise SegmentationModelFailed(
                    "the segmentation model could not be created"
                ) from error
            if model is None:
                raise SegmentationModelFailed(
                    "the segmentation model factory returned no model"
                )
        if not callable(getattr(model, "segment_image", None)) or not callable(
            getattr(model, "mask_at_point_blended", None)
        ):
            raise SegmentationModelFailed("the segmentation model is invalid")
        try:
            model.segment_image(source)
            candidate = model.mask_at_point_blended(
                {
                    "positive_points": list(positive_points),
                    "negative_points": list(negative_points),
                }
            )
        except Exception as error:
            raise SegmentationModelFailed("the segmentation model failed") from error
        validated_mask = self._validate_mask(candidate, source.size, "generated")
        if created_model:
            state.segmentation_model = model
        return validated_mask

    def _apply_candidate(
        self,
        state: AppState,
        candidate: object,
        operation: MaskOperation,
        source_size: tuple[int, int],
    ) -> np.ndarray:
        candidate_mask = self._validate_mask(candidate, source_size, "generated")
        if state.slice_mask is None or operation is MaskOperation.REPLACE:
            state.slice_mask = candidate_mask
            return candidate_mask
        existing = self._validate_mask(state.slice_mask, source_size, "existing")
        if operation is MaskOperation.ADD:
            state.slice_mask = np.maximum(existing, candidate_mask)
        elif operation is MaskOperation.SUBTRACT:
            state.slice_mask = np.minimum(existing, 255 - candidate_mask)
        else:
            raise InvalidMaskState("the mask operation is invalid")
        return state.slice_mask

    @staticmethod
    def _validate_mask(
        mask: object, source_size: tuple[int, int], label: str
    ) -> np.ndarray:
        if mask is None:
            if label == "generated":
                raise SegmentationModelFailed("the segmentation model returned no mask")
            raise InvalidMaskState("the existing mask is missing")
        try:
            candidate = np.asarray(mask)
        except (TypeError, ValueError):
            raise InvalidMaskState(f"the {label} mask is invalid") from None
        expected_shape = (source_size[1], source_size[0])
        if candidate.ndim != 2 or candidate.shape != expected_shape:
            raise InvalidMaskState(
                f"the {label} mask must match the segmentation source"
            )
        if candidate.dtype != np.uint8:
            raise InvalidMaskState(f"the {label} mask must use uint8 values")
        return candidate

    def _validate_existing_mask(
        self,
        state: AppState,
        operation: MaskOperation,
        source_size: tuple[int, int],
    ) -> None:
        if state.slice_mask is not None and operation is not MaskOperation.REPLACE:
            self._validate_mask(state.slice_mask, source_size, "existing")

    @staticmethod
    def _applied_result(
        state_id: str,
        state: AppState,
        source: Image.Image,
        mask: np.ndarray,
        *,
        point: Point | None,
        depth: int | None,
        operation: MaskOperation,
        positive_points: tuple[Point, ...] = (),
        negative_points: tuple[Point, ...] = (),
    ) -> AppliedMaskResult:
        return AppliedMaskResult(
            state_id=state_id,
            point=point,
            depth=depth,
            operation=operation,
            preview_image=state.apply_mask(source, mask),
            positive_points=positive_points,
            negative_points=negative_points,
            selected_slice=state.selected_slice,
        )
