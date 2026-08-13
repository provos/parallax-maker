from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest

from .controller import AppState, CompositeMode
from .segmentation_services import (
    CommitMultiPoint,
    InvalidMaskState,
    InvalidSegmentationPoint,
    MaskOperation,
    MultiPointModeRequired,
    NoPointsQueued,
    PointPolarity,
    SegmentationModelFailed,
    SegmentationNotReady,
    SegmentationService,
    SelectDepthPoint,
    SelectInstancePoint,
    SetMultiPointMode,
)
from .slice import ImageSlice


class RecordingStateRepository:
    def __init__(self, state: AppState, state_id: str = "appstate-test") -> None:
        self.state = state
        self.state_id = state_id
        self.loaded_ids: list[str] = []
        self.save_calls: list[object] = []

    def load(self, state_id: str) -> AppState:
        assert state_id == self.state_id
        self.loaded_ids.append(state_id)
        return self.state

    def save(self, *args, **kwargs) -> None:
        self.save_calls.append((args, kwargs))


class RecordingSegmentationModel:
    def __init__(self, mask: np.ndarray | None) -> None:
        self.mask = mask
        self.segmented_images: list[Image.Image] = []
        self.point_inputs: list[dict[str, list[tuple[int, int]]]] = []

    def segment_image(self, image: Image.Image) -> None:
        self.segmented_images.append(image)

    def mask_at_point_blended(self, point_input):
        self.point_inputs.append(point_input)
        return None if self.mask is None else self.mask.copy()


class FailingSegmentationModel(RecordingSegmentationModel):
    def segment_image(self, image: Image.Image) -> None:
        super().segment_image(image)
        raise RuntimeError("inference failed")


def make_ready_state() -> AppState:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgData = Image.new("RGB", (4, 3), (20, 40, 60))
    state.depthMapData = np.array(
        [[10, 60, 130, 220], [20, 70, 140, 230], [30, 80, 150, 240]],
        dtype=np.uint8,
    )
    state.imgThresholds = [0, 90, 180, 255]
    return state


def make_service(
    state: AppState,
    mask: np.ndarray | None = None,
) -> tuple[SegmentationService, RecordingStateRepository, list]:
    if mask is None:
        mask = np.array(
            [[0, 255, 0, 0], [0, 255, 255, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
    repository = RecordingStateRepository(state)
    models = []

    def model_factory():
        model = RecordingSegmentationModel(mask)
        models.append(model)
        return model

    return (
        SegmentationService(repository, model_factory=model_factory),
        repository,
        models,
    )


def test_select_depth_point_records_point_depth_and_replaces_mask() -> None:
    state = make_ready_state()
    service, repository, _ = make_service(state)

    result = service.select_depth_point(
        SelectDepthPoint("appstate-test", (2, 1), MaskOperation.REPLACE)
    )

    assert result.point == (2, 1)
    assert result.depth == 140
    assert result.operation is MaskOperation.REPLACE
    assert result.selected_slice is None
    assert state.slice_pixel == (2, 1)
    assert state.slice_pixel_depth == 140
    expected_mask = np.array(
        [[0, 0, 255, 0], [0, 0, 255, 0], [0, 0, 255, 0]], dtype=np.uint8
    )
    assert np.array_equal(state.slice_mask, expected_mask)
    assert isinstance(result.preview_image, Image.Image)
    assert repository.save_calls == []


def test_depth_equal_to_first_threshold_preserves_wraparound_empty_mask() -> None:
    state = make_ready_state()
    state.depthMapData[0, 0] = 0
    service, repository, _ = make_service(state)

    result = service.select_depth_point(
        SelectDepthPoint("appstate-test", (0, 0), MaskOperation.REPLACE)
    )

    assert result.depth == 0
    assert state.slice_pixel == (0, 0)
    assert state.slice_pixel_depth == 0
    assert np.count_nonzero(state.slice_mask) == 0
    assert repository.save_calls == []


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            MaskOperation.REPLACE,
            [[0, 255, 0], [0, 255, 255], [0, 0, 0]],
        ),
        (
            MaskOperation.ADD,
            [[255, 255, 0], [0, 255, 255], [0, 0, 255]],
        ),
        (
            MaskOperation.SUBTRACT,
            [[255, 0, 0], [0, 0, 0], [0, 0, 255]],
        ),
    ],
)
def test_single_instance_point_combines_masks_exactly(operation, expected) -> None:
    state = make_ready_state()
    state.imgData = Image.new("RGB", (3, 3), "navy")
    state.depthMapData = np.full((3, 3), 100, dtype=np.uint8)
    state.slice_mask = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    candidate = np.array([[0, 255, 0], [0, 255, 255], [0, 0, 0]], dtype=np.uint8)
    service, repository, models = make_service(state, candidate)

    result = service.select_instance_point(
        SelectInstancePoint("appstate-test", (1, 1), operation, PointPolarity.NEGATIVE)
    )

    assert np.array_equal(state.slice_mask, np.array(expected, dtype=np.uint8))
    assert result.positive_points == ((1, 1),)
    assert result.negative_points == ()
    assert models[0].point_inputs == [
        {"positive_points": [(1, 1)], "negative_points": []}
    ]
    assert repository.save_calls == []


def test_multipoint_queue_records_polarity_without_changing_mask() -> None:
    state = make_ready_state()
    state.multi_point_mode = True
    original_mask = np.full((3, 4), 17, dtype=np.uint8)
    state.slice_mask = original_mask.copy()
    service, repository, models = make_service(state)

    positive = service.select_instance_point(
        SelectInstancePoint(
            "appstate-test",
            (1, 1),
            MaskOperation.REPLACE,
            PointPolarity.POSITIVE,
        )
    )
    negative = service.select_instance_point(
        SelectInstancePoint(
            "appstate-test",
            (2, 1),
            MaskOperation.SUBTRACT,
            PointPolarity.NEGATIVE,
        )
    )

    assert positive.polarity is PointPolarity.POSITIVE
    assert positive.queue_size == 1
    assert negative.polarity is PointPolarity.NEGATIVE
    assert negative.queue_size == 2
    assert state.points_selected == [((1, 1), False), ((2, 1), True)]
    assert np.array_equal(state.slice_mask, original_mask)
    assert models == []
    assert repository.save_calls == []


def test_commit_partitions_points_and_retains_queue_and_mode() -> None:
    state = make_ready_state()
    state.multi_point_mode = True
    state.points_selected = [((1, 1), False), ((2, 1), True), ((3, 2), False)]
    service, repository, models = make_service(state)

    result = service.commit_multi_point(CommitMultiPoint("appstate-test"))

    assert result.positive_points == ((1, 1), (3, 2))
    assert result.negative_points == ((2, 1),)
    assert models[0].point_inputs == [
        {
            "positive_points": [(1, 1), (3, 2)],
            "negative_points": [(2, 1)],
        }
    ]
    assert state.points_selected == [((1, 1), False), ((2, 1), True), ((3, 2), False)]
    assert state.multi_point_mode is True
    assert repository.save_calls == []


@pytest.mark.parametrize("enabled", [True, False])
def test_set_multi_point_mode_clears_queue(enabled) -> None:
    state = make_ready_state()
    state.multi_point_mode = not enabled
    state.points_selected = [((1, 1), False), ((2, 1), True)]
    service, repository, _ = make_service(state)

    result = service.set_multi_point_mode(
        SetMultiPointMode("appstate-test", enabled=enabled)
    )

    assert result.enabled is enabled
    assert result.cleared_points == 2
    assert state.multi_point_mode is enabled
    assert state.points_selected == []
    assert repository.save_calls == []


def test_selected_slice_is_model_source_and_preview_base() -> None:
    state = make_ready_state()
    selected_image = np.zeros((3, 4, 4), dtype=np.uint8)
    selected_image[:, :, :3] = (200, 100, 50)
    selected_image[:, :, 3] = 255
    state.image_slices = [ImageSlice(selected_image, depth=100)]
    state.selected_slice = 0
    source = Image.new("RGBA", (4, 3), (90, 80, 70, 255))
    candidate = np.array(
        [[0, 255, 0, 0], [0, 255, 255, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    expected_preview = state.apply_mask(source, candidate)
    service, _, models = make_service(state)

    # AppState uses slots, so patch the class method for this one state.
    original = AppState.slice_image_composed
    calls = []

    def selected_source(candidate, index, mode):
        if candidate is state:
            calls.append((index, mode))
            return source
        return original(candidate, index, mode)

    AppState.slice_image_composed = selected_source
    try:
        result = service.select_instance_point(
            SelectInstancePoint(
                "appstate-test",
                (1, 1),
                MaskOperation.REPLACE,
                PointPolarity.POSITIVE,
            )
        )
    finally:
        AppState.slice_image_composed = original

    assert calls == [(0, CompositeMode.NONE)]
    assert models[0].segmented_images == [source]
    assert np.array_equal(
        np.asarray(result.preview_image), np.asarray(expected_preview)
    )
    assert result.selected_slice == 0
    assert state.selected_slice == 0


def test_instance_model_is_injected_once_and_reused() -> None:
    state = make_ready_state()
    service, _, models = make_service(state)

    service.select_instance_point(
        SelectInstancePoint(
            "appstate-test",
            (1, 1),
            MaskOperation.REPLACE,
            PointPolarity.POSITIVE,
        )
    )
    service.select_instance_point(
        SelectInstancePoint(
            "appstate-test",
            (2, 1),
            MaskOperation.REPLACE,
            PointPolarity.POSITIVE,
        )
    )

    assert len(models) == 1
    assert state.segmentation_model is models[0]
    assert len(models[0].segmented_images) == 2


def test_invalid_mask_operation_fails_before_state_mutation() -> None:
    state = make_ready_state()
    service, repository, models = make_service(state)

    with pytest.raises(InvalidMaskState, match="operation"):
        service.select_instance_point(
            SelectInstancePoint(
                "appstate-test", (1, 1), "merge", PointPolarity.POSITIVE
            )
        )

    assert state.slice_pixel is None
    assert state.slice_mask is None
    assert models == []
    assert repository.save_calls == []


def test_invalid_point_polarity_fails_before_state_mutation() -> None:
    state = make_ready_state()
    service, repository, models = make_service(state)

    with pytest.raises(InvalidSegmentationPoint, match="polarity"):
        service.select_instance_point(
            SelectInstancePoint(
                "appstate-test", (1, 1), MaskOperation.REPLACE, "neutral"
            )
        )

    assert state.slice_pixel is None
    assert state.slice_mask is None
    assert models == []
    assert repository.save_calls == []


@pytest.mark.parametrize(
    "queued",
    [
        [(1, 1)],
        [((1, 1),)],
        [((1, 1), "negative")],
        [((99, 1), False)],
    ],
)
def test_commit_rejects_malformed_or_outside_queued_points(queued) -> None:
    state = make_ready_state()
    state.multi_point_mode = True
    state.points_selected = queued
    service, repository, models = make_service(state)

    with pytest.raises(InvalidSegmentationPoint, match="queued|point"):
        service.commit_multi_point(CommitMultiPoint("appstate-test"))

    assert state.slice_mask is None
    assert state.points_selected == queued
    assert models == []
    assert repository.save_calls == []


@pytest.mark.parametrize(
    "existing_mask",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((3, 4), dtype=np.float32),
    ],
)
def test_existing_mask_is_validated_before_instance_model_invocation(
    existing_mask,
) -> None:
    state = make_ready_state()
    state.slice_mask = existing_mask
    service, repository, models = make_service(state)

    with pytest.raises(InvalidMaskState):
        service.select_instance_point(
            SelectInstancePoint(
                "appstate-test",
                (1, 1),
                MaskOperation.ADD,
                PointPolarity.POSITIVE,
            )
        )

    assert models == []
    assert state.segmentation_model is None
    assert state.slice_mask is existing_mask
    assert repository.save_calls == []


def test_commit_does_not_revalidate_depth_or_thresholds_after_points_are_queued() -> (
    None
):
    state = make_ready_state()
    state.multi_point_mode = True
    state.points_selected = [((1, 1), False), ((2, 1), True)]
    state.depthMapData = None
    state.imgThresholds = None
    service, repository, models = make_service(state)

    result = service.commit_multi_point(CommitMultiPoint("appstate-test"))

    assert result.positive_points == ((1, 1),)
    assert result.negative_points == ((2, 1),)
    assert models[0].point_inputs == [
        {"positive_points": [(1, 1)], "negative_points": [(2, 1)]}
    ]
    assert state.points_selected == [((1, 1), False), ((2, 1), True)]
    assert repository.save_calls == []


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda state: setattr(state, "imgData", None), SegmentationNotReady, "image"),
        (
            lambda state: setattr(state, "depthMapData", None),
            SegmentationNotReady,
            "depth",
        ),
        (
            lambda state: setattr(state, "imgThresholds", None),
            SegmentationNotReady,
            "threshold",
        ),
        (
            lambda state: setattr(state, "imgThresholds", [0, 255]),
            SegmentationNotReady,
            "threshold",
        ),
        (
            lambda state: setattr(state, "selected_slice", 99),
            SegmentationNotReady,
            "selected slice",
        ),
    ],
)
def test_instance_selection_validates_state(mutation, error, message) -> None:
    state = make_ready_state()
    mutation(state)
    service, repository, _ = make_service(state)

    with pytest.raises(error, match=message):
        service.select_instance_point(
            SelectInstancePoint(
                "appstate-test",
                (1, 1),
                MaskOperation.REPLACE,
                PointPolarity.POSITIVE,
            )
        )

    assert repository.save_calls == []


@pytest.mark.parametrize("point", [(-1, 0), (0, -1), (4, 0), (0, 3)])
def test_point_must_be_inside_image_and_depth_bounds(point) -> None:
    state = make_ready_state()
    service, repository, _ = make_service(state)

    with pytest.raises(InvalidSegmentationPoint, match="point"):
        service.select_depth_point(
            SelectDepthPoint("appstate-test", point, MaskOperation.REPLACE)
        )

    assert state.slice_pixel is None
    assert repository.save_calls == []


def test_commit_requires_multi_point_mode() -> None:
    state = make_ready_state()
    state.points_selected = [((1, 1), False)]
    service, _, _ = make_service(state)

    with pytest.raises(MultiPointModeRequired):
        service.commit_multi_point(CommitMultiPoint("appstate-test"))


def test_commit_requires_queued_points() -> None:
    state = make_ready_state()
    state.multi_point_mode = True
    service, _, _ = make_service(state)

    with pytest.raises(NoPointsQueued):
        service.commit_multi_point(CommitMultiPoint("appstate-test"))


@pytest.mark.parametrize(
    ("mask", "error"),
    [
        (None, SegmentationModelFailed),
        (np.zeros((2, 2), dtype=np.uint8), InvalidMaskState),
        (np.zeros((3, 4), dtype=np.float32), InvalidMaskState),
    ],
)
def test_model_mask_must_be_present_with_matching_shape_and_dtype(mask, error) -> None:
    state = make_ready_state()
    repository = RecordingStateRepository(state)
    model = RecordingSegmentationModel(mask)
    service = SegmentationService(repository, model_factory=lambda: model)

    with pytest.raises(error):
        service.select_instance_point(
            SelectInstancePoint(
                "appstate-test",
                (1, 1),
                MaskOperation.REPLACE,
                PointPolarity.POSITIVE,
            )
        )

    assert state.slice_mask is None
    assert state.segmentation_model is None
    assert repository.save_calls == []


@pytest.mark.parametrize(
    "model",
    [
        object(),
        FailingSegmentationModel(np.zeros((3, 4), dtype=np.uint8)),
        RecordingSegmentationModel(None),
        RecordingSegmentationModel(np.zeros((2, 2), dtype=np.uint8)),
        RecordingSegmentationModel(np.zeros((3, 4), dtype=np.float32)),
    ],
)
def test_new_invalid_model_is_not_cached_and_retry_calls_factory_again(model) -> None:
    state = make_ready_state()
    repository = RecordingStateRepository(state)
    valid_model = RecordingSegmentationModel(np.zeros((3, 4), dtype=np.uint8))
    factory_calls = []

    def model_factory():
        factory_calls.append(True)
        return model if len(factory_calls) == 1 else valid_model

    service = SegmentationService(repository, model_factory=model_factory)
    command = SelectInstancePoint(
        "appstate-test",
        (1, 1),
        MaskOperation.REPLACE,
        PointPolarity.POSITIVE,
    )

    with pytest.raises((SegmentationModelFailed, InvalidMaskState)):
        service.select_instance_point(command)

    assert state.segmentation_model is None
    assert state.slice_mask is None

    result = service.select_instance_point(command)

    assert result.positive_points == ((1, 1),)
    assert factory_calls == [True, True]
    assert state.segmentation_model is valid_model
    assert repository.save_calls == []


def test_existing_cached_model_remains_owned_after_inference_failure() -> None:
    state = make_ready_state()
    cached_model = FailingSegmentationModel(np.zeros((3, 4), dtype=np.uint8))
    state.segmentation_model = cached_model
    repository = RecordingStateRepository(state)
    factory = MagicMock()
    service = SegmentationService(repository, model_factory=factory)

    with pytest.raises(SegmentationModelFailed, match="failed"):
        service.select_instance_point(
            SelectInstancePoint(
                "appstate-test",
                (1, 1),
                MaskOperation.REPLACE,
                PointPolarity.POSITIVE,
            )
        )

    assert state.segmentation_model is cached_model
    assert state.slice_mask is None
    factory.assert_not_called()
    assert repository.save_calls == []
