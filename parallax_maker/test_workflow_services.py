from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
import pytest

from .controller import AppState
from .slice import ImageSlice
from .workflow_services import (
    ConfigureThresholds,
    GenerateDepth,
    GenerateSlices,
    StateSaveOptions,
    UpdateThresholdValues,
    UploadImage,
    WorkflowNotReady,
    WorkflowService,
    WorkflowUnchanged,
)


class RecordingStateRepository:
    """In-memory repository that makes command persistence observable."""

    def __init__(self, state: AppState, state_id: str = "appstate-test") -> None:
        self.state = state
        self.state_id = state_id
        self.create_count = 0
        self.loaded_ids: list[str] = []
        self.saves: list[tuple[str, AppState, StateSaveOptions]] = []

    def create(self) -> tuple[AppState, str]:
        self.create_count += 1
        self.state.filename = self.state_id
        return self.state, self.state_id

    def load(self, state_id: str) -> AppState:
        assert state_id == self.state_id
        self.loaded_ids.append(state_id)
        return self.state

    def save(self, state_id: str, state: AppState, options: StateSaveOptions) -> None:
        self.saves.append((state_id, state, options))


def test_upload_image_initializes_state_without_persisting_json() -> None:
    state = AppState()
    state.depthMapData = np.full((3, 4), 99, dtype=np.uint8)
    state.image_slices = [ImageSlice(np.zeros((3, 4, 4), dtype=np.uint8), 12)]
    state.selected_slice = 0
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)
    uploaded = Image.new("RGBA", (4, 3), (12, 34, 56, 128))

    result = service.upload_image(UploadImage(uploaded))

    assert result.state_id == "appstate-test"
    assert repository.create_count == 1
    assert repository.saves == []
    assert state.imgData.mode == "RGB"
    assert state.imgData.size == (4, 3)
    assert state.imgData.getpixel((0, 0)) == (12, 34, 56)
    assert state.depthMapData is None
    assert state.image_slices == []
    assert state.selected_slice is None


@pytest.mark.parametrize("image", [None, np.zeros((3, 4, 3), dtype=np.uint8)])
def test_upload_image_requires_a_pil_image_before_creating_state(image) -> None:
    state = AppState()
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowNotReady, match="input image"):
        service.upload_image(UploadImage(image))

    assert repository.create_count == 0
    assert repository.loaded_ids == []
    assert repository.saves == []


@dataclass(frozen=True)
class StubDepthModel:
    model_name: str


def test_generate_depth_injects_model_and_generator_resets_thresholds() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.imgData = Image.new("RGBA", (4, 3), (20, 40, 60, 128))
    state.imgThresholds = [0, 64, 128, 192, 255]
    repository = RecordingStateRepository(state)
    factory_calls: list[str] = []
    generator_calls: list[tuple[np.ndarray, StubDepthModel, object]] = []
    progress_reporter = object()
    expected_depth = np.array(
        [[0, 20, 40, 60], [80, 100, 120, 140], [160, 180, 220, 255]],
        dtype=np.uint8,
    )

    def model_factory(model: str) -> StubDepthModel:
        factory_calls.append(model)
        return StubDepthModel(model)

    def depth_generator(image, model, progress_callback=None):
        generator_calls.append((image.copy(), model, progress_callback))
        return expected_depth.copy()

    service = WorkflowService(
        repository,
        depth_model_factory=model_factory,
        depth_generator=depth_generator,
        progress_reporter=progress_reporter,
    )

    result = service.generate_depth(GenerateDepth("appstate-test", "dinov2"))

    assert result.state_id == "appstate-test"
    assert factory_calls == ["dinov2"]
    assert len(generator_calls) == 1
    generated_from, model, progress = generator_calls[0]
    assert generated_from.shape == (3, 4, 3)
    assert tuple(generated_from[0, 0]) == (20, 40, 60)
    assert model == StubDepthModel("dinov2")
    assert progress is progress_reporter
    assert state.depth_estimation_model == model
    assert np.array_equal(state.depthMapData, expected_depth)
    assert state.imgThresholds is None
    assert len(repository.saves) == 1
    saved_id, saved_state, options = repository.saves[0]
    assert saved_id == "appstate-test"
    assert saved_state is state
    assert options.save_image_slices is False
    assert options.save_depth_map is True
    assert options.save_input_image is False


def test_generate_depth_reuses_equal_model_already_held_by_state() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.imgData = Image.new("RGB", (2, 2), "navy")
    existing_model = StubDepthModel("dinov2")
    state.depth_estimation_model = existing_model
    repository = RecordingStateRepository(state)
    received_models = []

    def depth_generator(image, model, progress_callback=None):
        del image, progress_callback
        received_models.append(model)
        return np.array([[0, 85], [170, 255]], dtype=np.uint8)

    service = WorkflowService(
        repository,
        depth_model_factory=lambda model: StubDepthModel(model),
        depth_generator=depth_generator,
    )

    service.generate_depth(GenerateDepth("appstate-test", "dinov2"))

    assert state.depth_estimation_model is existing_model
    assert received_models == [existing_model]


def test_generate_depth_requires_an_input_image() -> None:
    state = AppState()
    state.filename = "appstate-test"
    repository = RecordingStateRepository(state)

    def unexpected_dependency(*args, **kwargs):
        raise AssertionError("depth dependencies must not run without an input image")

    service = WorkflowService(
        repository,
        depth_model_factory=unexpected_dependency,
        depth_generator=unexpected_dependency,
    )

    with pytest.raises(WorkflowNotReady, match="input image"):
        service.generate_depth(GenerateDepth("appstate-test", "dinov2"))

    assert repository.saves == []


def test_configure_thresholds_uses_injected_analyzer_without_saving() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.depthMapData = np.arange(24, dtype=np.uint8).reshape(4, 6)
    repository = RecordingStateRepository(state)
    analyzer_calls: list[tuple[np.ndarray, int]] = []

    def threshold_analyzer(depth_map, num_slices):
        analyzer_calls.append((depth_map.copy(), num_slices))
        return [0, 50, 120, 200, 255]

    service = WorkflowService(repository, threshold_analyzer=threshold_analyzer)

    result = service.configure_thresholds(
        ConfigureThresholds("appstate-test", num_slices=4)
    )

    assert result.state_id == "appstate-test"
    assert result.thresholds == [0, 50, 120, 200, 255]
    assert result.missing_depth is False
    assert state.num_slices == 4
    assert state.imgThresholds == result.thresholds
    assert len(analyzer_calls) == 1
    assert np.array_equal(analyzer_calls[0][0], state.depthMapData)
    assert analyzer_calls[0][1] == 4
    assert repository.saves == []


def test_configure_thresholds_without_depth_uses_uniform_fallback() -> None:
    state = AppState()
    state.filename = "appstate-test"
    repository = RecordingStateRepository(state)

    def unexpected_analyzer(*args, **kwargs):
        raise AssertionError("threshold analyzer must not run without a depth map")

    service = WorkflowService(repository, threshold_analyzer=unexpected_analyzer)

    result = service.configure_thresholds(
        ConfigureThresholds("appstate-test", num_slices=4)
    )

    assert result.thresholds == [0, 63, 127, 191, 255]
    assert result.missing_depth is True
    assert state.imgThresholds == result.thresholds
    assert repository.saves == []


def test_configure_thresholds_recomputes_stale_boundaries_for_new_slice_count() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgThresholds = [0, 80, 160, 255]
    state.depthMapData = np.arange(24, dtype=np.uint8).reshape(4, 6)
    repository = RecordingStateRepository(state)
    analyzer_calls = []

    def threshold_analyzer(depth_map, num_slices):
        analyzer_calls.append((depth_map.copy(), num_slices))
        return [0, 50, 100, 150, 255]

    service = WorkflowService(repository, threshold_analyzer=threshold_analyzer)

    result = service.configure_thresholds(
        ConfigureThresholds("appstate-test", num_slices=4)
    )

    assert result.thresholds == [0, 50, 100, 150, 255]
    assert state.num_slices == 4
    assert state.imgThresholds == result.thresholds
    assert len(analyzer_calls) == 1
    assert analyzer_calls[0][1] == 4
    assert repository.saves == []


def test_configure_thresholds_signals_when_state_would_not_change() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgThresholds = [0, 80, 160, 255]
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowUnchanged, match="unchanged"):
        service.configure_thresholds(ConfigureThresholds("appstate-test", num_slices=3))

    assert state.imgThresholds == [0, 80, 160, 255]
    assert repository.saves == []


@pytest.mark.parametrize("num_slices", [0, -1])
def test_configure_thresholds_requires_a_positive_slice_count(num_slices) -> None:
    state = AppState()
    state.filename = "appstate-test"
    original_num_slices = state.num_slices
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowNotReady, match="positive"):
        service.configure_thresholds(
            ConfigureThresholds("appstate-test", num_slices=num_slices)
        )

    assert state.num_slices == original_num_slices
    assert state.imgThresholds is None
    assert repository.saves == []


def test_update_threshold_values_normalizes_and_only_mutates_cached_state() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.imgThresholds = [0, 10, 20, 30, 40, 255]
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    result = service.update_threshold_values(
        UpdateThresholdValues("appstate-test", values=[0, 40, 35, 255], num_slices=5)
    )

    assert result.state_id == "appstate-test"
    assert result.values == [1, 40, 41, 254]
    assert result.preview_image is None
    assert state.imgThresholds == [0, 1, 40, 41, 254, 255]
    assert repository.saves == []


def test_update_threshold_values_recomputes_selected_mask_preview() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgData = Image.new("RGB", (4, 3), (120, 80, 40))
    state.depthMapData = np.array(
        [[10, 60, 130, 220], [20, 70, 140, 230], [30, 80, 150, 240]],
        dtype=np.uint8,
    )
    state.imgThresholds = [0, 100, 180, 255]
    state.slice_pixel = (2, 1)
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    result = service.update_threshold_values(
        UpdateThresholdValues("appstate-test", values=[90, 160], num_slices=3)
    )

    assert result.values == [90, 160]
    assert isinstance(result.preview_image, Image.Image)
    assert result.preview_image.size == state.imgData.size
    assert state.slice_mask.shape == state.depthMapData.shape
    assert state.slice_mask[1, 2] == 255
    assert state.slice_mask[0, 0] == 0
    assert repository.saves == []


def test_update_threshold_values_signals_when_state_would_not_change() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgThresholds = [0, 90, 160, 255]
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowUnchanged, match="unchanged"):
        service.update_threshold_values(
            UpdateThresholdValues("appstate-test", values=[90, 160], num_slices=3)
        )

    assert repository.saves == []


@pytest.mark.parametrize("thresholds", [None, []])
def test_update_threshold_values_requires_configured_thresholds(thresholds) -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgThresholds = thresholds
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowNotReady, match="threshold"):
        service.update_threshold_values(
            UpdateThresholdValues("appstate-test", values=[90, 160], num_slices=3)
        )

    assert state.imgThresholds == thresholds
    assert repository.saves == []


@pytest.mark.parametrize("values", [None, [], [90], [70, 110, 160]])
def test_update_threshold_values_requires_one_edit_per_interior_boundary(
    values,
) -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgThresholds = [0, 90, 160, 255]
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowNotReady, match="threshold"):
        service.update_threshold_values(
            UpdateThresholdValues("appstate-test", values=values, num_slices=3)
        )

    assert state.imgThresholds == [0, 90, 160, 255]
    assert repository.saves == []


def test_update_threshold_values_requires_matching_state_slice_count() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 4
    state.imgThresholds = [0, 90, 160, 255]
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowNotReady, match="threshold"):
        service.update_threshold_values(
            UpdateThresholdValues("appstate-test", values=[70, 110, 160], num_slices=4)
        )

    assert state.imgThresholds == [0, 90, 160, 255]
    assert repository.saves == []


@pytest.mark.parametrize(
    ("image", "depth_map", "slice_pixel", "message"),
    [
        (None, np.arange(12, dtype=np.uint8).reshape(3, 4), (2, 1), "input image"),
        (Image.new("RGB", (4, 3)), None, (2, 1), "depth map"),
        (
            Image.new("RGB", (4, 3)),
            np.arange(12, dtype=np.uint8).reshape(3, 4),
            (20, 10),
            "selected pixel",
        ),
    ],
)
def test_update_threshold_values_validates_preview_prerequisites(
    image, depth_map, slice_pixel, message
) -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgData = image
    state.depthMapData = depth_map
    state.slice_pixel = slice_pixel
    state.imgThresholds = [0, 90, 160, 255]
    repository = RecordingStateRepository(state)
    service = WorkflowService(repository)

    with pytest.raises(WorkflowNotReady, match=message):
        service.update_threshold_values(
            UpdateThresholdValues("appstate-test", values=[80, 150], num_slices=3)
        )

    assert state.imgThresholds == [0, 90, 160, 255]
    assert repository.saves == []


def test_generate_slices_injects_generator_and_persists_all_artifacts() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.imgData = Image.new("RGB", (5, 4), (11, 22, 33))
    state.depthMapData = np.arange(20, dtype=np.uint8).reshape(4, 5)
    state.num_slices = 3
    state.imgThresholds = [0, 10, 20, 255]
    repository = RecordingStateRepository(state)
    generator_calls = []
    expected_slices = [
        ImageSlice(np.full((4, 5, 4), index, dtype=np.uint8), depth)
        for index, depth in enumerate((10, 20, 255), start=1)
    ]

    def slice_generator(image, depth_map, thresholds, num_expand):
        generator_calls.append(
            (image.copy(), depth_map.copy(), list(thresholds), num_expand)
        )
        return expected_slices

    service = WorkflowService(
        repository, slice_generator=slice_generator, slice_expand=3
    )

    result = service.generate_slices(GenerateSlices("appstate-test"))

    assert result.state_id == "appstate-test"
    assert result.slice_count == 3
    assert state.image_slices is expected_slices
    assert len(generator_calls) == 1
    image, depth_map, thresholds, num_expand = generator_calls[0]
    assert image.shape == (4, 5, 3)
    assert tuple(image[0, 0]) == (11, 22, 33)
    assert np.array_equal(depth_map, state.depthMapData)
    assert thresholds == state.imgThresholds
    assert num_expand == 3
    assert len(repository.saves) == 1
    saved_id, saved_state, options = repository.saves[0]
    assert saved_id == "appstate-test"
    assert saved_state is state
    assert options.save_image_slices is True
    assert options.save_depth_map is True
    assert options.save_input_image is True


def test_generate_slices_requires_a_depth_map_without_saving() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.imgData = Image.new("RGB", (4, 3), "black")
    repository = RecordingStateRepository(state)

    def unexpected_generator(*args, **kwargs):
        raise AssertionError("slice generator must not run without a depth map")

    service = WorkflowService(repository, slice_generator=unexpected_generator)

    with pytest.raises(WorkflowNotReady, match="depth map"):
        service.generate_slices(GenerateSlices("appstate-test"))

    assert state.image_slices == []
    assert repository.saves == []


def test_generate_slices_requires_an_input_image_without_saving() -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.num_slices = 3
    state.depthMapData = np.arange(12, dtype=np.uint8).reshape(3, 4)
    state.imgThresholds = [0, 90, 160, 255]
    repository = RecordingStateRepository(state)

    def unexpected_generator(*args, **kwargs):
        raise AssertionError("slice generator must not run without an input image")

    service = WorkflowService(repository, slice_generator=unexpected_generator)

    with pytest.raises(WorkflowNotReady, match="input image"):
        service.generate_slices(GenerateSlices("appstate-test"))

    assert state.image_slices == []
    assert repository.saves == []


@pytest.mark.parametrize("thresholds", [None, [0, 128, 255]])
def test_generate_slices_requires_complete_threshold_boundaries(thresholds) -> None:
    state = AppState()
    state.filename = "appstate-test"
    state.imgData = Image.new("RGB", (4, 3), "black")
    state.depthMapData = np.arange(12, dtype=np.uint8).reshape(3, 4)
    state.num_slices = 4
    state.imgThresholds = thresholds
    repository = RecordingStateRepository(state)

    def unexpected_generator(*args, **kwargs):
        raise AssertionError("slice generator must not run with invalid thresholds")

    service = WorkflowService(repository, slice_generator=unexpected_generator)

    with pytest.raises(WorkflowNotReady, match="threshold"):
        service.generate_slices(GenerateSlices("appstate-test"))

    assert state.image_slices == []
    assert repository.saves == []
