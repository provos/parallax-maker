"""Contract tests for the framework-neutral inpainting workflow service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from .controller import AppState
from .inpainting_services import (
    ApplyInpaintingCandidate,
    DeleteInpaintingMask,
    EraseInpainting,
    GenerateInpaintingCandidates,
    InpaintingMaskNotFound,
    InpaintingMode,
    InpaintingModelFailed,
    InpaintingNotReady,
    InpaintingService,
    InpaintingUnchanged,
    InvalidInpaintingCandidate,
    LoadInpaintingMask,
    MoveSliceVersion,
    SaveInpaintingMask,
    SelectInpaintingCandidate,
    SliceVersionDirection,
    SliceVersionUnavailable,
    UpdateInpaintingModel,
    UpdateInpaintingPrompts,
)
from .slice import ImageSlice


class MemoryStateRepository:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.loaded: list[str] = []
        self.saved: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(self, state_id: str) -> AppState:
        self.loaded.append(state_id)
        return self.state

    def save(self, *args: object, **kwargs: object) -> None:
        self.saved.append((args, kwargs))


class RecordingPipeline:
    instances: list["RecordingPipeline"] = []

    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        self.kwargs = kwargs
        self.load_calls = 0
        self.inpaint_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.fail = False
        RecordingPipeline.instances.append(self)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RecordingPipeline)
            and self.model == other.model
            and self.kwargs == other.kwargs
        )

    def load_model(self) -> object:
        self.load_calls += 1
        return object()

    def inpaint(self, *args: object, **kwargs: object) -> Image.Image:
        self.inpaint_calls.append((args, kwargs))
        if self.fail:
            raise RuntimeError("model exploded")
        input_image = args[2]
        assert isinstance(input_image, (Image.Image, np.ndarray))
        if isinstance(input_image, np.ndarray):
            height, width = input_image.shape[:2]
        else:
            width, height = input_image.size
        value = 40 + len(self.inpaint_calls)
        return Image.new("RGBA", (width, height), (value, value + 1, value + 2, 255))


def make_state(tmp_path: Path) -> AppState:
    state = AppState()
    state.filename = str(tmp_path)
    state.imgData = Image.new("RGB", (6, 4), (10, 20, 30))
    image = np.zeros((4, 6, 4), dtype=np.uint8)
    image[:, :, :3] = (12, 34, 56)
    image[:, :, 3] = 255
    image[1:3, 2:5, 3] = 80
    filename = tmp_path / "image_slice_0.png"
    Image.fromarray(image, mode="RGBA").save(filename)
    state.image_slices = [ImageSlice(image.copy(), depth=50, filename=str(filename))]
    state.selected_slice = 0
    return state


def make_service(
    state: AppState,
    *,
    patcher=None,
) -> tuple[InpaintingService, MemoryStateRepository]:
    repository = MemoryStateRepository(state)
    kwargs: dict[str, object] = {
        "state_repository": repository,
        "pipeline_factory": RecordingPipeline,
    }
    if patcher is not None:
        kwargs["patcher"] = patcher
    return InpaintingService(**kwargs), repository


def save_mask(service: InpaintingService, state_id: str = "state") -> Image.Image:
    canvas = Image.new("RGBA", (3, 2), (255, 0, 0, 0))
    canvas.putpixel((1, 1), (255, 0, 0, 255))
    service.save_mask(
        SaveInpaintingMask(
            state_id=state_id,
            canvas_image=canvas,
            padding=1,
            show_crop_region=True,
        )
    )
    return canvas


def generate_command(mode: InpaintingMode) -> GenerateInpaintingCandidates:
    return GenerateInpaintingCandidates(
        state_id="state",
        mode=mode,
        model_name="fake-model",
        workflow=None,
        positive_prompt="bright sky",
        negative_prompt="clouds",
        strength=0.63,
        guidance_scale=6.25,
        padding=7,
        blur=3,
    )


@pytest.fixture(autouse=True)
def clear_pipeline_instances() -> None:
    RecordingPipeline.instances.clear()


def test_mask_save_load_and_delete_round_trip(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    save_mask_source = Image.new("RGBA", (3, 2))
    save_mask_source.putalpha(Image.new("L", save_mask_source.size, 255))
    # Transport RGB must be ignored; only canvas alpha becomes the persisted mask.
    saved = service.save_mask(SaveInpaintingMask("state", save_mask_source, 1, True))

    mask_path = state.mask_filename(0)
    assert mask_path.exists()
    with Image.open(mask_path) as mask:
        assert mask.mode == "L"
        assert mask.size == state.imgData.size
        assert np.asarray(mask).min() == 255
    assert saved.bounding_box == (0, 0, 6, 4)

    loaded = service.load_mask(LoadInpaintingMask(state_id="state"))
    assert loaded.mask.mode == "L"
    assert loaded.mask.size == state.imgData.size
    assert np.asarray(loaded.mask).min() == 255

    deleted = service.delete_mask(DeleteInpaintingMask(state_id="state"))
    assert deleted.deleted is True
    assert not mask_path.exists()
    assert service.delete_mask(DeleteInpaintingMask(state_id="state")).deleted is False


def test_mask_commands_validate_selected_slice_and_existing_artifact(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)
    state.selected_slice = None

    with pytest.raises(InpaintingNotReady):
        save_mask(service)
    with pytest.raises(InpaintingNotReady):
        service.load_mask(LoadInpaintingMask("state"))

    state.selected_slice = 0
    with pytest.raises(InpaintingMaskNotFound):
        service.load_mask(LoadInpaintingMask("state"))


def test_prompt_and_model_updates_persist_and_model_change_invalidates_caches(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.image_slices[0].positive_prompt = "old"
    state.image_slices[0].negative_prompt = "old"
    state.pipeline_spec = object()
    state.upscaler = object()
    service, repository = make_service(state)

    service.update_prompts(UpdateInpaintingPrompts("state", None, None))
    assert state.image_slices[0].positive_prompt == ""
    assert state.image_slices[0].negative_prompt == ""

    service.update_model(UpdateInpaintingModel("state", "new-model"))
    assert state.inpainting_model_name == "new-model"
    assert state.pipeline_spec is None
    assert state.upscaler is None
    assert len(repository.saved) == 2

    with pytest.raises(InpaintingUnchanged):
        service.update_model(UpdateInpaintingModel("state", "new-model"))


def test_paint_generation_forwards_parameters_and_does_not_mutate_slice(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    original = state.image_slices[0].image.copy()
    patch_inputs: list[np.ndarray] = []

    def recording_patcher(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        patch_inputs.append(image)
        image[0, 0] = (200, 201, 202, 203)
        assert mask.shape == image.shape[:2]
        return image

    service, _ = make_service(state, patcher=recording_patcher)
    save_mask(service)
    state.selected_inpainting = 2
    result = service.generate_candidates(generate_command(InpaintingMode.PAINT))

    assert len(result.candidates) == 3
    assert state.selected_inpainting is None
    assert np.array_equal(state.image_slices[0].image, original)
    assert patch_inputs and patch_inputs[0] is not state.image_slices[0].image
    pipeline = RecordingPipeline.instances[-1]
    assert pipeline.load_calls == 1
    assert len(pipeline.inpaint_calls) == 3
    args, kwargs = pipeline.inpaint_calls[0]
    assert args[0:2] == ("bright sky", "clouds")
    assert kwargs == {
        "strength": 0.63,
        "guidance_scale": 6.25,
        "blur_radius": 3,
        "padding": 7,
        "crop": True,
    }
    assert state.image_slices[0].positive_prompt == "bright sky"
    assert state.image_slices[0].negative_prompt == "clouds"


def test_fill_generation_uses_inverse_alpha_and_returns_three_candidates(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    observed_masks: list[np.ndarray] = []

    def recording_patcher(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observed_masks.append(mask.copy())
        return image

    service, _ = make_service(state, patcher=recording_patcher)
    expected = 255 - state.image_slices[0].image[:, :, 3]

    result = service.generate_candidates(generate_command(InpaintingMode.FILL))

    assert len(result.candidates) == 3
    assert len(observed_masks) == 1
    assert np.array_equal(observed_masks[0], expected)
    assert np.array_equal(
        np.asarray(RecordingPipeline.instances[-1].inpaint_calls[0][0][3]), expected
    )


def test_enhance_returns_two_candidates_and_restores_original_alpha(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)
    source_alpha = state.image_slices[0].image[:, :, 3].copy()
    upscaler_calls: list[tuple[str, str]] = []

    def fake_upscale(
        _state: AppState, image: np.ndarray, prompt: str, negative_prompt: str
    ) -> Image.Image:
        upscaler_calls.append((prompt, negative_prompt))
        height, width = image.shape[:2]
        return Image.new("RGBA", (width * 2, height * 2), (90, 91, 92, 0))

    with patch.object(AppState, "upscale_image", fake_upscale):
        result = service.generate_candidates(generate_command(InpaintingMode.ENHANCE))

    assert len(result.candidates) == 2
    assert upscaler_calls == [("bright sky", "clouds")] * 2
    for candidate in result.candidates:
        assert candidate.size == (6, 4)
        assert np.array_equal(np.asarray(candidate)[:, :, 3], source_alpha)
    assert RecordingPipeline.instances[-1].load_calls == 1


def test_generation_reuses_equal_pipeline_and_wraps_model_failures(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state, patcher=lambda image, mask: image)
    save_mask(service)

    service.generate_candidates(generate_command(InpaintingMode.PAINT))
    first = state.pipeline_spec
    service.generate_candidates(generate_command(InpaintingMode.PAINT))
    assert state.pipeline_spec is first
    assert getattr(first, "load_calls") == 1

    state.selected_inpainting = 1
    first.fail = True
    with pytest.raises(InpaintingModelFailed) as exc_info:
        service.generate_candidates(generate_command(InpaintingMode.PAINT))
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "model exploded"
    assert state.selected_inpainting == 1


def test_non_comfyui_workflow_changes_do_not_reload_the_pipeline(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state, patcher=lambda image, mask: image)
    save_mask(service)
    command = generate_command(InpaintingMode.PAINT)

    service.generate_candidates(replace(command, workflow=b"first workflow"))
    first = state.pipeline_spec
    service.generate_candidates(replace(command, workflow=b"second workflow"))

    assert state.pipeline_spec is first
    assert getattr(first, "load_calls") == 1


def test_comfyui_workflow_is_saved_and_supplied_to_the_pipeline(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state, patcher=lambda image, mask: image)
    save_mask(service)
    command = replace(
        generate_command(InpaintingMode.PAINT),
        model_name="comfyui",
        workflow=b'{"prompt": {}}',
    )

    service.generate_candidates(command)

    workflow_path = state.workflow_path()
    assert workflow_path.read_bytes() == b'{"prompt": {}}'
    pipeline = RecordingPipeline.instances[-1]
    assert pipeline.kwargs["workflow_path"] == workflow_path


def test_invalid_mode_and_pipeline_load_failure_are_domain_errors(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)
    invalid = replace(generate_command(InpaintingMode.PAINT), mode="paint")
    with pytest.raises(InpaintingNotReady, match="mode"):
        service.generate_candidates(invalid)

    def failing_factory(*args: object, **kwargs: object) -> RecordingPipeline:
        del args, kwargs
        raise RuntimeError("cannot load")

    repository = MemoryStateRepository(state)
    failing_service = InpaintingService(
        state_repository=repository,
        pipeline_factory=failing_factory,
    )
    with pytest.raises(InpaintingModelFailed) as exc_info:
        failing_service.generate_candidates(generate_command(InpaintingMode.ENHANCE))
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "cannot load"


def test_candidate_selection_toggles_and_generation_resets_it(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state, patcher=lambda image, mask: image)
    save_mask(service)

    selected = service.select_candidate(SelectInpaintingCandidate("state", 1, 3))
    assert selected.selected_index == 1
    assert state.selected_inpainting == 1
    deselected = service.select_candidate(SelectInpaintingCandidate("state", 1, 3))
    assert deselected.selected_index is None
    assert state.selected_inpainting is None

    with pytest.raises(InvalidInpaintingCandidate):
        service.select_candidate(SelectInpaintingCandidate("state", 3, 3))

    state.selected_inpainting = 2
    service.generate_candidates(generate_command(InpaintingMode.PAINT))
    assert state.selected_inpainting is None


def test_apply_requires_valid_selection_versions_image_and_saves_json_only(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    service, repository = make_service(state)
    candidates = [Image.new("RGBA", (6, 4), (index, 2, 3, 255)) for index in range(2)]

    with pytest.raises(InvalidInpaintingCandidate):
        service.apply_candidate(ApplyInpaintingCandidate("state", candidates))

    state.selected_inpainting = 4
    with pytest.raises(InvalidInpaintingCandidate):
        service.apply_candidate(ApplyInpaintingCandidate("state", candidates))

    state.selected_inpainting = 1
    result = service.apply_candidate(ApplyInpaintingCandidate("state", candidates))
    assert np.array_equal(state.image_slices[0].image, np.asarray(candidates[1]))
    assert Path(result.image_filename).exists()
    assert state.selected_inpainting is None
    assert len(repository.saved) == 1
    save_args, save_kwargs = repository.saved[0]
    options = save_kwargs.get("options", save_args[2])
    assert options.save_image_slices is False
    assert options.save_depth_map is False
    assert options.save_input_image is False


def test_erase_versions_slice_and_missing_mask_is_domain_error(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    original_alpha = state.image_slices[0].image[:, :, 3].copy()
    service, repository = make_service(state)

    with pytest.raises(InpaintingMaskNotFound):
        service.erase(EraseInpainting("state"))

    save_mask(service)
    result = service.erase(EraseInpainting("state"))
    assert Path(result.image_filename).exists()
    assert np.any(state.image_slices[0].image[:, :, 3] < original_alpha)
    assert len(repository.saved) == 1


def test_undo_and_redo_move_slice_versions_and_persist_mapping(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.image_slices[0].new_version(np.full((4, 6, 4), 77, dtype=np.uint8))
    service, repository = make_service(state)

    backward = service.move_slice_version(
        MoveSliceVersion("state", 0, SliceVersionDirection.BACKWARD)
    )
    assert backward.slice_index == 0
    assert np.any(state.image_slices[0].image != 77)
    service.move_slice_version(
        MoveSliceVersion("state", 0, SliceVersionDirection.FORWARD)
    )
    assert np.all(state.image_slices[0].image == 77)
    assert len(repository.saved) == 2

    with pytest.raises(SliceVersionUnavailable):
        service.move_slice_version(
            MoveSliceVersion("state", 0, SliceVersionDirection.FORWARD)
        )
