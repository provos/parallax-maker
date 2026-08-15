"""Framework-neutral services for the interactive inpainting workflow.

UI adapters own transport decoding/encoding, component rendering, callback
control flow, and log messages.  This module owns inpainting state mutation,
artifact persistence, model lifecycle, candidate generation, and slice
versioning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
from PIL import Image

from .controller import AppState
from .inpainting import InpaintingModel, patch_image
from .segmentation import remove_mask_from_alpha
from .utils import find_square_bounding_box
from .workflow_services import StateSaveOptions


class InpaintingMode(Enum):
    PAINT = "paint"
    FILL = "fill"
    ENHANCE = "enhance"


class SliceVersionDirection(Enum):
    BACKWARD = "backward"
    FORWARD = "forward"


class InpaintingServiceError(Exception):
    """Base class for inpainting-domain failures."""


class InpaintingNotReady(InpaintingServiceError):
    """Required workflow state is missing or invalid."""


class InpaintingMaskNotFound(InpaintingServiceError):
    """The selected slice has no saved inpainting mask."""


class InvalidInpaintingCandidate(InpaintingServiceError):
    """A candidate selection or candidate image is invalid."""


class InpaintingModelFailed(InpaintingServiceError):
    """An inpainting model could not be created, loaded, or executed."""


class InpaintingUnchanged(InpaintingServiceError):
    """The requested mutation would not change inpainting state."""


class SliceVersionUnavailable(InpaintingServiceError):
    """The requested backward or forward slice version does not exist."""


class InpaintingStateRepository(Protocol):
    def load(self, state_id: str) -> AppState: ...

    def save(
        self, state_id: str, state: AppState, options: StateSaveOptions
    ) -> None: ...


class CachedInpaintingStateRepository:
    """Adapt the existing AppState cache and JSON persistence API."""

    def load(self, state_id: str) -> AppState:
        return AppState.from_cache(state_id)

    def save(self, state_id: str, state: AppState, options: StateSaveOptions) -> None:
        state.to_file(
            state_id,
            save_image_slices=options.save_image_slices,
            save_depth_map=options.save_depth_map,
            save_input_image=options.save_input_image,
        )


class InpaintingArtifactRepository(Protocol):
    def save_mask(
        self, state: AppState, slice_index: int, mask: Image.Image
    ) -> str: ...

    def load_mask(self, state: AppState, slice_index: int) -> Image.Image: ...

    def delete_mask(self, state: AppState, slice_index: int) -> bool: ...

    def save_workflow(self, state: AppState, workflow: bytes) -> Path: ...


class FilesystemInpaintingArtifactRepository:
    """Store masks and ComfyUI workflows in the existing state directory."""

    def save_mask(self, state: AppState, slice_index: int, mask: Image.Image) -> str:
        return state.save_image_mask(slice_index, mask)

    def load_mask(self, state: AppState, slice_index: int) -> Image.Image:
        path = state.mask_filename(slice_index)
        if not Path(path).exists():
            raise InpaintingMaskNotFound(f"no mask found for slice {slice_index}")
        with Image.open(path) as image:
            return image.convert("L").copy()

    def delete_mask(self, state: AppState, slice_index: int) -> bool:
        path = Path(state.mask_filename(slice_index))
        if not path.exists():
            return False
        path.unlink()
        return True

    def save_workflow(self, state: AppState, workflow: bytes) -> Path:
        path = state.workflow_path()
        if not path.exists() or path.read_bytes() != workflow:
            path.write_bytes(workflow)
        return path


class InpaintingPipelineLike(Protocol):
    def load_model(self) -> object: ...

    def inpaint(
        self,
        prompt: str,
        negative_prompt: str,
        init_image: object,
        mask_image: object,
        **kwargs: object,
    ) -> object: ...


@dataclass(frozen=True)
class SaveInpaintingMask:
    state_id: str
    canvas_image: Image.Image
    padding: int
    show_crop_region: bool


@dataclass(frozen=True)
class SavedInpaintingMaskResult:
    state_id: str
    slice_index: int
    mask_filename: str
    bounding_box: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class DeleteInpaintingMask:
    state_id: str


@dataclass(frozen=True)
class DeletedInpaintingMaskResult:
    state_id: str
    slice_index: int
    deleted: bool


@dataclass(frozen=True)
class LoadInpaintingMask:
    state_id: str


@dataclass(frozen=True)
class LoadedInpaintingMaskResult:
    state_id: str
    slice_index: int
    mask: Image.Image


@dataclass(frozen=True)
class UpdateInpaintingPrompts:
    state_id: str
    positive_prompt: str | None
    negative_prompt: str | None


@dataclass(frozen=True)
class UpdatedInpaintingPromptsResult:
    state_id: str
    slice_index: int
    positive_prompt: str
    negative_prompt: str


@dataclass(frozen=True)
class UpdateInpaintingModel:
    state_id: str
    model_name: str


@dataclass(frozen=True)
class UpdatedInpaintingModelResult:
    state_id: str
    model_name: str


@dataclass(frozen=True)
class GenerateInpaintingCandidates:
    state_id: str
    mode: InpaintingMode
    model_name: str
    workflow: bytes | None
    positive_prompt: str | None
    negative_prompt: str | None
    strength: float
    guidance_scale: float
    padding: int
    blur: int


@dataclass(frozen=True)
class GeneratedInpaintingCandidatesResult:
    state_id: str
    slice_index: int
    mode: InpaintingMode
    candidates: tuple[Image.Image, ...]


@dataclass(frozen=True)
class SelectInpaintingCandidate:
    state_id: str
    candidate_index: int
    candidate_count: int


@dataclass(frozen=True)
class SelectedInpaintingCandidateResult:
    state_id: str
    selected_index: int | None


@dataclass(frozen=True)
class ClearInpaintingSelection:
    state_id: str


@dataclass(frozen=True)
class ClearedInpaintingSelectionResult:
    state_id: str
    previously_selected: int | None


@dataclass(frozen=True)
class ApplyInpaintingCandidate:
    state_id: str
    candidates: Sequence[Image.Image]


@dataclass(frozen=True)
class AppliedInpaintingCandidateResult:
    state_id: str
    slice_index: int
    image_filename: str


@dataclass(frozen=True)
class EraseInpainting:
    state_id: str


@dataclass(frozen=True)
class ErasedInpaintingResult:
    state_id: str
    slice_index: int
    image_filename: str


@dataclass(frozen=True)
class MoveSliceVersion:
    state_id: str
    slice_index: int
    direction: SliceVersionDirection


@dataclass(frozen=True)
class MovedSliceVersionResult:
    state_id: str
    slice_index: int
    image_filename: str
    direction: SliceVersionDirection


class InpaintingService:
    """Orchestrate inpainting without depending on a UI framework."""

    JSON_ONLY = StateSaveOptions(
        save_image_slices=False,
        save_depth_map=False,
        save_input_image=False,
    )

    def __init__(
        self,
        state_repository: InpaintingStateRepository | None = None,
        artifact_repository: InpaintingArtifactRepository | None = None,
        *,
        pipeline_factory: Callable[..., InpaintingPipelineLike] = InpaintingModel,
        patcher: Callable[[np.ndarray, np.ndarray], np.ndarray] = patch_image,
        bounding_box_finder: Callable[..., tuple[int, int, int, int]] = (
            find_square_bounding_box
        ),
    ) -> None:
        self._states = state_repository or CachedInpaintingStateRepository()
        self._artifacts = (
            artifact_repository or FilesystemInpaintingArtifactRepository()
        )
        self._pipeline_factory = pipeline_factory
        self._patcher = patcher
        self._bounding_box_finder = bounding_box_finder

    def save_mask(self, command: SaveInpaintingMask) -> SavedInpaintingMaskResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        source = self._source_image(state)
        if not isinstance(command.canvas_image, Image.Image):
            raise InpaintingNotReady("the canvas mask must be a PIL image")
        if "A" not in command.canvas_image.getbands():
            raise InpaintingNotReady("the canvas mask must have an alpha channel")
        padding = self._nonnegative_integer(command.padding, "mask padding")
        if not isinstance(command.show_crop_region, bool):
            raise InpaintingNotReady("the crop-region flag must be a boolean")
        mask = command.canvas_image.getchannel("A").resize(
            source.size, resample=Image.Resampling.BICUBIC
        )
        bounding_box = None
        if command.show_crop_region:
            bounding_box = tuple(self._bounding_box_finder(mask, padding=padding))
            if len(bounding_box) != 4:
                raise InpaintingNotReady("the inpainting mask bounds are invalid")
        filename = self._artifacts.save_mask(state, index, mask)
        return SavedInpaintingMaskResult(
            state_id=command.state_id,
            slice_index=index,
            mask_filename=filename,
            bounding_box=bounding_box,
        )

    def delete_mask(self, command: DeleteInpaintingMask) -> DeletedInpaintingMaskResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        return DeletedInpaintingMaskResult(
            state_id=command.state_id,
            slice_index=index,
            deleted=self._artifacts.delete_mask(state, index),
        )

    def load_mask(self, command: LoadInpaintingMask) -> LoadedInpaintingMaskResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        mask = self._load_mask(state, index)
        return LoadedInpaintingMaskResult(command.state_id, index, mask)

    def update_prompts(
        self, command: UpdateInpaintingPrompts
    ) -> UpdatedInpaintingPromptsResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        positive, negative = self._prompts(command)
        image_slice = state.image_slices[index]
        if (
            image_slice.positive_prompt == positive
            and image_slice.negative_prompt == negative
        ):
            raise InpaintingUnchanged("inpainting prompts are unchanged")
        image_slice.positive_prompt = positive
        image_slice.negative_prompt = negative
        self._states.save(command.state_id, state, self.JSON_ONLY)
        return UpdatedInpaintingPromptsResult(
            command.state_id, index, positive, negative
        )

    def update_model(
        self, command: UpdateInpaintingModel
    ) -> UpdatedInpaintingModelResult:
        state = self._states.load(command.state_id)
        if not isinstance(command.model_name, str) or not command.model_name:
            raise InpaintingNotReady("an inpainting model is required")
        if state.inpainting_model_name == command.model_name:
            raise InpaintingUnchanged("the inpainting model is unchanged")
        state.inpainting_model_name = command.model_name
        state.pipeline_spec = None
        state.inpainting_pipeline_cache_identity = None
        state.upscaler = None
        state.selected_inpainting = None
        self._states.save(command.state_id, state, self.JSON_ONLY)
        return UpdatedInpaintingModelResult(command.state_id, command.model_name)

    def generate_candidates(
        self, command: GenerateInpaintingCandidates
    ) -> GeneratedInpaintingCandidatesResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        mode = self._mode(command.mode)
        image = self._slice_image(state, index).copy()
        positive, negative = self._prompts(command)
        self._validate_generation_parameters(command)

        # Prompt persistence intentionally precedes model work and survives a
        # later inference failure, matching the callback's established behavior.
        state.image_slices[index].positive_prompt = positive
        state.image_slices[index].negative_prompt = negative
        self._states.save(command.state_id, state, self.JSON_ONLY)

        pipeline = self._resolve_pipeline(state, command)
        try:
            if mode in (InpaintingMode.PAINT, InpaintingMode.FILL):
                mask = self._generation_mask(state, index, image, mode)
                patched = self._patcher(image.copy(), mask.copy())
                candidates = tuple(
                    self._generated_image(
                        pipeline.inpaint(
                            positive,
                            negative,
                            patched.copy(),
                            mask.copy(),
                            strength=command.strength,
                            guidance_scale=command.guidance_scale,
                            blur_radius=command.blur,
                            padding=command.padding,
                            crop=True,
                        ),
                        image.shape,
                    )
                    for _ in range(3)
                )
            else:
                candidates = tuple(
                    self._enhance(state, image.copy(), positive, negative)
                    for _ in range(2)
                )
        except InpaintingServiceError:
            raise
        except Exception as error:
            raise InpaintingModelFailed(
                "inpainting candidate generation failed"
            ) from error

        # Only replace the old selection after new candidates exist. If model
        # work fails, the UI can continue displaying and applying its old set.
        state.selected_inpainting = None
        return GeneratedInpaintingCandidatesResult(
            command.state_id, index, mode, candidates
        )

    def select_candidate(
        self, command: SelectInpaintingCandidate
    ) -> SelectedInpaintingCandidateResult:
        state = self._states.load(command.state_id)
        self._selected_slice(state)
        if (
            isinstance(command.candidate_index, bool)
            or not isinstance(command.candidate_index, int)
            or isinstance(command.candidate_count, bool)
            or not isinstance(command.candidate_count, int)
            or command.candidate_count <= 0
            or command.candidate_index < 0
            or command.candidate_index >= command.candidate_count
        ):
            raise InvalidInpaintingCandidate("the candidate index is invalid")
        if state.selected_inpainting == command.candidate_index:
            state.selected_inpainting = None
        else:
            state.selected_inpainting = command.candidate_index
        return SelectedInpaintingCandidateResult(
            command.state_id, state.selected_inpainting
        )

    def clear_selection(
        self, command: ClearInpaintingSelection
    ) -> ClearedInpaintingSelectionResult:
        state = self._states.load(command.state_id)
        previous = state.selected_inpainting
        state.selected_inpainting = None
        return ClearedInpaintingSelectionResult(command.state_id, previous)

    def apply_candidate(
        self, command: ApplyInpaintingCandidate
    ) -> AppliedInpaintingCandidateResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        selected = state.selected_inpainting
        try:
            candidate_count = len(command.candidates)
        except TypeError:
            raise InvalidInpaintingCandidate(
                "inpainting candidates must be a sequence"
            ) from None
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 0
            or selected >= candidate_count
        ):
            raise InvalidInpaintingCandidate(
                "no valid inpainting candidate is selected"
            )
        image = self._candidate_image(command.candidates[selected], state, index)
        filename = state.image_slices[index].new_version(np.array(image))
        state.selected_inpainting = None
        self._states.save(command.state_id, state, self.JSON_ONLY)
        return AppliedInpaintingCandidateResult(command.state_id, index, filename)

    def erase(self, command: EraseInpainting) -> ErasedInpaintingResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        image = self._slice_image(state, index).copy()
        mask = np.asarray(self._load_mask(state, index))
        if mask.shape != image.shape[:2]:
            raise InpaintingNotReady("the inpainting mask must match the slice")
        image[:, :, 3] = remove_mask_from_alpha(image, mask)
        filename = state.image_slices[index].new_version(image)
        state.selected_inpainting = None
        self._states.save(command.state_id, state, self.JSON_ONLY)
        return ErasedInpaintingResult(command.state_id, index, filename)

    def move_slice_version(self, command: MoveSliceVersion) -> MovedSliceVersionResult:
        state = self._states.load(command.state_id)
        index = self._slice_index(state, command.slice_index)
        if not isinstance(command.direction, SliceVersionDirection):
            raise InpaintingNotReady("the slice version direction is invalid")
        forward = command.direction is SliceVersionDirection.FORWARD
        if not state.image_slices[index].undo(forward=forward):
            raise SliceVersionUnavailable("the requested slice version is unavailable")
        state.selected_inpainting = None
        self._states.save(command.state_id, state, self.JSON_ONLY)
        return MovedSliceVersionResult(
            command.state_id,
            index,
            str(state.image_slices[index].filename),
            command.direction,
        )

    def _resolve_pipeline(
        self, state: AppState, command: GenerateInpaintingCandidates
    ) -> InpaintingPipelineLike:
        workflow_path = None
        configuration = self._pipeline_configuration(state, command)
        try:
            if command.model_name == "comfyui" and command.workflow is not None:
                workflow_path = self._artifacts.save_workflow(state, command.workflow)
            candidate = self._pipeline_factory(
                command.model_name,
                server_address=state.server_address,
                workflow_path=workflow_path,
                api_key=state.api_key,
            )
            if not callable(getattr(candidate, "load_model", None)) or not callable(
                getattr(candidate, "inpaint", None)
            ):
                raise InpaintingModelFailed("the inpainting pipeline is invalid")
            if not self._can_reuse_pipeline(state, configuration):
                candidate.load_model()
                state.pipeline_spec = candidate
                state.inpainting_pipeline_cache_identity = (
                    candidate,
                    configuration,
                )
                state.upscaler = None
            pipeline = state.pipeline_spec
        except InpaintingServiceError:
            raise
        except Exception as error:
            raise InpaintingModelFailed(
                "the inpainting model could not be loaded"
            ) from error
        if pipeline is None or not callable(getattr(pipeline, "inpaint", None)):
            raise InpaintingModelFailed("the inpainting pipeline is invalid")
        return pipeline

    @staticmethod
    def _pipeline_configuration(
        state: AppState, command: GenerateInpaintingCandidates
    ) -> tuple[object, ...]:
        workflow_digest = (
            hashlib.sha256(command.workflow).digest()
            if command.model_name == "comfyui" and command.workflow is not None
            else None
        )
        return (
            command.model_name,
            state.server_address,
            state.api_key,
            workflow_digest,
        )

    @staticmethod
    def _can_reuse_pipeline(
        state: AppState,
        configuration: tuple[object, ...],
    ) -> bool:
        identity = state.inpainting_pipeline_cache_identity
        if identity is None:
            return False
        cached_pipeline, cached_configuration = identity
        return (
            cached_pipeline is state.pipeline_spec
            and cached_configuration == configuration
        )

    def _generation_mask(
        self,
        state: AppState,
        index: int,
        image: np.ndarray,
        mode: InpaintingMode,
    ) -> np.ndarray:
        if mode is InpaintingMode.PAINT:
            mask = np.asarray(self._load_mask(state, index))
        else:
            mask = 255 - image[:, :, 3]
        if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
            raise InpaintingNotReady("the inpainting mask must match the slice")
        return mask

    @staticmethod
    def _enhance(
        state: AppState,
        image: np.ndarray,
        positive_prompt: str,
        negative_prompt: str,
    ) -> Image.Image:
        upscaled = state.upscale_image(
            image, prompt=positive_prompt, negative_prompt=negative_prompt
        )
        if not isinstance(upscaled, Image.Image):
            upscaled = Image.fromarray(np.asarray(upscaled))
        upscaled = upscaled.resize(
            (image.shape[1], image.shape[0]), Image.Resampling.LANCZOS
        ).convert("RGBA")
        result = np.array(upscaled)
        result[:, :, 3] = image[:, :, 3]
        return Image.fromarray(result, mode="RGBA")

    @staticmethod
    def _prompts(
        command: UpdateInpaintingPrompts | GenerateInpaintingCandidates,
    ) -> tuple[str, str]:
        values: list[str] = []
        for label, value in (
            ("positive prompt", command.positive_prompt),
            ("negative prompt", command.negative_prompt),
        ):
            if value is None:
                values.append("")
            elif isinstance(value, str):
                values.append(value)
            else:
                raise InpaintingNotReady(f"the {label} must be text")
        return values[0], values[1]

    @classmethod
    def _validate_generation_parameters(
        cls, command: GenerateInpaintingCandidates
    ) -> None:
        if not isinstance(command.model_name, str) or not command.model_name:
            raise InpaintingNotReady("an inpainting model is required")
        if command.workflow is not None and not isinstance(command.workflow, bytes):
            raise InpaintingNotReady("the inpainting workflow must contain bytes")
        strength = cls._finite_number(command.strength, "inpainting strength")
        if strength < 0 or strength > 1:
            raise InpaintingNotReady(
                "the inpainting strength must be between zero and one"
            )
        guidance_scale = cls._finite_number(
            command.guidance_scale, "inpainting guidance scale"
        )
        if guidance_scale <= 0:
            raise InpaintingNotReady("the inpainting guidance scale must be positive")
        cls._nonnegative_integer(command.padding, "mask padding")
        cls._nonnegative_integer(command.blur, "mask blur")

    @staticmethod
    def _finite_number(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise InpaintingNotReady(f"the {label} must be numeric")
        normalized = float(value)
        if not np.isfinite(normalized):
            raise InpaintingNotReady(f"the {label} must be finite")
        return normalized

    @staticmethod
    def _nonnegative_integer(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise InpaintingNotReady(f"the {label} must be an integer")
        normalized = int(value)
        if normalized < 0:
            raise InpaintingNotReady(f"the {label} cannot be negative")
        return normalized

    @staticmethod
    def _mode(mode: InpaintingMode) -> InpaintingMode:
        if not isinstance(mode, InpaintingMode):
            raise InpaintingNotReady("the inpainting mode is invalid")
        return mode

    @staticmethod
    def _source_image(state: AppState) -> Image.Image:
        if not isinstance(state.imgData, Image.Image):
            raise InpaintingNotReady("an input image is required")
        return state.imgData

    @classmethod
    def _selected_slice(cls, state: AppState) -> int:
        if state.selected_slice is None:
            raise InpaintingNotReady("no slice is selected")
        return cls._slice_index(state, state.selected_slice)

    @staticmethod
    def _slice_index(state: AppState, index: int) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(state.image_slices)
        ):
            raise InpaintingNotReady("the selected slice index is invalid")
        return index

    @staticmethod
    def _slice_image(state: AppState, index: int) -> np.ndarray:
        image = np.asarray(state.image_slices[index].image)
        if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
            raise InpaintingNotReady("the selected slice must be an RGBA uint8 image")
        return image

    def _load_mask(self, state: AppState, index: int) -> Image.Image:
        mask = self._artifacts.load_mask(state, index)
        if not isinstance(mask, Image.Image):
            raise InpaintingNotReady("the saved inpainting mask is invalid")
        return mask.convert("L")

    @staticmethod
    def _as_rgba(
        image: object, error_type: type[InpaintingServiceError]
    ) -> Image.Image:
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(np.asarray(image))
            except (TypeError, ValueError) as error:
                raise error_type("the inpainting image is invalid") from error
        try:
            return image.convert("RGBA")
        except (TypeError, ValueError) as error:
            raise error_type("the inpainting image is invalid") from error

    @classmethod
    def _generated_image(
        cls, image: object, expected_shape: tuple[int, ...]
    ) -> Image.Image:
        candidate = cls._as_rgba(image, InpaintingModelFailed)
        if candidate.size != (expected_shape[1], expected_shape[0]):
            raise InpaintingModelFailed(
                "the inpainting model returned an image with the wrong size"
            )
        return candidate

    @classmethod
    def _candidate_image(
        cls, image: object, state: AppState, index: int
    ) -> Image.Image:
        candidate = cls._as_rgba(image, InvalidInpaintingCandidate)
        expected = cls._slice_image(state, index).shape
        if candidate.size != (expected[1], expected[0]):
            raise InvalidInpaintingCandidate(
                "the inpainting candidate must match the selected slice"
            )
        return candidate
