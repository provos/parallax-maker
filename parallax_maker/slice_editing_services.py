"""Framework-neutral services for slice editing and mask tools.

UI adapters own transport decoding/encoding, component rendering, callback
control flow, and log messages. This module owns the "slice editing" and
"mask tools" mutations that `webui.py`/`components.py` still perform inline
on `AppState` (create/delete/add-mask/remove-mask/copy/paste/balance/set-depth
/upload/invert/feather/checkerboard), reproducing their exact save flags,
versioning and selection/preview side effects.

Two Dash quirks are load-bearing for several commands here and are documented
where they matter below (see also PARITY.md "Known quirks"):

- Any command that only sets ``STORE_UPDATE_SLICE`` (create/add-mask/
  remove-mask/paste/balance/set-depth) chains into Dash's ``update_slices``
  (webui.py:913), which - whenever a slice is still selected afterward -
  unconditionally clears ``slice_pixel``/``slice_pixel_depth``/``slice_mask``
  as a side effect of recomposing the preview. A mask used to create/add/
  paste does not survive the operation. ``_refresh_selection`` below
  reproduces this exactly so callers don't have to.
- ``record_depth_input`` (``SetSliceDepth`` here) additionally sets
  ``STORE_INPAINTING``, which chains into ``react_selected_slice_change``
  (CMP-07) and always clears ``state.selected_inpainting`` - not only when
  the edited slice was the selected one.

``webui.py``'s own ``balance_slices_request`` reads a nonexistent
``state.image_depths`` attribute and 500s on every click, independently of
and unrelated to the (now-fixed) ``AppState.balance_slices_depths`` bug; that
Dash callback is unreachable today. ``BalanceSlices`` below implements the
*intended*, now-correct behavior new UIs can actually use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .controller import AppState, CompositeMode
from .segmentation import blend_with_alpha, create_slice_from_mask, remove_mask_from_alpha
from .slice import ImageSlice
from .workflow_services import StateSaveOptions

#: Mirrors webui.EXPAND_MASK; kept independent so this module has no
#: dependency on the frozen Dash entry point.
DEFAULT_MASK_EXPAND = 5

#: cv2.blur kernel size used by Dash's "Feather" button (components.py:1649).
DEFAULT_FEATHER_AMOUNT = 10

#: Depth assigned to a slice created with no mask selected (webui.py:844).
DEFAULT_NEW_SLICE_DEPTH = 127


class SliceEditingServiceError(Exception):
    """Base class for slice-editing/mask-tool domain failures."""


class SliceEditingNotReady(SliceEditingServiceError):
    """Required image, slice-selection, mask, or clipboard state is missing."""


class InvalidSliceIndex(SliceEditingServiceError):
    """A slice index is missing, out of range, or otherwise invalid."""


class SliceEditingUnchanged(SliceEditingServiceError):
    """The requested mutation would not change slice-editing state."""


class SliceEditingStateRepository(Protocol):
    def load(self, state_id: str) -> AppState: ...

    def save(
        self, state_id: str, state: AppState, options: StateSaveOptions
    ) -> None: ...


class CachedSliceEditingStateRepository:
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


@dataclass(frozen=True)
class CreateSlice:
    state_id: str


@dataclass(frozen=True)
class CreatedSliceResult:
    state_id: str
    slice_index: int
    empty: bool
    preview_image: Image.Image | None


@dataclass(frozen=True)
class DeleteSlice:
    state_id: str
    slice_index: int


@dataclass(frozen=True)
class DeletedSliceResult:
    state_id: str
    slice_index: int
    preview_image: Image.Image


@dataclass(frozen=True)
class AddMaskToSlice:
    state_id: str


@dataclass(frozen=True)
class AddedMaskToSliceResult:
    state_id: str
    slice_index: int
    image_filename: str
    preview_image: Image.Image | None


@dataclass(frozen=True)
class RemoveMaskFromSlice:
    state_id: str


@dataclass(frozen=True)
class RemovedMaskFromSliceResult:
    state_id: str
    slice_index: int
    image_filename: str
    preview_image: Image.Image | None


@dataclass(frozen=True)
class CopyToClipboard:
    state_id: str


@dataclass(frozen=True)
class CopiedToClipboardResult:
    state_id: str
    slice_index: int | None


@dataclass(frozen=True)
class PasteClipboard:
    state_id: str


@dataclass(frozen=True)
class PastedClipboardResult:
    state_id: str
    slice_index: int
    image_filename: str
    preview_image: Image.Image | None


@dataclass(frozen=True)
class BalanceSlices:
    state_id: str


@dataclass(frozen=True)
class BalancedSlicesResult:
    state_id: str
    depths: tuple[float, ...]
    preview_image: Image.Image | None


@dataclass(frozen=True)
class SetSliceDepth:
    state_id: str
    slice_index: int
    depth: float


@dataclass(frozen=True)
class SetGroundPlane:
    """Marks (or unmarks) a slice as the scene's horizontal ground plane."""

    state_id: str
    slice_index: int
    is_ground: bool


@dataclass(frozen=True)
class SetGroundPlaneResult:
    state_id: str
    slice_index: int
    is_ground: bool
    changed: bool


@dataclass(frozen=True)
class SetSliceDepthResult:
    state_id: str
    slice_index: int
    new_index: int
    reordered: bool
    preview_image: Image.Image | None


@dataclass(frozen=True)
class ReplaceSliceImage:
    state_id: str
    slice_index: int
    image: Image.Image


@dataclass(frozen=True)
class ReplacedSliceImageResult:
    state_id: str
    slice_index: int
    image_filename: str
    aspect_ratio_fixed: bool
    source_aspect_ratio: float
    target_aspect_ratio: float
    composed_input_image: Image.Image


@dataclass(frozen=True)
class InvertMask:
    state_id: str


@dataclass(frozen=True)
class InvertedMaskResult:
    state_id: str
    preview_image: Image.Image


@dataclass(frozen=True)
class FeatherMask:
    state_id: str
    amount: int = DEFAULT_FEATHER_AMOUNT


@dataclass(frozen=True)
class FeatheredMaskResult:
    state_id: str
    preview_image: Image.Image


@dataclass(frozen=True)
class SetCheckerboard:
    state_id: str
    enabled: bool


@dataclass(frozen=True)
class SetCheckerboardResult:
    state_id: str
    enabled: bool
    preview_image: Image.Image | None


class SliceEditingService:
    """Slice list editing and mask tool commands, independent of a UI framework."""

    JSON_ONLY = StateSaveOptions(
        save_image_slices=False,
        save_depth_map=False,
        save_input_image=False,
    )
    JSON_AND_INPUT = StateSaveOptions(
        save_image_slices=False,
        save_depth_map=False,
        save_input_image=True,
    )

    def __init__(
        self,
        state_repository: SliceEditingStateRepository | None = None,
        *,
        mask_expand: int = DEFAULT_MASK_EXPAND,
    ) -> None:
        self._states = state_repository or CachedSliceEditingStateRepository()
        self._mask_expand = mask_expand

    # -- slice list -----------------------------------------------------

    def create_slice(self, command: CreateSlice) -> CreatedSliceResult:
        state = self._states.load(command.state_id)
        source = self._require_image(state)

        empty = state.slice_mask is None
        if empty:
            image = Image.new("RGBA", source.size, (0, 0, 0, 0))
            depth = DEFAULT_NEW_SLICE_DEPTH
        else:
            depth = (
                state.slice_pixel_depth
                if state.slice_pixel is not None
                else DEFAULT_NEW_SLICE_DEPTH
            )
            image = create_slice_from_mask(
                source, state.slice_mask, num_expand=self._mask_expand
            )

        image_slice = ImageSlice(image, depth)
        index = state.add_slice(image_slice)
        state.selected_slice = index
        image_slice.save_image()
        self._states.save(command.state_id, state, self.JSON_ONLY)

        preview = self._refresh_selection(state)
        return CreatedSliceResult(
            state_id=command.state_id,
            slice_index=index,
            empty=empty,
            preview_image=preview,
        )

    def delete_slice(self, command: DeleteSlice) -> DeletedSliceResult:
        state = self._states.load(command.state_id)
        index = self._slice_index(state, command.slice_index)
        source = self._require_image(state)

        if not state.delete_slice(index):
            raise InvalidSliceIndex(f"slice index {index} is invalid")
        self._states.save(command.state_id, state, self.JSON_ONLY)

        # Mirrors delete_slice_request's own direct IMAGE.src output
        # (webui.py:644): always the raw input image, since delete_slice()
        # already clears the selection/mask/pixel itself.
        return DeletedSliceResult(
            state_id=command.state_id, slice_index=index, preview_image=source
        )

    def add_mask_to_slice(self, command: AddMaskToSlice) -> AddedMaskToSliceResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        mask = self._require_mask(state)
        source = self._require_image(state)

        merge_image = create_slice_from_mask(source, mask, num_expand=self._mask_expand)
        blend_with_alpha(state.image_slices[index].image, merge_image)
        filename = state.image_slices[index].new_version()
        self._states.save(command.state_id, state, self.JSON_ONLY)

        preview = self._refresh_selection(state)
        return AddedMaskToSliceResult(
            state_id=command.state_id,
            slice_index=index,
            image_filename=str(filename),
            preview_image=preview,
        )

    def remove_mask_from_slice(
        self, command: RemoveMaskFromSlice
    ) -> RemovedMaskFromSliceResult:
        state = self._states.load(command.state_id)
        index = self._selected_slice(state)
        mask = self._require_mask(state)

        final_mask = remove_mask_from_alpha(state.image_slices[index].image, mask)
        state.image_slices[index].image[:, :, 3] = final_mask
        filename = state.image_slices[index].new_version()
        self._states.save(command.state_id, state, self.JSON_ONLY)

        preview = self._refresh_selection(state)
        return RemovedMaskFromSliceResult(
            state_id=command.state_id,
            slice_index=index,
            image_filename=str(filename),
            preview_image=preview,
        )

    def copy_to_clipboard(self, command: CopyToClipboard) -> CopiedToClipboardResult:
        state = self._states.load(command.state_id)
        mask = self._require_mask(state)

        if state.selected_slice is not None:
            image = state.slice_image_composed(state.selected_slice, CompositeMode.NONE)
        else:
            image = self._require_image(state)
        pixels = np.array(image.convert("RGBA"))
        pixels[:, :, 3] = mask
        state.clipboard_image = pixels

        # No STORE_UPDATE_SLICE output in Dash's copy_to_clipboard: no save,
        # no preview change, no interaction-state clearing.
        return CopiedToClipboardResult(
            state_id=command.state_id, slice_index=state.selected_slice
        )

    def paste_clipboard(self, command: PasteClipboard) -> PastedClipboardResult:
        state = self._states.load(command.state_id)
        if state.clipboard_image is None:
            raise SliceEditingNotReady("nothing is in the clipboard")
        index = self._selected_slice(state)

        blend_with_alpha(state.image_slices[index].image, state.clipboard_image)
        filename = state.image_slices[index].new_version()
        self._states.save(command.state_id, state, self.JSON_ONLY)

        preview = self._refresh_selection(state)
        return PastedClipboardResult(
            state_id=command.state_id,
            slice_index=index,
            image_filename=str(filename),
            preview_image=preview,
        )

    def balance_slices(self, command: BalanceSlices) -> BalancedSlicesResult:
        state = self._states.load(command.state_id)
        if len(state.image_slices) == 0:
            # Mirrors the *intent* of webui.py:883's guard (itself unreachable
            # due to a separate, pre-existing `state.image_depths` bug; see
            # the module docstring): nothing to balance is a no-op, not an
            # error.
            raise SliceEditingUnchanged("there are no slices to balance")

        state.balance_slices_depths()
        self._states.save(command.state_id, state, self.JSON_ONLY)

        preview = self._refresh_selection(state)
        depths = tuple(image_slice.depth for image_slice in state.image_slices)
        return BalancedSlicesResult(
            state_id=command.state_id, depths=depths, preview_image=preview
        )

    def set_slice_depth(self, command: SetSliceDepth) -> SetSliceDepthResult:
        state = self._states.load(command.state_id)
        index = self._slice_index(state, command.slice_index)
        depth = self._finite_number(command.depth, "slice depth")

        new_index = state.change_slice_depth(index, depth)
        reordered = new_index != index
        if reordered:
            state.selected_slice = None
        # record_depth_input also flips STORE_INPAINTING, which always clears
        # the inpainting candidate selection via react_selected_slice_change
        # (CMP-07) - regardless of whether this specific slice was selected.
        state.selected_inpainting = None
        self._states.save(command.state_id, state, self.JSON_ONLY)

        preview = self._refresh_selection(state)
        return SetSliceDepthResult(
            state_id=command.state_id,
            slice_index=index,
            new_index=new_index,
            reordered=reordered,
            preview_image=preview,
        )

    def set_ground_plane(self, command: SetGroundPlane) -> SetGroundPlaneResult:
        """At most one slice is the ground: marking one unmarks any other.

        Marking requires a ground in view (the horizon above the image's
        bottom edge, see ``Camera.ground_height``).
        """
        state = self._states.load(command.state_id)
        index = self._slice_index(state, command.slice_index)
        target = state.image_slices[index]

        if command.is_ground:
            width, height = state.imgData.size
            try:
                state.camera.ground_height(width, height)
            except ValueError as error:
                raise SliceEditingNotReady(
                    f"no ground in view: {error}; lower the camera pitch"
                ) from None

        changed = target.is_ground_plane != command.is_ground
        for i, image_slice in enumerate(state.image_slices):
            want = command.is_ground and i == index
            if image_slice.is_ground_plane != want:
                image_slice.is_ground_plane = want
                changed = True
        if changed:
            self._states.save(command.state_id, state, self.JSON_ONLY)
        return SetGroundPlaneResult(
            state_id=command.state_id,
            slice_index=index,
            is_ground=command.is_ground,
            changed=changed,
        )

    def replace_slice_image(self, command: ReplaceSliceImage) -> ReplacedSliceImageResult:
        state = self._states.load(command.state_id)
        index = self._slice_index(state, command.slice_index)
        if not isinstance(command.image, Image.Image):
            raise SliceEditingNotReady("the replacement image must be a PIL image")

        existing = state.image_slices[index].image
        target_aspect_ratio = existing.shape[1] / existing.shape[0]
        image = command.image.convert("RGBA")
        source_aspect_ratio = image.size[0] / image.size[1]

        aspect_ratio_fixed = source_aspect_ratio != target_aspect_ratio
        canvas_size = (existing.shape[1], existing.shape[0])
        if image.size != canvas_size:
            # Intentional fix (see PARITY.md "Known quirks"): Dash's slice_upload
            # resizes to the uploaded image's own height, which can shrink the
            # slice below the canvas size and crash the recompose below. Every
            # slice must match the canvas, so resize to it exactly.
            image = image.resize(canvas_size, Image.LANCZOS)

        filename = state.image_slices[index].new_version(np.array(image))

        composed = state.image_slices[0].image.copy()
        for image_slice in state.image_slices[1:]:
            blend_with_alpha(composed, image_slice.image)
        state.imgData = Image.fromarray(composed)
        # Persist the recomposed input too; Dash leaves the stale input file
        # on disk, so a later restore silently loses the upload.
        self._states.save(command.state_id, state, self.JSON_AND_INPUT)

        return ReplacedSliceImageResult(
            state_id=command.state_id,
            slice_index=index,
            image_filename=str(filename),
            aspect_ratio_fixed=aspect_ratio_fixed,
            source_aspect_ratio=source_aspect_ratio,
            target_aspect_ratio=target_aspect_ratio,
            composed_input_image=state.imgData,
        )

    # -- mask tools -------------------------------------------------------

    def invert_mask(self, command: InvertMask) -> InvertedMaskResult:
        state = self._states.load(command.state_id)
        source = self._require_image(state)
        if state.slice_mask is None:
            shape = (source.size[1], source.size[0])
            state.slice_mask = np.zeros(shape, dtype=np.uint8)
        state.slice_mask = 255 - state.slice_mask

        preview = state.apply_mask(source, state.slice_mask)
        return InvertedMaskResult(state_id=command.state_id, preview_image=preview)

    def feather_mask(self, command: FeatherMask) -> FeatheredMaskResult:
        state = self._states.load(command.state_id)
        source = self._require_image(state)
        mask = self._require_mask(state)
        amount = self._nonnegative_integer(command.amount, "feather amount")
        if amount == 0:
            raise SliceEditingNotReady("the feather amount must be positive")

        state.slice_mask = cv2.blur(mask, (amount, amount))
        preview = state.apply_mask(source, state.slice_mask)
        return FeatheredMaskResult(state_id=command.state_id, preview_image=preview)

    def set_checkerboard(self, command: SetCheckerboard) -> SetCheckerboardResult:
        state = self._states.load(command.state_id)
        if not isinstance(command.enabled, bool):
            raise SliceEditingNotReady("the checkerboard flag must be a boolean")
        state.use_checkerboard = command.enabled

        preview = None
        if state.selected_slice is not None:
            mode = (
                CompositeMode.CHECKERBOARD
                if state.use_checkerboard
                else CompositeMode.GRAYSCALE
            )
            preview = state.slice_image_composed(state.selected_slice, mode=mode)
        return SetCheckerboardResult(
            state_id=command.state_id,
            enabled=state.use_checkerboard,
            preview_image=preview,
        )

    # -- shared helpers -----------------------------------------------------

    @staticmethod
    def _refresh_selection(state: AppState) -> Image.Image | None:
        return refresh_selection_preview(state)

    @staticmethod
    def _require_image(state: AppState) -> Image.Image:
        if not isinstance(state.imgData, Image.Image):
            raise SliceEditingNotReady("an input image is required")
        return state.imgData

    @staticmethod
    def _require_mask(state: AppState) -> np.ndarray:
        if state.slice_mask is None:
            raise SliceEditingNotReady("no mask is selected")
        return state.slice_mask

    @classmethod
    def _selected_slice(cls, state: AppState) -> int:
        if state.selected_slice is None:
            raise SliceEditingNotReady("no slice is selected")
        return cls._slice_index(state, state.selected_slice)

    @staticmethod
    def _slice_index(state: AppState, index: int) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(state.image_slices)
        ):
            raise InvalidSliceIndex(f"slice index {index!r} is invalid")
        return index

    @staticmethod
    def _finite_number(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise SliceEditingNotReady(f"the {label} must be numeric")
        normalized = float(value)
        if not np.isfinite(normalized):
            raise SliceEditingNotReady(f"the {label} must be finite")
        return normalized

    @staticmethod
    def _nonnegative_integer(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise SliceEditingNotReady(f"the {label} must be an integer")
        normalized = int(value)
        if normalized < 0:
            raise SliceEditingNotReady(f"the {label} cannot be negative")
        return normalized


def refresh_selection_preview(state: AppState) -> Image.Image | None:
    """Reproduce update_slices' (webui.py:913) selection side effect.

    Whenever a slice remains selected after a slice-list mutation, Dash
    recomposes its preview and unconditionally clears
    slice_pixel/slice_pixel_depth/slice_mask - the mask used to perform
    the mutation does not survive it. Returns None (no preview change)
    when nothing is selected, matching update_slices' own `no_update`.

    Public (not just the service's own concern) because the API layer's
    undo/redo endpoints trigger the exact same downstream Dash chain
    (undo_slice_request also only sets STORE_UPDATE_SLICE=True) without going
    through a SliceEditingService command of their own - they call the
    existing InpaintingService.move_slice_version instead.
    """
    if state.selected_slice is None:
        return None
    mode = (
        CompositeMode.CHECKERBOARD
        if state.use_checkerboard
        else CompositeMode.GRAYSCALE
    )
    preview = state.slice_image_composed(state.selected_slice, mode=mode)
    state.slice_pixel = None
    state.slice_pixel_depth = None
    state.slice_mask = None
    return preview
