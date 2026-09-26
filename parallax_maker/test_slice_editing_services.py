"""Contract tests for the framework-neutral slice-editing/mask-tool service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from .controller import AppState
from .slice import ImageSlice
from .slice_editing_services import (
    AddMaskToSlice,
    BalanceSlices,
    CopyToClipboard,
    CreateSlice,
    DeleteSlice,
    FeatherMask,
    InvalidSliceIndex,
    InvertMask,
    PasteClipboard,
    RemoveMaskFromSlice,
    ReplaceSliceImage,
    SetCheckerboard,
    SetSliceDepth,
    SliceEditingNotReady,
    SliceEditingService,
    SliceEditingUnchanged,
)
from .workflow_services import StateSaveOptions


class MemoryStateRepository:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.loaded: list[str] = []
        self.saved: list[tuple[str, StateSaveOptions]] = []

    def load(self, state_id: str) -> AppState:
        self.loaded.append(state_id)
        return self.state

    def save(self, state_id: str, state: AppState, options: StateSaveOptions) -> None:
        self.saved.append((state_id, options))


JSON_ONLY = StateSaveOptions(
    save_image_slices=False, save_depth_map=False, save_input_image=False
)


def make_slice(tmp_path: Path, name: str, depth: float, alpha_block=None) -> ImageSlice:
    image = np.zeros((10, 20, 4), dtype=np.uint8)
    image[:, :, :3] = (10, 20, 30)
    image[:, :, 3] = 255
    if alpha_block is not None:
        y0, y1, x0, x1 = alpha_block
        image[y0:y1, x0:x1, 3] = 0
    filename = tmp_path / f"{name}.png"
    Image.fromarray(image, mode="RGBA").save(filename)
    return ImageSlice(image.copy(), depth=depth, filename=str(filename))


def make_state(tmp_path: Path) -> AppState:
    state = AppState()
    state.filename = str(tmp_path)
    state.imgData = Image.new("RGB", (20, 10), (100, 110, 120))
    state.image_slices = [
        make_slice(tmp_path, "image_slice_0", depth=50),
        make_slice(tmp_path, "image_slice_1", depth=150, alpha_block=(2, 8, 5, 15)),
    ]
    return state


def make_service(state: AppState) -> tuple[SliceEditingService, MemoryStateRepository]:
    repository = MemoryStateRepository(state)
    return SliceEditingService(state_repository=repository), repository


def make_mask(state: AppState, *, inside=(2, 8, 5, 15), value=255) -> np.ndarray:
    width, height = state.imgData.size
    mask = np.zeros((height, width), dtype=np.uint8)
    y0, y1, x0, x1 = inside
    mask[y0:y1, x0:x1] = value
    return mask


def make_empty_state(tmp_path: Path) -> AppState:
    """A state with an input image but no slices yet - the trigger for
    creating a "rest of image" slice on the first mask-based create."""
    state = AppState()
    state.filename = str(tmp_path)
    state.imgData = Image.new("RGB", (20, 10), (100, 110, 120))
    state.image_slices = []
    return state


def make_custom_slice(
    tmp_path: Path, name: str, depth: float, alpha: np.ndarray, *, rgb=(10, 20, 30)
) -> ImageSlice:
    """Like ``make_slice``, but with an explicit, arbitrary alpha channel."""
    height, width = alpha.shape
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, :3] = rgb
    image[:, :, 3] = alpha
    filename = tmp_path / f"{name}.png"
    Image.fromarray(image, mode="RGBA").save(filename)
    return ImageSlice(image.copy(), depth=depth, filename=str(filename))


# --- create_slice ------------------------------------------------------------


def test_create_slice_from_mask_appends_and_selects(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.slice_mask = make_mask(state)
    state.slice_pixel = (6, 4)
    state.slice_pixel_depth = 77
    service, repository = make_service(state)

    result = service.create_slice(CreateSlice(state_id="s"))

    assert result.empty is False
    assert len(state.image_slices) == 3
    assert state.selected_slice == result.slice_index
    assert state.image_slices[result.slice_index].depth == 77
    assert result.preview_image is not None
    # update_slices' side effect: the mask/pixel used to create it is cleared.
    assert state.slice_mask is None
    assert state.slice_pixel is None
    assert state.slice_pixel_depth is None
    assert repository.saved == [("s", JSON_ONLY)]


def test_create_slice_without_mask_is_empty_at_default_depth(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    assert state.slice_mask is None
    service, _ = make_service(state)

    result = service.create_slice(CreateSlice(state_id="s"))

    assert result.empty is True
    new_slice = state.image_slices[result.slice_index]
    assert new_slice.depth == 127
    assert np.all(new_slice.image[:, :, 3] == 0)


def test_create_slice_requires_an_image(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.imgData = None
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.create_slice(CreateSlice(state_id="s"))


# --- delete_slice --------------------------------------------------------------


def test_delete_slice_clears_selection_and_previews_the_input_image(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 1
    state.slice_mask = make_mask(state)
    service, repository = make_service(state)

    result = service.delete_slice(DeleteSlice(state_id="s", slice_index=1))

    assert len(state.image_slices) == 1
    assert state.selected_slice is None
    assert state.slice_mask is None
    assert result.preview_image is state.imgData
    assert repository.saved == [("s", JSON_ONLY)]


def test_delete_slice_invalid_index_raises(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    with pytest.raises(InvalidSliceIndex):
        service.delete_slice(DeleteSlice(state_id="s", slice_index=5))


# --- add_mask_to_slice / remove_mask_from_slice ---------------------------------


def test_add_mask_to_slice_blends_mask_and_bumps_version(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 1  # has a transparent "hole" at (2:8, 5:15)
    assert state.image_slices[1].image[6, 8, 3] == 0
    state.slice_mask = make_mask(state)  # covers the same region
    original_filename = state.image_slices[1].filename
    service, repository = make_service(state)

    result = service.add_mask_to_slice(AddMaskToSlice(state_id="s"))

    assert result.slice_index == 1
    assert state.image_slices[1].filename != original_filename
    assert result.image_filename == state.image_slices[1].filename
    # The hole is filled in (create_slice_from_mask draws from state.imgData).
    assert state.image_slices[1].image[6, 8, 3] > 0
    assert state.image_slices[1].image[0, 0, 3] == 255  # outside the mask: unchanged
    assert repository.saved == [("s", JSON_ONLY)]
    # Selection is retained (still valid), but mask/pixel are cleared.
    assert state.selected_slice == 1
    assert state.slice_mask is None
    assert result.preview_image is not None


def test_add_mask_to_slice_requires_selection(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.slice_mask = make_mask(state)
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.add_mask_to_slice(AddMaskToSlice(state_id="s"))


def test_add_mask_to_slice_requires_mask(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 0
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.add_mask_to_slice(AddMaskToSlice(state_id="s"))


def test_remove_mask_from_slice_clears_alpha_under_mask(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 0
    assert state.image_slices[0].image[6, 8, 3] == 255
    state.slice_mask = make_mask(state)
    service, repository = make_service(state)

    result = service.remove_mask_from_slice(RemoveMaskFromSlice(state_id="s"))

    assert state.image_slices[0].image[6, 8, 3] == 0
    assert state.image_slices[0].image[0, 0, 3] == 255  # outside the mask: unchanged
    assert result.slice_index == 0
    assert repository.saved == [("s", JSON_ONLY)]
    assert state.slice_mask is None


def test_remove_mask_from_slice_requires_selection(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.slice_mask = make_mask(state)
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.remove_mask_from_slice(RemoveMaskFromSlice(state_id="s"))


# --- copy_to_clipboard / paste_clipboard ----------------------------------------


def test_copy_to_clipboard_with_selection_uses_composed_slice(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 1
    state.slice_mask = make_mask(state)
    service, repository = make_service(state)

    result = service.copy_to_clipboard(CopyToClipboard(state_id="s"))

    assert result.slice_index == 1
    assert state.clipboard_image is not None
    assert state.clipboard_image[:, :, 3][6, 8] == 255
    assert state.clipboard_image[:, :, 3][0, 0] == 0
    assert repository.saved == []  # no STORE_UPDATE_SLICE output in Dash: no save


def test_copy_to_clipboard_without_selection_uses_full_image(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.slice_mask = make_mask(state)
    service, _ = make_service(state)

    result = service.copy_to_clipboard(CopyToClipboard(state_id="s"))

    assert result.slice_index is None
    assert state.clipboard_image.shape[:2] == (10, 20)


def test_copy_to_clipboard_requires_mask(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.copy_to_clipboard(CopyToClipboard(state_id="s"))


def test_paste_clipboard_blends_into_selected_slice(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 0
    clipboard = np.array(state.imgData.convert("RGBA"))
    clipboard[:, :, 3] = make_mask(state)
    state.clipboard_image = clipboard
    original_filename = state.image_slices[0].filename
    service, repository = make_service(state)

    result = service.paste_clipboard(PasteClipboard(state_id="s"))

    assert result.slice_index == 0
    assert state.image_slices[0].filename != original_filename
    assert state.image_slices[0].image[6, 8, 3] == 255
    assert repository.saved == [("s", JSON_ONLY)]


def test_paste_clipboard_requires_clipboard(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 0
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.paste_clipboard(PasteClipboard(state_id="s"))


def test_paste_clipboard_requires_selection(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.clipboard_image = np.array(state.imgData.convert("RGBA"))
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.paste_clipboard(PasteClipboard(state_id="s"))


# --- balance_slices --------------------------------------------------------------


def test_balance_slices_spreads_depths_evenly(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.image_slices.append(make_slice(tmp_path, "image_slice_2", depth=200))
    service, repository = make_service(state)

    result = service.balance_slices(BalanceSlices(state_id="s"))

    assert result.depths == (0, 127, 255)
    assert [s.depth for s in state.image_slices] == [0, 127, 255]
    assert repository.saved == [("s", JSON_ONLY)]


def test_balance_slices_with_no_slices_is_unchanged(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.image_slices = []
    service, repository = make_service(state)

    with pytest.raises(SliceEditingUnchanged):
        service.balance_slices(BalanceSlices(state_id="s"))
    assert repository.saved == []


# --- set_slice_depth --------------------------------------------------------------


def test_set_slice_depth_without_reorder_keeps_selection(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 0  # depth 50
    service, repository = make_service(state)

    result = service.set_slice_depth(SetSliceDepth(state_id="s", slice_index=0, depth=51))

    assert result.reordered is False
    assert result.new_index == 0
    assert state.selected_slice == 0
    assert repository.saved == [("s", JSON_ONLY)]


def test_set_slice_depth_reorder_clears_selection_even_for_a_different_slice(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 1  # depth 150, untouched by this edit
    service, _ = make_service(state)

    # Push slice 0 (depth 50) past slice 1 (depth 150).
    result = service.set_slice_depth(SetSliceDepth(state_id="s", slice_index=0, depth=200))

    assert result.reordered is True
    assert result.new_index == 1
    assert state.selected_slice is None


def test_set_slice_depth_always_clears_inpainting_selection(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 0
    state.selected_inpainting = 2
    service, _ = make_service(state)

    service.set_slice_depth(SetSliceDepth(state_id="s", slice_index=0, depth=51))

    assert state.selected_inpainting is None


def test_set_slice_depth_invalid_index_raises(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    with pytest.raises(InvalidSliceIndex):
        service.set_slice_depth(SetSliceDepth(state_id="s", slice_index=9, depth=1))


# --- replace_slice_image (upload) -------------------------------------------------


def test_replace_slice_image_matching_aspect_is_not_resized(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    replacement = Image.new("RGBA", (20, 10), (1, 2, 3, 255))
    original_filename = state.image_slices[0].filename
    service, repository = make_service(state)

    result = service.replace_slice_image(
        ReplaceSliceImage(state_id="s", slice_index=0, image=replacement)
    )

    assert result.aspect_ratio_fixed is False
    assert state.image_slices[0].image.shape[:2] == (10, 20)
    assert tuple(state.image_slices[0].image[0, 0]) == (1, 2, 3, 255)
    assert state.image_slices[0].filename != original_filename
    assert repository.saved == [("s", SliceEditingService.JSON_AND_INPUT)]
    assert isinstance(result.composed_input_image, Image.Image)


def test_replace_slice_image_is_resized_to_the_slice_canvas(tmp_path: Path) -> None:
    """Intentional fix: Dash resized to the upload's own height, which could
    shrink the slice and crash the recompose. Uploads now always match the
    canvas, so the recompose over every slice succeeds."""
    state = make_state(tmp_path)  # two slices, both 10x20
    replacement = Image.new("RGBA", (4, 1), (5, 6, 7, 255))  # aspect 4.0 != 2.0
    service, repository = make_service(state)

    result = service.replace_slice_image(
        ReplaceSliceImage(state_id="s", slice_index=0, image=replacement)
    )

    assert result.aspect_ratio_fixed is True
    assert result.target_aspect_ratio == pytest.approx(2.0)
    assert result.source_aspect_ratio == pytest.approx(4.0)
    assert state.image_slices[0].image.shape[:2] == (10, 20)
    assert result.composed_input_image.size == (20, 10)
    assert repository.saved == [("s", SliceEditingService.JSON_AND_INPUT)]


def test_replace_slice_image_same_aspect_different_size_is_resized(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    replacement = Image.new("RGBA", (40, 20), (5, 6, 7, 255))
    service, _ = make_service(state)

    result = service.replace_slice_image(
        ReplaceSliceImage(state_id="s", slice_index=1, image=replacement)
    )

    assert result.aspect_ratio_fixed is False
    assert state.image_slices[1].image.shape[:2] == (10, 20)
    assert tuple(state.image_slices[1].image[5, 5]) == (5, 6, 7, 255)


def test_replace_slice_image_invalid_index_raises(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    with pytest.raises(InvalidSliceIndex):
        service.replace_slice_image(
            ReplaceSliceImage(
                state_id="s", slice_index=9, image=Image.new("RGBA", (1, 1))
            )
        )


# --- invert_mask / feather_mask ---------------------------------------------------


def test_invert_mask_creates_an_all_zero_mask_first(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    assert state.slice_mask is None
    service, _ = make_service(state)

    result = service.invert_mask(InvertMask(state_id="s"))

    assert np.all(state.slice_mask == 255)
    assert result.preview_image is not None


def test_invert_mask_flips_an_existing_mask(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.slice_mask = make_mask(state)
    service, _ = make_service(state)

    service.invert_mask(InvertMask(state_id="s"))

    assert state.slice_mask[6, 8] == 0  # was 255
    assert state.slice_mask[0, 0] == 255  # was 0


def test_feather_mask_requires_a_mask(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.feather_mask(FeatherMask(state_id="s"))


def test_feather_mask_rejects_a_zero_amount(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.slice_mask = np.zeros((10, 20), dtype=np.uint8)
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.feather_mask(FeatherMask(state_id="s", amount=0))


def test_feather_mask_blurs_the_edge_while_keeping_the_interior_opaque(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.imgData = Image.new("RGB", (40, 40), (100, 110, 120))
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    state.slice_mask = mask
    service, _ = make_service(state)

    service.feather_mask(FeatherMask(state_id="s", amount=6))

    # Deep interior stays fully opaque...
    assert state.slice_mask[20, 20] == 255
    # ...but a point just outside the original hard edge now has an
    # intermediate value instead of the original mask's binary 0/255.
    edge_value = state.slice_mask[10, 8]
    assert 0 < edge_value < 255
    # Far away, untouched by a small local blur.
    assert state.slice_mask[0, 0] == 0


# --- set_checkerboard ---------------------------------------------------------------


def test_set_checkerboard_without_selection_has_no_preview(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    result = service.set_checkerboard(SetCheckerboard(state_id="s", enabled=True))

    assert result.enabled is True
    assert state.use_checkerboard is True
    assert result.preview_image is None
    assert isinstance(state, AppState)  # sanity: no crash without a selection


def test_set_checkerboard_with_selection_recomposes_preview(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.selected_slice = 0
    service, _ = make_service(state)

    result = service.set_checkerboard(SetCheckerboard(state_id="s", enabled=True))

    assert result.preview_image is not None


def test_set_checkerboard_requires_a_boolean(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    service, _ = make_service(state)

    with pytest.raises(SliceEditingNotReady):
        service.set_checkerboard(SetCheckerboard(state_id="s", enabled="yes"))  # type: ignore[arg-type]


# --- "rest of image" slice: creation and sync -----------------------------------


def test_create_slice_first_cut_also_creates_a_rest_slice(tmp_path: Path) -> None:
    state = make_empty_state(tmp_path)
    state.slice_mask = make_mask(state)
    service, repository = make_service(state)

    result = service.create_slice(CreateSlice(state_id="s"))

    assert len(state.image_slices) == 2
    rest = next(s for s in state.image_slices if s.is_rest)
    obj = next(s for s in state.image_slices if not s.is_rest)
    assert rest.depth == 0
    assert result.rest_slice_filename == str(rest.filename)

    # Complementary alpha: the object and rest slice exactly cover the image
    # between them, everywhere - not just inside the (binary) mask.
    total = obj.image[:, :, 3].astype(int) + rest.image[:, :, 3].astype(int)
    assert np.all(total == 255)

    # The rest slice's RGB comes from the input image.
    source_rgb = np.array(state.imgData.convert("RGB"))
    np.testing.assert_array_equal(rest.image[:, :, :3], source_rgb)

    # Selection follows the object slice even though the rest slice (depth 0)
    # may have been inserted ahead of it.
    obj_index = state.image_slices.index(obj)
    assert state.selected_slice == obj_index
    assert result.slice_index == obj_index
    assert repository.saved == [("s", JSON_ONLY)]


def test_create_slice_without_a_mask_never_touches_the_rest_slice(
    tmp_path: Path,
) -> None:
    state = make_empty_state(tmp_path)
    assert state.slice_mask is None
    service, _ = make_service(state)

    result = service.create_slice(CreateSlice(state_id="s"))

    assert result.empty is True
    assert len(state.image_slices) == 1
    assert result.rest_slice_filename is None
    assert not state.image_slices[0].is_rest


def test_create_slice_in_a_depth_split_project_does_not_add_a_rest_slice(
    tmp_path: Path,
) -> None:
    """Slices already exist (as if split by depth), but none is the rest
    slice: creating from a mask behaves exactly as it did before this
    feature, with no rest slice added."""
    state = make_state(tmp_path)
    state.slice_mask = make_mask(state)
    service, _ = make_service(state)

    result = service.create_slice(CreateSlice(state_id="s"))

    assert len(state.image_slices) == 3
    assert not any(s.is_rest for s in state.image_slices)
    assert result.rest_slice_filename is None


def test_create_slice_second_cut_subtracts_from_the_existing_rest_slice(
    tmp_path: Path,
) -> None:
    state = make_empty_state(tmp_path)
    state.slice_mask = make_mask(state, inside=(2, 8, 5, 15))
    service, repository = make_service(state)

    service.create_slice(CreateSlice(state_id="s"))  # first cut: creates the rest
    rest = next(s for s in state.image_slices if s.is_rest)
    rest_filename_before = rest.filename
    rest_alpha_before = rest.image[:, :, 3].copy()

    # A second, disjoint cut.
    state.slice_mask = make_mask(state, inside=(0, 2, 0, 5))
    repository.saved.clear()
    result = service.create_slice(CreateSlice(state_id="s"))

    assert len(state.image_slices) == 3
    assert result.rest_slice_filename is None  # not newly created, only updated
    rest_after = next(s for s in state.image_slices if s.is_rest)
    assert rest_after is rest  # same slice, new version
    assert rest_after.filename != rest_filename_before

    # The rest slice's alpha only ever decreases, and it strictly decreases
    # somewhere (where the new object slice now covers it).
    assert np.all(rest_after.image[:, :, 3] <= rest_alpha_before)
    assert np.any(rest_after.image[:, :, 3] < rest_alpha_before)
    assert repository.saved == [("s", JSON_ONLY)]


# --- "rest of image" slice: kept in sync on delete/add-mask/remove-mask --------


def test_delete_slice_restores_only_uncovered_pixels_and_preserves_rest_edits(
    tmp_path: Path,
) -> None:
    height, width = 10, 20
    state = AppState()
    state.filename = str(tmp_path)
    source_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    source_rgb[:, :, 0] = 111
    source_rgb[:, :, 1] = 222
    source_rgb[:, :, 2] = 33
    state.imgData = Image.fromarray(source_rgb, mode="RGB")

    # Object A (to be deleted) covers rows 0:5, cols 0:10.
    alpha_a = np.zeros((height, width), dtype=np.uint8)
    alpha_a[0:5, 0:10] = 255
    object_a = make_custom_slice(tmp_path, "image_slice_1", depth=50, alpha=alpha_a)

    # Object B (stays) covers a small patch that overlaps A's region.
    alpha_b = np.zeros((height, width), dtype=np.uint8)
    alpha_b[3:5, 5:8] = 255
    object_b = make_custom_slice(tmp_path, "image_slice_2", depth=100, alpha=alpha_b)

    # The rest slice covers everything neither A nor B does.
    rest_alpha = np.where((alpha_a > 0) | (alpha_b > 0), 0, 255).astype(np.uint8)
    rest_image = np.zeros((height, width, 4), dtype=np.uint8)
    rest_image[:, :, :3] = source_rgb
    rest_image[:, :, 3] = rest_alpha
    # Simulate prior inpainting on the rest slice: one pixel inside the area
    # that will be uncovered by the delete, one pixel well outside it.
    rest_image[1, 2] = (9, 9, 9, 222)
    rest_image[6, 1] = (7, 7, 7, 255)
    rest_filename = tmp_path / "image_slice_0.png"
    Image.fromarray(rest_image, mode="RGBA").save(rest_filename)
    rest_slice = ImageSlice(rest_image.copy(), depth=0, filename=str(rest_filename))
    rest_slice.is_rest = True

    state.image_slices = [rest_slice, object_a, object_b]
    service, repository = make_service(state)

    service.delete_slice(DeleteSlice(state_id="s", slice_index=1))

    assert state.image_slices == [rest_slice, object_b]
    assert rest_slice.filename != str(rest_filename)  # a new version was saved

    # (1, 2): inside A, not covered by B - made fully opaque again, keeping
    # the partly visible "inpainted" color rather than overwriting it.
    assert tuple(int(v) for v in rest_slice.image[1, 2]) == (9, 9, 9, 255)
    # (0, 0): inside A, transparent on the rest - refilled from the input.
    assert tuple(int(v) for v in rest_slice.image[0, 0]) == (111, 222, 33, 255)
    # (4, 6): inside A, but still covered by B - left exactly as it was.
    assert tuple(int(v) for v in rest_slice.image[4, 6]) == (111, 222, 33, 0)
    # (6, 1): outside A entirely - the "inpainted" pixel survives untouched.
    assert tuple(int(v) for v in rest_slice.image[6, 1]) == (7, 7, 7, 255)
    assert repository.saved == [("s", JSON_ONLY)]


def test_delete_slice_of_the_rest_slice_itself_removes_it(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.image_slices[0].is_rest = True
    service, _ = make_service(state)

    service.delete_slice(DeleteSlice(state_id="s", slice_index=0))

    assert len(state.image_slices) == 1
    assert not any(s.is_rest for s in state.image_slices)


def test_add_mask_to_slice_subtracts_the_added_area_from_the_rest(
    tmp_path: Path,
) -> None:
    height, width = 10, 20
    state = AppState()
    state.filename = str(tmp_path)
    state.imgData = Image.new("RGB", (width, height), (100, 110, 120))

    # The object already covers a small patch; the rest covers everything else.
    alpha_object = np.zeros((height, width), dtype=np.uint8)
    alpha_object[0:2, 0:4] = 255
    object_slice = make_custom_slice(tmp_path, "image_slice_1", depth=50, alpha=alpha_object)

    rest_alpha = np.where(alpha_object > 0, 0, 255).astype(np.uint8)
    rest_slice = make_custom_slice(tmp_path, "image_slice_0", depth=0, alpha=rest_alpha)
    rest_slice.is_rest = True

    state.image_slices = [rest_slice, object_slice]
    state.selected_slice = 1
    state.slice_mask = make_mask(state, inside=(4, 8, 10, 18))
    service, repository = make_service(state)

    original_rest_filename = rest_slice.filename
    service.add_mask_to_slice(AddMaskToSlice(state_id="s"))

    assert rest_slice.filename != original_rest_filename  # version bumped
    # Where the object grew (the newly-added mask), the rest gives up alpha.
    assert rest_slice.image[6, 14, 3] < 255
    # Where the object already covered, the rest is unaffected (still zero).
    assert rest_slice.image[0, 0, 3] == 0
    # Far from both (outside the feathered mask's reach too), untouched.
    assert rest_slice.image[9, 0, 3] == 255
    assert repository.saved == [("s", JSON_ONLY)]


def test_remove_mask_from_slice_restores_only_uncovered_pixels_into_the_rest(
    tmp_path: Path,
) -> None:
    height, width = 10, 20
    state = AppState()
    state.filename = str(tmp_path)
    source_rgb = np.full((height, width, 3), (50, 60, 70), dtype=np.uint8)
    state.imgData = Image.fromarray(source_rgb, mode="RGB")

    # The object covers rows 2:8, cols 2:18.
    alpha_object = np.zeros((height, width), dtype=np.uint8)
    alpha_object[2:8, 2:18] = 255
    object_slice = make_custom_slice(tmp_path, "image_slice_1", depth=50, alpha=alpha_object)

    # Another slice keeps covering a sub-patch of the object's area.
    alpha_other = np.zeros((height, width), dtype=np.uint8)
    alpha_other[4:6, 10:14] = 255
    other_slice = make_custom_slice(tmp_path, "image_slice_2", depth=100, alpha=alpha_other)

    rest_alpha = np.where(
        (alpha_object > 0) | (alpha_other > 0), 0, 255
    ).astype(np.uint8)
    rest_slice = make_custom_slice(
        tmp_path, "image_slice_0", depth=0, alpha=rest_alpha, rgb=(50, 60, 70)
    )
    rest_slice.is_rest = True

    state.image_slices = [rest_slice, object_slice, other_slice]
    state.selected_slice = 1
    # Remove the mask covering the object's entire region.
    state.slice_mask = make_mask(state, inside=(2, 8, 2, 18))
    service, repository = make_service(state)

    original_rest_filename = rest_slice.filename
    service.remove_mask_from_slice(RemoveMaskFromSlice(state_id="s"))

    assert rest_slice.filename != original_rest_filename
    # Inside the object's region but outside the other slice: restored.
    assert tuple(int(v) for v in rest_slice.image[2, 2]) == (50, 60, 70, 255)
    # Inside both the object's region and the other slice: left alone.
    assert rest_slice.image[5, 12, 3] == 0
    # Entirely outside the object's region: untouched.
    assert rest_slice.image[9, 19, 3] == 255
    assert repository.saved == [("s", JSON_ONLY)]


def test_remove_mask_from_slice_restores_a_feathered_removal_only_partly(
    tmp_path: Path,
) -> None:
    height, width = 10, 20
    state = AppState()
    state.filename = str(tmp_path)
    source_rgb = np.full((height, width, 3), (50, 60, 70), dtype=np.uint8)
    state.imgData = Image.fromarray(source_rgb, mode="RGB")

    alpha_object = np.zeros((height, width), dtype=np.uint8)
    alpha_object[2:8, 2:18] = 255
    object_slice = make_custom_slice(tmp_path, "image_slice_1", depth=50, alpha=alpha_object)
    rest_slice = make_custom_slice(
        tmp_path,
        "image_slice_0",
        depth=0,
        alpha=(255 - alpha_object).astype(np.uint8),
        rgb=(50, 60, 70),
    )
    rest_slice.is_rest = True

    state.image_slices = [rest_slice, object_slice]
    state.selected_slice = 1
    # A feathered mask: removes only part of the object's alpha.
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[2:8, 2:18] = 100
    state.slice_mask = mask
    service, _ = make_service(state)

    service.remove_mask_from_slice(RemoveMaskFromSlice(state_id="s"))

    remaining = int(object_slice.image[4, 4, 3])
    assert 0 < remaining < 255
    # The rest slice gets back exactly what the object lost.
    assert int(rest_slice.image[4, 4, 3]) == 255 - remaining
    assert int(rest_slice.image[0, 0, 3]) == 255
