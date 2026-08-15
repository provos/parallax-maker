"""Web UI callback adapters for inpainting commands outside components.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from dash.exceptions import PreventUpdate
import pytest

from . import webui
from .inpainting_services import (
    ClearInpaintingSelection,
    InpaintingNotReady,
    InpaintingUnchanged,
    MoveSliceVersion,
    SliceVersionDirection,
    SliceVersionUnavailable,
    UpdateInpaintingModel,
    UpdateInpaintingPrompts,
)


@pytest.fixture
def service(monkeypatch):
    inpainting_service = MagicMock()
    monkeypatch.setattr(webui, "inpainting_service", inpainting_service)
    return inpainting_service


def test_prompt_adapter_constructs_command(service: MagicMock) -> None:
    webui.update_prompt_text("positive", "negative", "state")

    service.update_prompts.assert_called_once_with(
        UpdateInpaintingPrompts("state", "positive", "negative")
    )


@pytest.mark.parametrize(
    "error",
    [InpaintingUnchanged("same"), InpaintingNotReady("slice")],
)
def test_prompt_adapter_maps_domain_control_flow_to_prevent_update(
    service: MagicMock, error: Exception
) -> None:
    service.update_prompts.side_effect = error

    with pytest.raises(PreventUpdate):
        webui.update_prompt_text("positive", "negative", "state")


def test_model_adapter_constructs_command_and_maps_unchanged(
    service: MagicMock,
) -> None:
    webui.remember_inpaint_model("fake-model", "state")
    service.update_model.assert_called_once_with(
        UpdateInpaintingModel("state", "fake-model")
    )

    service.reset_mock()
    service.update_model.side_effect = InpaintingUnchanged("same")
    with pytest.raises(PreventUpdate):
        webui.remember_inpaint_model("fake-model", "state")


def test_model_adapter_maps_not_ready_to_prevent_update(service: MagicMock) -> None:
    service.update_model.side_effect = InpaintingNotReady("model")

    with pytest.raises(PreventUpdate):
        webui.remember_inpaint_model("fake-model", "state")


@pytest.mark.parametrize(
    ("backward", "forward", "index", "direction"),
    [
        ([None, 1], [None, None], 1, SliceVersionDirection.BACKWARD),
        ([None, None], [1, None], 0, SliceVersionDirection.FORWARD),
    ],
)
def test_undo_redo_adapter_constructs_exact_move_command(
    service: MagicMock,
    backward: list[int | None],
    forward: list[int | None],
    index: int,
    direction: SliceVersionDirection,
) -> None:
    assert webui.undo_slice(backward, forward, "state") is True
    service.move_slice_version.assert_called_once_with(
        MoveSliceVersion("state", index, direction)
    )


def test_undo_redo_adapter_maps_unavailable_version_to_prevent_update(
    service: MagicMock,
) -> None:
    service.move_slice_version.side_effect = SliceVersionUnavailable("end")

    with pytest.raises(PreventUpdate):
        webui.undo_slice([1], [None], "state")


def test_undo_redo_adapter_maps_not_ready_to_prevent_update(service: MagicMock) -> None:
    service.move_slice_version.side_effect = InpaintingNotReady("slice")

    with pytest.raises(PreventUpdate):
        webui.undo_slice([1], [None], "state")


def test_slice_selection_clears_stale_inpainting_candidate(
    service: MagicMock, monkeypatch
) -> None:
    state = MagicMock()
    state.selected_slice = None
    state.use_checkerboard = False
    state.image_slices = [MagicMock(positive_prompt="p", negative_prompt="n")]
    state.serve_slice_image_composed.return_value = "/selected.png"
    monkeypatch.setattr(webui.AppState, "from_cache", MagicMock(return_value=state))
    monkeypatch.setattr(webui, "ctx", MagicMock(triggered_id={"index": 0}))

    result = webui.display_slice(
        [1],
        [None],
        [{"index": 0}],
        ["/slice.png"],
        ["hidden"],
        "state",
    )

    service.clear_selection.assert_called_once_with(ClearInpaintingSelection("state"))
    assert result == ("/selected.png", ["overlay"], "p", "n", True)


def test_slice_selection_maps_clear_error_to_prevent_update(
    service: MagicMock, monkeypatch
) -> None:
    state = MagicMock()
    state.selected_slice = None
    state.use_checkerboard = False
    state.image_slices = [MagicMock(positive_prompt="p", negative_prompt="n")]
    state.serve_slice_image_composed.return_value = "/selected.png"
    monkeypatch.setattr(webui.AppState, "from_cache", MagicMock(return_value=state))
    monkeypatch.setattr(webui, "ctx", MagicMock(triggered_id={"index": 0}))
    service.clear_selection.side_effect = InpaintingNotReady("state")

    with pytest.raises(PreventUpdate):
        webui.display_slice(
            [1],
            [None],
            [{"index": 0}],
            ["/slice.png"],
            ["hidden"],
            "state",
        )
