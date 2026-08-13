from __future__ import annotations

from unittest.mock import MagicMock

from dash.exceptions import PreventUpdate
from PIL import Image
import pytest

from . import constants as C
from . import webui
from .segmentation_services import (
    AppliedMaskResult,
    CommitMultiPoint,
    InvalidSegmentationPoint,
    MaskOperation,
    PointPolarity,
    QueuedPointResult,
    SegmentationNotReady,
    SelectDepthPoint,
    SelectInstancePoint,
)


@pytest.fixture
def segmentation_adapter(monkeypatch):
    service = MagicMock()
    monkeypatch.setattr(webui, "segmentation_service", service, raising=False)
    context = MagicMock()
    monkeypatch.setattr(webui, "ctx", context)
    find_pixel = MagicMock(return_value=(12, 8))
    monkeypatch.setattr(webui, "find_pixel_from_event", find_pixel)
    state = MagicMock()
    state.imgData = Image.new("RGB", (20, 10), "black")
    state.serve_main_image.return_value = "/tmp-images/appstate-test/main.bmp"
    from_cache = MagicMock(return_value=state)
    monkeypatch.setattr(webui.AppState, "from_cache", from_cache)
    return service, context, find_pixel, state, from_cache


def applied_result(
    *,
    point=(12, 8),
    depth=140,
    operation=MaskOperation.REPLACE,
    positive_points=(),
    negative_points=(),
):
    return AppliedMaskResult(
        state_id="appstate-test",
        point=point,
        depth=depth,
        operation=operation,
        preview_image=Image.new("RGB", (20, 10), "purple"),
        positive_points=positive_points,
        negative_points=negative_points,
        selected_slice=None,
    )


@pytest.mark.parametrize(
    ("shift", "ctrl", "operation"),
    [
        (False, False, MaskOperation.REPLACE),
        (True, False, MaskOperation.ADD),
        (False, True, MaskOperation.SUBTRACT),
        (True, True, MaskOperation.ADD),
    ],
)
def test_depth_click_constructs_exact_command_with_shift_priority(
    segmentation_adapter, shift, ctrl, operation
) -> None:
    service, context, _, state, _ = segmentation_adapter
    context.triggered_id = "el"
    service.select_depth_point.return_value = applied_result(operation=operation)
    event = {"shiftKey": shift, "ctrlKey": ctrl}

    result = webui.click_event(
        None,
        None,
        event,
        {"width": 20},
        "depth",
        "appstate-test",
        ["before"],
    )

    service.select_depth_point.assert_called_once_with(
        SelectDepthPoint(state_id="appstate-test", point=(12, 8), operation=operation)
    )
    state.serve_main_image.assert_called_once_with(
        service.select_depth_point.return_value.preview_image
    )
    assert result == (
        "/tmp-images/appstate-test/main.bmp",
        ["before", "Click event at pixel coordinates (12, 8) at depth 140"],
        "",
        webui.no_update,
    )


@pytest.mark.parametrize(
    ("ctrl", "polarity"),
    [(False, PointPolarity.POSITIVE), (True, PointPolarity.NEGATIVE)],
)
def test_instance_click_constructs_exact_command_and_preserves_logs(
    segmentation_adapter, ctrl, polarity
) -> None:
    service, context, _, state, _ = segmentation_adapter
    context.triggered_id = "el"
    service.select_instance_point.return_value = applied_result(
        positive_points=((12, 8),), negative_points=()
    )
    event = {"shiftKey": False, "ctrlKey": ctrl}

    result = webui.click_event(
        None, None, event, {"width": 20}, "segment", "appstate-test", []
    )

    service.select_instance_point.assert_called_once_with(
        SelectInstancePoint(
            state_id="appstate-test",
            point=(12, 8),
            operation=MaskOperation.SUBTRACT if ctrl else MaskOperation.REPLACE,
            polarity=polarity,
        )
    )
    assert result[0] == "/tmp-images/appstate-test/main.bmp"
    assert result[1] == [
        "Click event at pixel coordinates (12, 8) at depth 140",
        "Committed points [(12, 8)] and [] for Segment Anything",
    ]
    state.serve_main_image.assert_called_once()


def test_queued_instance_click_returns_four_no_updates_except_raw_event(
    segmentation_adapter,
) -> None:
    service, context, _, state, _ = segmentation_adapter
    context.triggered_id = "el"
    service.select_instance_point.return_value = QueuedPointResult(
        state_id="appstate-test",
        point=(12, 8),
        depth=140,
        polarity=PointPolarity.NEGATIVE,
        queue_size=2,
    )
    event = {"shiftKey": False, "ctrlKey": True, "clientX": 123}

    result = webui.click_event(
        None, None, event, {"width": 20}, "segment", "appstate-test", ["before"]
    )

    assert result == (webui.no_update, webui.no_update, webui.no_update, event)
    state.serve_main_image.assert_not_called()


def test_commit_constructs_command_and_translates_exact_log(
    segmentation_adapter,
) -> None:
    service, context, find_pixel, state, _ = segmentation_adapter
    context.triggered_id = C.SEG_MULTI_COMMIT
    service.commit_multi_point.return_value = applied_result(
        point=None,
        depth=None,
        positive_points=((2, 3), (4, 5)),
        negative_points=((6, 7),),
    )

    result = webui.click_event(
        1, None, None, None, "segment", "appstate-test", ["before"]
    )

    service.commit_multi_point.assert_called_once_with(
        CommitMultiPoint(state_id="appstate-test")
    )
    find_pixel.assert_not_called()
    assert result == (
        "/tmp-images/appstate-test/main.bmp",
        [
            "before",
            "Committed points [(2, 3), (4, 5)] and [(6, 7)] for Segment Anything",
        ],
        "",
        webui.no_update,
    )
    state.serve_main_image.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        SegmentationNotReady("depth map"),
        InvalidSegmentationPoint("outside image"),
    ],
)
def test_click_adapter_translates_domain_errors_to_prevent_update(
    segmentation_adapter, error
) -> None:
    service, context, _, _, _ = segmentation_adapter
    context.triggered_id = "el"
    service.select_depth_point.side_effect = error

    with pytest.raises(PreventUpdate):
        webui.click_event(
            None,
            None,
            {"shiftKey": False, "ctrlKey": False},
            {"width": 20},
            "depth",
            "appstate-test",
            [],
        )


def test_commit_domain_error_maps_to_prevent_update(segmentation_adapter) -> None:
    service, context, _, _, _ = segmentation_adapter
    context.triggered_id = C.SEG_MULTI_COMMIT
    service.commit_multi_point.side_effect = SegmentationNotReady("points")

    with pytest.raises(PreventUpdate):
        webui.click_event(1, None, None, None, "segment", "appstate-test", [])
