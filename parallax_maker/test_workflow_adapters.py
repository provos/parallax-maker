from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock

from dash.exceptions import PreventUpdate
from PIL import Image
import pytest

from . import webui
from .workflow_services import (
    ConfigureThresholds,
    ConfigureThresholdsResult,
    GenerateDepth,
    GenerateDepthResult,
    GenerateSlices,
    GenerateSlicesResult,
    UpdateThresholdValues,
    UpdateThresholdValuesResult,
    UploadImage,
    UploadImageResult,
    WorkflowNotReady,
    WorkflowUnchanged,
)


@pytest.fixture
def workflow_service(monkeypatch):
    service = MagicMock()
    monkeypatch.setattr(webui, "workflow_service", service)
    return service


def _image_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def test_upload_callback_constructs_command_and_translates_result(
    workflow_service, monkeypatch
) -> None:
    workflow_service.upload_image.return_value = UploadImageResult("appstate-new")
    state = MagicMock()
    state.serve_input_image.return_value = "/tmp-images/appstate-new/input.png"
    from_cache = MagicMock(return_value=state)
    monkeypatch.setattr(webui.AppState, "from_cache", from_cache)
    contents = _image_data_url(Image.new("RGB", (3, 2), (12, 34, 56)))

    result = webui.update_input_image(contents, ["visible", "hidden"])

    command = workflow_service.upload_image.call_args.args[0]
    assert isinstance(command, UploadImage)
    assert command.image.size == (3, 2)
    assert command.image.convert("RGB").getpixel((0, 0)) == (12, 34, 56)
    from_cache.assert_called_once_with("appstate-new")
    assert result[:4] == (
        "appstate-new",
        True,
        True,
        "/tmp-images/appstate-new/input.png",
    )
    assert result[4].id == "depthmap-image"
    assert result[5] is False


def test_upload_callback_translates_not_ready_to_prevent_update(
    workflow_service,
) -> None:
    workflow_service.upload_image.side_effect = WorkflowNotReady("input image")
    contents = _image_data_url(Image.new("RGB", (3, 2), "black"))

    with pytest.raises(PreventUpdate):
        webui.update_input_image(contents, ["visible", "hidden"])


def test_depth_callback_constructs_command_and_translates_result(
    workflow_service,
) -> None:
    workflow_service.generate_depth.return_value = GenerateDepthResult("appstate-test")

    result = webui.generate_depth_map_callback(True, "appstate-test", "dinov2")

    workflow_service.generate_depth.assert_called_once_with(
        GenerateDepth(state_id="appstate-test", model_name="dinov2")
    )
    assert result == (True, "")


def test_depth_callback_translates_not_ready_to_prevent_update(
    workflow_service,
) -> None:
    workflow_service.generate_depth.side_effect = WorkflowNotReady("input image")

    with pytest.raises(PreventUpdate):
        webui.generate_depth_map_callback(True, "appstate-test", "dinov2")


def test_configure_thresholds_callback_constructs_command_and_appends_logs(
    workflow_service,
) -> None:
    workflow_service.configure_thresholds.return_value = ConfigureThresholdsResult(
        state_id="appstate-test",
        thresholds=[0, 63, 127, 191, 255],
        missing_depth=True,
    )
    logs = ["before"]

    result = webui.update_thresholds(None, 4, "appstate-test", logs)

    workflow_service.configure_thresholds.assert_called_once_with(
        ConfigureThresholds(state_id="appstate-test", num_slices=4)
    )
    assert result == (
        True,
        [
            "before",
            "No depth map data available",
            "Thresholds: [0, 63, 127, 191, 255]",
        ],
    )


@pytest.mark.parametrize(
    "error",
    [WorkflowUnchanged("unchanged"), WorkflowNotReady("positive slice count")],
)
def test_configure_thresholds_callback_translates_service_control_flow(
    workflow_service, error
) -> None:
    workflow_service.configure_thresholds.side_effect = error

    with pytest.raises(PreventUpdate):
        webui.update_thresholds(None, 4, "appstate-test", [])


def test_update_thresholds_callback_constructs_command_without_preview(
    workflow_service,
) -> None:
    workflow_service.update_threshold_values.return_value = UpdateThresholdValuesResult(
        state_id="appstate-test", values=[80, 150], preview_image=None
    )

    result = webui.update_threshold_values([80, 150], 3, "appstate-test")

    workflow_service.update_threshold_values.assert_called_once_with(
        UpdateThresholdValues(state_id="appstate-test", values=[80, 150], num_slices=3)
    )
    assert result == ([80, 150], webui.no_update)


def test_update_thresholds_callback_serves_service_preview(
    workflow_service, monkeypatch
) -> None:
    preview = Image.new("RGB", (4, 3), "purple")
    workflow_service.update_threshold_values.return_value = UpdateThresholdValuesResult(
        state_id="result-state", values=[80, 150], preview_image=preview
    )
    state = MagicMock()
    state.serve_main_image.return_value = "/tmp-images/result-state/main.bmp"
    from_cache = MagicMock(return_value=state)
    monkeypatch.setattr(webui.AppState, "from_cache", from_cache)

    result = webui.update_threshold_values([80, 150], 3, "command-state")

    workflow_service.update_threshold_values.assert_called_once_with(
        UpdateThresholdValues(state_id="command-state", values=[80, 150], num_slices=3)
    )
    from_cache.assert_called_once_with("result-state")
    state.serve_main_image.assert_called_once_with(preview)
    assert result == ([80, 150], "/tmp-images/result-state/main.bmp")


@pytest.mark.parametrize(
    "error",
    [WorkflowUnchanged("unchanged"), WorkflowNotReady("thresholds")],
)
def test_update_thresholds_callback_translates_service_control_flow(
    workflow_service, error
) -> None:
    workflow_service.update_threshold_values.side_effect = error

    with pytest.raises(PreventUpdate):
        webui.update_threshold_values([80, 150], 3, "appstate-test")


def test_generate_slices_callback_constructs_command_and_translates_result(
    workflow_service,
) -> None:
    workflow_service.generate_slices.return_value = GenerateSlicesResult(
        state_id="appstate-test", slice_count=3
    )

    result = webui.generate_slices(True, "appstate-test")

    workflow_service.generate_slices.assert_called_once_with(
        GenerateSlices(state_id="appstate-test")
    )
    assert result == (True, "")


def test_generate_slices_callback_translates_not_ready_to_prevent_update(
    workflow_service,
) -> None:
    workflow_service.generate_slices.side_effect = WorkflowNotReady("thresholds")

    with pytest.raises(PreventUpdate):
        webui.generate_slices(True, "appstate-test")
