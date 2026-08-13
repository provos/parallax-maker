from __future__ import annotations

from unittest.mock import MagicMock

import dash
from dash.exceptions import PreventUpdate
import pytest

from .components import make_segmentation_callbacks
from .segmentation_services import (
    MultiPointModeResult,
    SegmentationNotReady,
    SetMultiPointMode,
)


def register_toggle_callback(service):
    app = dash.Dash(__name__)
    make_segmentation_callbacks(app, service)
    matching = [
        entry["callback"]
        for key, entry in app.callback_map.items()
        if "multi-point.className" in key
    ]
    assert len(matching) == 1
    return matching[0].__wrapped__


def test_segmentation_callback_registration_requires_a_service() -> None:
    with pytest.raises(TypeError, match="segmentation_service"):
        make_segmentation_callbacks(dash.Dash(__name__))


def test_toggle_multi_point_enables_service_and_preserves_adapter_outputs() -> None:
    service = MagicMock()
    service.set_multi_point_mode.return_value = MultiPointModeResult(
        state_id="appstate-test", enabled=True, cleared_points=2
    )
    toggle = register_toggle_callback(service)

    result = toggle(
        1,
        "appstate-test",
        "button color-not-selected hover:color-not-selected-light",
    )

    service.set_multi_point_mode.assert_called_once_with(
        SetMultiPointMode(state_id="appstate-test", enabled=True)
    )
    assert result == (
        "button color-is-selected hover:color-is-selected-light",
        True,
    )


def test_toggle_multi_point_disables_service_and_preserves_adapter_outputs() -> None:
    service = MagicMock()
    service.set_multi_point_mode.return_value = MultiPointModeResult(
        state_id="appstate-test", enabled=False, cleared_points=3
    )
    toggle = register_toggle_callback(service)

    result = toggle(2, "appstate-test", "button color-is-selected")

    service.set_multi_point_mode.assert_called_once_with(
        SetMultiPointMode(state_id="appstate-test", enabled=False)
    )
    assert result == ("button color-not-selected", True)


def test_toggle_multi_point_translates_domain_error_to_prevent_update() -> None:
    service = MagicMock()
    service.set_multi_point_mode.side_effect = SegmentationNotReady("state")
    toggle = register_toggle_callback(service)

    with pytest.raises(PreventUpdate):
        toggle(1, "appstate-test", "button color-not-selected")

    service.set_multi_point_mode.assert_called_once_with(
        SetMultiPointMode(state_id="appstate-test", enabled=True)
    )
