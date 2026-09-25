"""Dash adapter contract tests for the framework-neutral inpainting service."""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock

import dash
from dash import html
from dash.exceptions import PreventUpdate
from PIL import Image
import pytest

from . import components
from .inpainting_services import (
    ApplyInpaintingCandidate,
    AppliedInpaintingCandidateResult,
    ClearInpaintingSelection,
    DeleteInpaintingMask,
    EraseInpainting,
    ErasedInpaintingResult,
    GenerateInpaintingCandidates,
    GeneratedInpaintingCandidatesResult,
    InpaintingMode,
    InpaintingNotReady,
    LoadInpaintingMask,
    LoadedInpaintingMaskResult,
    SaveInpaintingMask,
    SavedInpaintingMaskResult,
    SelectInpaintingCandidate,
    SelectedInpaintingCandidateResult,
)


def image_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def callback_named(app: dash.Dash, name: str):
    matches = []
    for entry in app.callback_map.values():
        callback = entry.get("callback")
        if callback is None:
            continue
        target = getattr(callback, "__wrapped__", callback)
        if target.__name__ == name:
            matches.append(target)
    assert (
        len(matches) == 1
    ), f"expected one callback named {name}, found {len(matches)}"
    return matches[0]


def register_inpainting_callbacks(service: MagicMock) -> dash.Dash:
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div()
    components.make_inpainting_container_callbacks(app, service)
    return app


def register_canvas_callbacks(service: MagicMock) -> dash.Dash:
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div()
    components.make_canvas_callbacks(app, service)
    return app


def test_inpainting_and_canvas_callback_registration_require_a_service() -> None:
    with pytest.raises(TypeError, match="inpainting_service"):
        components.make_inpainting_container_callbacks(dash.Dash(__name__))
    with pytest.raises(TypeError, match="inpainting_service"):
        components.make_canvas_callbacks(dash.Dash(__name__))


@pytest.mark.parametrize(
    ("triggered_id", "clicks", "mode", "candidate_count"),
    [
        (
            components.C.BTN_GENERATE_INPAINTING,
            (1, None, None),
            InpaintingMode.PAINT,
            3,
        ),
        (components.C.BTN_FILL_INPAINTING, (None, 1, None), InpaintingMode.FILL, 3),
        (components.C.BTN_ENHANCE, (None, None, 1), InpaintingMode.ENHANCE, 2),
    ],
)
def test_generate_adapter_constructs_command_and_renders_service_candidates(
    monkeypatch,
    triggered_id: str,
    clicks: tuple[int | None, int | None, int | None],
    mode: InpaintingMode,
    candidate_count: int,
) -> None:
    service = MagicMock()
    candidates = tuple(
        Image.new("RGBA", (5, 4), (index + 1, 2, 3, 255))
        for index in range(candidate_count)
    )
    service.generate_candidates.return_value = GeneratedInpaintingCandidatesResult(
        "state", 1, mode, candidates
    )
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "update_inpainting_image_display")
    context = MagicMock(triggered_id=triggered_id)
    monkeypatch.setattr(components, "ctx", context)
    workflow = "data:application/json;base64," + base64.b64encode(b"workflow").decode()

    children, loading, logs = callback(
        *clicks,
        "state",
        "comfyui",
        workflow,
        "positive",
        "negative",
        0.72,
        6.5,
        11,
        4,
        ["before"],
    )

    service.generate_candidates.assert_called_once_with(
        GenerateInpaintingCandidates(
            state_id="state",
            mode=mode,
            model_name="comfyui",
            workflow=b"workflow",
            positive_prompt="positive",
            negative_prompt="negative",
            strength=0.72,
            guidance_scale=6.5,
            padding=11,
            blur=4,
        )
    )
    assert loading == []
    assert logs == ["before"]
    assert len(children) == candidate_count
    assert [child.id["index"] for child in children] == list(range(candidate_count))
    assert all(child.src.startswith("data:image/png;base64,") for child in children)


def test_generate_adapter_logs_service_error_without_clearing_candidates(
    monkeypatch,
) -> None:
    service = MagicMock()
    service.generate_candidates.side_effect = InpaintingNotReady("mask")
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "update_inpainting_image_display")
    monkeypatch.setattr(
        components,
        "ctx",
        MagicMock(triggered_id=components.C.BTN_GENERATE_INPAINTING),
    )

    assert callback(
        1,
        None,
        None,
        "state",
        "fake-model",
        None,
        "",
        "",
        0.8,
        7.5,
        10,
        5,
        ["before"],
    ) == (components.no_update, [], ["before", "mask"])


def test_generate_adapter_ignores_workflow_for_non_comfyui_model(
    monkeypatch,
) -> None:
    service = MagicMock()
    service.generate_candidates.return_value = GeneratedInpaintingCandidatesResult(
        "state", 1, InpaintingMode.PAINT, ()
    )
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "update_inpainting_image_display")
    monkeypatch.setattr(
        components,
        "ctx",
        MagicMock(triggered_id=components.C.BTN_GENERATE_INPAINTING),
    )

    assert callback(
        1,
        None,
        None,
        "state",
        "fake-model",
        "malformed workflow upload",
        "",
        "",
        0.8,
        7.5,
        10,
        5,
        [],
    ) == ([], [], [])
    assert service.generate_candidates.call_args.args[0].workflow is None


def test_generate_adapter_logs_malformed_comfyui_workflow(monkeypatch) -> None:
    service = MagicMock()
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "update_inpainting_image_display")
    monkeypatch.setattr(
        components,
        "ctx",
        MagicMock(triggered_id=components.C.BTN_GENERATE_INPAINTING),
    )

    assert callback(
        1,
        None,
        None,
        "state",
        "comfyui",
        "malformed workflow upload",
        "",
        "",
        0.8,
        7.5,
        10,
        5,
        [],
    ) == (components.no_update, [], ["Invalid ComfyUI workflow data"])
    service.generate_candidates.assert_not_called()


@pytest.mark.parametrize(
    ("children", "classnames", "selected", "disabled"),
    [
        ([], [], 0, True),
        ([html.Img()], ["candidate"], None, True),
        ([html.Img()], ["candidate"], 0, True),
        ([html.Img()], ["candidate color-is-selected-light"], 0, False),
        ([html.Img()], ["candidate color-is-selected-light"], None, True),
        (
            [html.Img(), html.Img()],
            ["candidate", "candidate color-is-selected-light"],
            1,
            False,
        ),
    ],
)
def test_apply_is_enabled_only_when_a_candidate_is_visually_selected(
    monkeypatch, children, classnames, selected, disabled: bool
) -> None:
    service = MagicMock()
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "enable_apply_inpainting_button")
    monkeypatch.setattr(
        components.AppState,
        "from_cache",
        MagicMock(return_value=MagicMock(selected_inpainting=selected)),
    )

    assert callback(children, classnames, "state") is disabled


def test_candidate_selection_toggles_preview_and_css_through_service(
    monkeypatch,
) -> None:
    service = MagicMock()
    service.select_candidate.return_value = SelectedInpaintingCandidateResult(
        "state", 1
    )
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "select_inpainting_image")
    monkeypatch.setattr(components, "ctx", MagicMock(triggered_id={"index": 1}))
    images = [
        image_data_url(Image.new("RGBA", (2, 2), (1, 2, 3, 255))),
        image_data_url(Image.new("RGBA", (2, 2), (4, 5, 6, 255))),
    ]

    preview, classnames, loading = callback(
        [None, 1], "state", images, ["candidate", "candidate"]
    )

    service.select_candidate.assert_called_once_with(
        SelectInpaintingCandidate("state", 1, 2)
    )
    assert preview == images[1]
    assert classnames == ["candidate", "candidate color-is-selected-light"]
    assert loading == ""


def test_candidate_deselection_restores_composed_preview_and_clears_css(
    monkeypatch,
) -> None:
    service = MagicMock()
    service.select_candidate.return_value = SelectedInpaintingCandidateResult(
        "state", None
    )
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "select_inpainting_image")
    monkeypatch.setattr(components, "ctx", MagicMock(triggered_id={"index": 1}))
    state = MagicMock(selected_slice=2, use_checkerboard=True)
    state.serve_slice_image_composed.return_value = "/composed.png"
    monkeypatch.setattr(
        components.AppState, "from_cache", MagicMock(return_value=state)
    )

    result = callback(
        [None, 2],
        "state",
        ["/zero.png", "/one.png"],
        ["candidate", "candidate color-is-selected-light"],
    )

    assert result == ("/composed.png", ["candidate", "candidate"], "")
    state.serve_slice_image_composed.assert_called_once_with(
        2, components.CompositeMode.CHECKERBOARD
    )


def test_candidate_selection_domain_error_maps_to_prevent_update(monkeypatch) -> None:
    service = MagicMock()
    service.select_candidate.side_effect = InpaintingNotReady("state")
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "select_inpainting_image")
    monkeypatch.setattr(components, "ctx", MagicMock(triggered_id={"index": 0}))

    with pytest.raises(PreventUpdate):
        callback([1], "state", ["/candidate.png"], ["candidate"])


@pytest.mark.parametrize(
    ("filename", "selected_slice", "expected"),
    [
        (None, None, (True, True, True, True, True, True, None, [])),
        ("state", None, (True, True, True, True, True, True, None, [])),
        ("state", 2, (False, False, False, False, False, False, 2, [])),
    ],
)
def test_selected_slice_reaction_clears_selection_and_candidate_children(
    monkeypatch, filename, selected_slice, expected
) -> None:
    service = MagicMock()
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "react_selected_slice_change")
    monkeypatch.setattr(
        components.AppState,
        "from_cache",
        MagicMock(return_value=MagicMock(selected_slice=selected_slice)),
    )

    assert callback(True, filename) == expected
    if filename is None:
        service.clear_selection.assert_not_called()
    else:
        service.clear_selection.assert_called_once_with(
            ClearInpaintingSelection(filename)
        )


def test_selected_slice_reaction_maps_clear_error_to_prevent_update(
    monkeypatch,
) -> None:
    service = MagicMock()
    service.clear_selection.side_effect = InpaintingNotReady("state")
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "react_selected_slice_change")
    monkeypatch.setattr(
        components.AppState,
        "from_cache",
        MagicMock(return_value=MagicMock(selected_slice=1)),
    )

    with pytest.raises(PreventUpdate):
        callback(True, "state")


def test_apply_adapter_decodes_candidates_and_emits_one_log_entry() -> None:
    service = MagicMock()
    service.apply_candidate.return_value = AppliedInpaintingCandidateResult(
        "state", 1, "slice_v1.png"
    )
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "apply_inpainting")
    sources = [
        image_data_url(Image.new("RGBA", (3, 2), (1, 2, 3, 255))),
        image_data_url(Image.new("RGBA", (3, 2), (4, 5, 6, 255))),
    ]

    result = callback(1, "state", sources, ["before"])

    command = service.apply_candidate.call_args.args[0]
    assert isinstance(command, ApplyInpaintingCandidate)
    assert len(command.candidates) == 2
    assert [candidate.getpixel((0, 0)) for candidate in command.candidates] == [
        (1, 2, 3, 255),
        (4, 5, 6, 255),
    ]
    assert result == (
        True,
        ["before", "Inpainting applied to slice 1 with new image slice_v1.png"],
        True,
    )


def test_apply_domain_error_is_logged_without_updating_slice() -> None:
    service = MagicMock()
    service.apply_candidate.side_effect = InpaintingNotReady("selection")
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "apply_inpainting")

    assert callback(
        1,
        "state",
        [image_data_url(Image.new("RGBA", (2, 2)))],
        ["before"],
    ) == (
        components.no_update,
        ["before", "selection"],
        components.no_update,
    )


def test_apply_invalid_candidate_transport_is_logged() -> None:
    service = MagicMock()
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "apply_inpainting")

    assert callback(1, "state", ["not an image"], []) == (
        components.no_update,
        ["Invalid inpainting candidate image"],
        components.no_update,
    )
    service.apply_candidate.assert_not_called()


def test_erase_adapter_emits_exactly_one_log_entry() -> None:
    service = MagicMock()
    service.erase.return_value = ErasedInpaintingResult("state", 1, "slice_v1.png")
    app = register_inpainting_callbacks(service)
    callback = callback_named(app, "erase_inpainting")

    result = callback(1, "state", ["before"])

    service.erase.assert_called_once_with(EraseInpainting("state"))
    assert result == (True, ["before", "Inpainting erased for slice 1"])


def test_canvas_save_delete_and_load_translate_transport_at_the_boundary() -> None:
    service = MagicMock()
    service.save_mask.return_value = SavedInpaintingMaskResult(
        "state", 1, "slice_mask.png", (1, 2, 5, 6)
    )
    service.delete_mask.return_value = MagicMock(deleted=True, slice_index=1)
    service.load_mask.return_value = LoadedInpaintingMaskResult(
        "state", 1, Image.new("L", (3, 2), 127)
    )
    app = register_canvas_callbacks(service)
    save_callback = callback_named(app, "save_slice_mask")
    load_callback = callback_named(app, "load_canvas_mask")
    source = Image.new("RGBA", (3, 2), (99, 88, 77, 66))

    assert save_callback(image_data_url(source), "state", 12, ["crop"], ["before"]) == (
        (1, 2, 5, 6),
        ["before", "Saved mask for slice 1 to slice_mask.png"],
    )
    save_command = service.save_mask.call_args.args[0]
    assert isinstance(save_command, SaveInpaintingMask)
    assert save_command.padding == 12
    assert save_command.show_crop_region is True
    assert save_command.canvas_image.getpixel((0, 0)) == (99, 88, 77, 66)

    assert save_callback("", "state", 12, ["crop"], []) == (
        components.no_update,
        ["Deleted mask for slice 1"],
    )
    service.delete_mask.assert_called_once_with(DeleteInpaintingMask("state"))

    loaded_url, logs = load_callback(1, "state", [])
    service.load_mask.assert_called_once_with(LoadInpaintingMask("state"))
    loaded = Image.open(io.BytesIO(base64.b64decode(loaded_url.split(",")[1])))
    assert loaded.mode == "RGBA"
    assert loaded.getpixel((0, 0)) == (127, 0, 0, 127)
    assert logs == ["Loading mask for slice 1"]


def test_canvas_save_logs_service_and_transport_errors() -> None:
    service = MagicMock()
    app = register_canvas_callbacks(service)
    save_callback = callback_named(app, "save_slice_mask")

    service.save_mask.side_effect = InpaintingNotReady("no slice is selected")
    assert save_callback(
        image_data_url(Image.new("RGBA", (2, 2))), "state", 12, ["crop"], []
    ) == (components.no_update, ["no slice is selected"])

    service.save_mask.reset_mock(side_effect=True)
    assert save_callback("not an image", "state", 12, ["crop"], []) == (
        components.no_update,
        ["Invalid canvas mask data"],
    )
