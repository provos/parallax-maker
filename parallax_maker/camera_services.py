"""Framework-neutral service for manual 3D-camera navigation.

Reproduces the removed Dash `navigate_image` callback (`components.py`'s
`make_navigation_callbacks`, see ``git show 1904e1a:parallax_maker/
components.py`` for the exact source, ~line 1853) over HTTP: nudging
`AppState.camera.camera_position` by a fixed unit step (or resetting it to
`[0, 0, -camera_distance]`), deselecting the current slice, and re-rendering
the composited 3D-ish preview via `segmentation.render_view`.

Unlike most mutation services in this package, Dash's own callback never
persisted this to disk (no `state.to_file(...)` call anywhere in it) - the
camera position lives purely in the cached `AppState`, so this module has no
`StateRepository`/save step either; the caller (the HTTP route) is expected
to have already loaded the project's `AppState` (e.g. via `_load_state`).

Dash's original body also mutates in place (`camera_position +=
switch[nav_clicked]`, where `camera_position` is the *same* array object as
`state.camera.camera_position`, not a copy) before reassigning it through the
setter. That in-place mutation is not reproduced here: every branch below
builds a fresh `float32` array of shape `(3,)` and assigns it through
`Camera.camera_position`'s own setter, which is the only supported way to
change it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .controller import AppState
from .segmentation import render_view

#: The seven values `navigate_image` accepted, one per `NAV_*` button.
Direction = Literal["up", "down", "left", "right", "in", "out", "reset"]

#: Mirrors the Dash `switch` dict in `navigate_image`: a unit step per click
#: along each axis (image-space x right, y down) plus a z dolly. "in"/"out"
#: here are `NAV_ZOOM_IN`/`NAV_ZOOM_OUT`.
_STEPS: dict[str, tuple[float, float, float]] = {
    "up": (0.0, -1.0, 0.0),
    "down": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "out": (0.0, 0.0, -1.0),
    "in": (0.0, 0.0, 1.0),
}


class CameraServiceError(Exception):
    """Base class for camera-navigation domain failures."""


class InvalidNavigationDirection(CameraServiceError):
    """``direction`` is not one of the seven values `navigate_camera` accepts.

    The HTTP route validates this earlier via the request schema's
    ``Literal`` type, so this only guards direct/non-HTTP callers.
    """


@dataclass(frozen=True)
class NavigatedCameraResult:
    """Result of `navigate_camera`.

    ``preview_image`` is ``None`` exactly when there were no image slices to
    render (Dash's own early return before ever moving the camera or
    rendering) - callers should treat that as "nothing changed" and log
    "No image slices to navigate" instead of the position-changed line.
    """

    camera_position: np.ndarray
    preview_image: Image.Image | None


def navigate_camera(state: AppState, direction: Direction) -> NavigatedCameraResult:
    """Move ``state.camera``'s position and re-render the 3D-ish preview.

    Mirrors ``navigate_image`` byte-for-byte: deselects the current slice
    first (matching its unconditional `state.selected_slice = None`), then -
    if there is nothing to render - returns immediately with
    `preview_image=None`. Otherwise either resets the camera to
    `[0, 0, -camera_distance]` (`direction == "reset"`) or steps it by a
    fixed unit vector, then renders the composited view with
    `segmentation.render_view` using `state.camera_matrix()`/
    `state.get_cards()`, exactly like Dash's own call.
    """

    state.selected_slice = None
    if len(state.image_slices) == 0:
        return NavigatedCameraResult(
            camera_position=state.camera.camera_position, preview_image=None
        )

    if direction == "reset":
        camera_position = np.array(
            [0.0, 0.0, -state.camera.camera_distance], dtype=np.float32
        )
    elif direction in _STEPS:
        step = np.array(_STEPS[direction], dtype=np.float32)
        # A fresh array via `+` (not `+=`): see the module docstring on why
        # Dash's own in-place mutation is intentionally not reproduced.
        camera_position = state.camera.camera_position + step
        camera_position = camera_position.astype(np.float32, copy=False)
    else:
        raise InvalidNavigationDirection(f"invalid navigation direction: {direction!r}")

    state.camera.camera_position = camera_position
    camera_matrix = state.camera_matrix()
    card_corners_3d_list = state.get_cards()
    rendered = render_view(
        state.image_slices,
        camera_matrix,
        card_corners_3d_list,
        camera_position,
        camera_rotation=state.camera.rotation_world_to_camera(),
    )
    preview_image = Image.fromarray(rendered)

    return NavigatedCameraResult(
        camera_position=camera_position, preview_image=preview_image
    )
