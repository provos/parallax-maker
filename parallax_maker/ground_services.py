"""Ground-plane helpers for the UI: auto fit and the scene's side profile.

* ``fit_ground`` proposes the camera pitch and ground distance for a scene
  with a ground slice: the horizon goes on the ground mask's top edge (for
  the ocean, that edge *is* the horizon), and the ground meets the bottom of
  the frame at the nearest card that reaches the bottom - so that card stands
  on the ground instead of sinking into it.
* ``scene_profile`` summarizes the scene seen from the side (depths and
  heights of the camera, cards and ground) for the UI's side view.

World frame as in ``camera.py``: x right, y down, z forward (horizontal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import AppState
from .scene import ground_extent

#: A row counts as part of the ground mask's top edge when at least this
#: share of the image width is opaque there (ignores stray specks).
EDGE_ROW_COVERAGE = 0.05
#: A card "reaches the bottom" of the frame when it has opaque pixels in
#: this share of the image's bottom rows.
BOTTOM_BAND = 0.02
#: The fitted ground never starts at (or past) the max distance.
MAX_GROUND_NEAR_SHARE = 0.9


class GroundNotReady(Exception):
    """The project has no ground slice (or no image) to fit."""


@dataclass(frozen=True)
class GroundFit:
    horizon_row: float
    pitch: float
    ground_near: float


def ground_slice(state: AppState):
    return next((s for s in state.image_slices if s.is_ground_plane), None)


def _opaque(image: np.ndarray) -> np.ndarray:
    return image[..., 3] > 128


def mask_top_row(image: np.ndarray) -> float | None:
    """First row where the mask covers ``EDGE_ROW_COVERAGE`` of the width."""
    opaque = _opaque(image)
    rows = np.nonzero(opaque.sum(axis=1) >= EDGE_ROW_COVERAGE * opaque.shape[1])[0]
    return float(rows[0]) if len(rows) else None


def nearest_bottom_card_depth(state: AppState) -> float | None:
    """Card depth (z) of the nearest non-ground slice touching the bottom rows."""
    depths = []
    for image_slice in state.image_slices:
        if image_slice.is_ground_plane or image_slice.image is None:
            continue
        height = image_slice.image.shape[0]
        band = max(1, int(round(BOTTOM_BAND * height)))
        if _opaque(image_slice.image[height - band :]).any():
            depths.append(image_slice._depth_to_z(image_slice.depth, state.camera))
    return min(depths) if depths else None


def fit_ground(state: AppState) -> GroundFit:
    ground = ground_slice(state)
    if ground is None or state.imgData is None:
        raise GroundNotReady("mark a slice as the ground plane first")
    if ground.image is None:
        raise GroundNotReady("the ground slice is empty")
    width, height = state.imgData.size

    top = mask_top_row(ground.image)
    if top is None:
        raise GroundNotReady("the ground slice is empty")
    horizon = min(top, height - 2.0)
    pitch = state.camera.pitch_for_horizon(horizon, width, height)

    nearest = nearest_bottom_card_depth(state)
    ground_near = 0.0 if nearest is None else float(nearest)
    ground_near = min(ground_near, MAX_GROUND_NEAR_SHARE * state.camera.max_distance)
    return GroundFit(horizon_row=horizon, pitch=pitch, ground_near=ground_near)


@dataclass(frozen=True)
class ProfileCard:
    index: int
    z: float
    top: float  # world y of the card's top edge (y down)
    bottom: float


@dataclass(frozen=True)
class ProfileGround:
    height: float  # world y of the ground (camera at y = 0)
    near_z: float
    far_z: float
    backdrop_top: float | None  # world y of the backdrop's top, if any


@dataclass(frozen=True)
class SceneProfile:
    camera_z: float
    pitch: float
    half_fov: float  # vertical half field of view, degrees
    cards: list[ProfileCard]
    ground: ProfileGround | None


def scene_profile(state: AppState) -> SceneProfile | None:
    """The scene seen from the side (x dropped); None without an image."""
    if state.imgData is None:
        return None
    width, height = state.imgData.size
    cam = state.camera
    half_fov = float(np.degrees(np.arctan((height / 2) / cam.focal_length_px(width))))

    cards = []
    ground = None
    for index, image_slice in enumerate(state.image_slices):
        if image_slice.image is None:
            continue
        if image_slice.is_ground_plane:
            try:
                ground_y = cam.ground_height(width, height)
            except ValueError:
                continue
            first_row, far_row, far_z = ground_extent(image_slice.image, cam)
            near = cam.backproject_to_depth(
                [[width / 2, height]], cam.ground_near, width, height
            )
            backdrop_top = None
            if first_row < far_row:
                backdrop_top = float(
                    cam.backproject_to_depth(
                        [[width / 2, first_row]], far_z, width, height
                    )[0, 1]
                )
            ground = ProfileGround(
                height=float(ground_y),
                near_z=float(near[0, 2]),
                far_z=float(far_z),
                backdrop_top=backdrop_top,
            )
            continue
        corners = image_slice.create_card(height, width, cam)
        cards.append(
            ProfileCard(
                index=index,
                z=float(corners[0, 2]),
                top=float(corners[:, 1].min()),
                bottom=float(corners[:, 1].max()),
            )
        )
    return SceneProfile(
        camera_z=float(cam.reference_position()[2]),
        pitch=float(cam.pitch),
        half_fov=half_fov,
        cards=cards,
        ground=ground,
    )
