"""The card scene as planar layers, and its renderer.

Every slice becomes one planar layer in the world frame of ``camera.py``
(x right, y down, z forward and horizontal):

* a regular slice is a *vertical card*: the plane ``z = depth`` cut by the
  reference camera's rays through the image corners;
* the ground slice (``ImageSlice.is_ground_plane``) is the *horizontal ground*
  ``y = camera height`` from the bottom edge out to the scene's max
  distance, plus - for the thin band of ground rows between that far edge
  and the horizon - an upright *backdrop* standing on the far edge. The
  ground stays bounded (friendly to 3D tools), every ground pixel still has
  a place, and parallax that far away is negligible either way.

Each layer knows the image quad it came from (its corners' projection
through the reference camera). Rendering maps that quad onto the corners'
projection through the current camera - an exact homography for a plane -
and composites the layers back to front: the ground first, then the cards
from far to near. Upright cards never overlap the ground in front of them
(their base touches it), and a card pixel that reaches below its ground
contact (a foreground rock whose slice depth is behind its bottom edge)
covered the ground in the original image, so it must stay in front. At the
reference camera every homography is the identity, so the render
reproduces the original image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera

_VERTICAL = np.array([0.0, 0.0, 1.0])
_HORIZONTAL = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class SceneLayer:
    """One slice placed in the world as a textured planar quad."""

    image: np.ndarray  # H x W x 4 slice image, alpha cleared outside the quad
    corners: np.ndarray  # 4 x 3 world corners, in image order TL, TR, BR, BL
    source_quad: np.ndarray  # 4 x 2 image points the corners came from
    normal: np.ndarray  # plane: normal . X = offset
    offset: float
    kind: str  # "card", "ground" or "backdrop" (the ground's far band)

    @property
    def is_ground(self) -> bool:
        """The ground and its backdrop are drawn behind every card."""
        return self.kind != "card"


def ground_extent(image: np.ndarray, cam: Camera) -> tuple[float, float, float]:
    """``(first_row, far_row, far_z)`` of the ground slice.

    ``far_row`` is where the ground plane reaches the scene's max distance
    ``far_z``; rows from the slice's first visible row down to ``far_row``
    belong to the backdrop, rows below it to the ground.
    """
    height, width = image.shape[:2]
    ground_y = cam.ground_height(width, height)
    far_z = float(cam.max_distance)
    far_point = np.array([[0.0, ground_y, far_z]])
    far_row = float(
        project(far_point, cam, cam.reference_position(), width, height)[0, 1]
    )
    far_row = min(far_row, float(height - 1))

    visible = np.nonzero(image[..., 3].max(axis=1) > 0)[0]
    first_row = float(visible[0]) if len(visible) else float(height)
    return first_row, far_row, far_z


def _rows_only(image: np.ndarray, top: int, bottom: int) -> np.ndarray:
    """``image`` with alpha cleared outside rows ``[top, bottom)``."""
    masked = image.copy()
    masked[:top, :, 3] = 0
    masked[bottom:, :, 3] = 0
    return masked


def ground_layers(image: np.ndarray, cam: Camera) -> list[SceneLayer]:
    """The ground (and, when its rows reach past the far edge, the backdrop)."""
    height, width = image.shape[:2]
    ground_y = cam.ground_height(width, height)
    first_row, far_row, far_z = ground_extent(image, cam)

    top = max(first_row, far_row)
    quad = np.array([[0, top], [width, top], [width, height], [0, height]], np.float32)
    corners = cam.backproject_to_plane(quad, _HORIZONTAL, ground_y, width, height)
    boundary = int(np.ceil(top))
    layers = [
        SceneLayer(
            _rows_only(image, boundary, height),
            corners,
            quad,
            _HORIZONTAL,
            ground_y,
            "ground",
        )
    ]

    if first_row < far_row:
        quad = np.array(
            [[0, first_row], [width, first_row], [width, far_row], [0, far_row]],
            np.float32,
        )
        corners = cam.backproject_to_depth(quad, far_z, width, height)
        # One extra row overlaps the ground's first row, so no seam opens
        # between the two when the camera moves (drawn after the ground).
        layers.append(
            SceneLayer(
                _rows_only(image, int(first_row), boundary + 1),
                corners,
                quad,
                _VERTICAL,
                far_z,
                "backdrop",
            )
        )
    return layers


def card_layer(image_slice, cam: Camera) -> SceneLayer:
    image = image_slice.image
    height, width = image.shape[:2]
    corners = image_slice.create_card(height, width, cam)
    source_quad = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    return SceneLayer(
        image, corners, source_quad, _VERTICAL, float(corners[0, 2]), "card"
    )


def build_layers(image_slices, cam: Camera) -> list[SceneLayer]:
    """One layer per slice; at most one slice may be the ground."""
    grounds = [s for s in image_slices if s.is_ground_plane]
    if len(grounds) > 1:
        raise ValueError("at most one slice can be the ground plane")
    layers = []
    for image_slice in image_slices:
        if image_slice.is_ground_plane:
            layers.extend(ground_layers(image_slice.image, cam))
        else:
            layers.append(card_layer(image_slice, cam))
    return layers


def project(points, cam: Camera, position, width: int, height: int) -> np.ndarray:
    """Pixel coordinates of world ``points`` seen from ``position`` with the
    camera's orientation (NaN for points not in front of the camera)."""
    relative = np.asarray(points, np.float64) - np.asarray(position, np.float64)
    in_camera = relative @ cam.rotation_world_to_camera().T
    fl_px = cam.focal_length_px(width)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fl_px * in_camera[:, 0] / in_camera[:, 2] + width / 2
        v = fl_px * in_camera[:, 1] / in_camera[:, 2] + height / 2
    projected = np.stack([u, v], axis=1)
    projected[in_camera[:, 2] <= 0] = np.nan
    return projected


def render_scene(layers: list[SceneLayer], cam: Camera, camera_position) -> np.ndarray:
    """Render the layers from ``camera_position`` (reference orientation).

    Returns an RGBA image composited over black whose alpha is the largest
    layer alpha at each pixel (as the original renderer did). A layer with
    a corner behind the camera is skipped.
    """
    height, width = layers[0].image.shape[:2]
    position = np.asarray(camera_position, np.float64)

    # Back to front: the ground (then its backdrop), then cards from far to
    # near (stable, so equal depths keep slice order).
    order = sorted(
        range(len(layers)),
        key=lambda i: (0, 0.0) if layers[i].is_ground else (1, -layers[i].offset),
    )

    out = np.zeros((height, width, 3), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    for index in order:
        layer = layers[index]
        destination = project(layer.corners, cam, position, width, height)
        if not np.all(np.isfinite(destination)):
            continue
        homography = cv2.getPerspectiveTransform(
            layer.source_quad.astype(np.float32), destination.astype(np.float32)
        )
        warped = cv2.warpPerspective(
            layer.image,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        alpha = warped[..., 3].astype(np.float32) / 255.0
        out = out * (1 - alpha[..., None]) + warped[..., :3] * alpha[..., None]
        coverage = np.maximum(coverage, alpha)

    rendered = np.zeros((height, width, 4), dtype=np.uint8)
    rendered[..., :3] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    rendered[..., 3] = np.clip(np.rint(coverage * 255), 0, 255).astype(np.uint8)
    return rendered


def render_state_view(image_slices, cam: Camera, camera_position) -> np.ndarray:
    """Convenience: build the layers and render them."""
    return render_scene(build_layers(image_slices, cam), cam, camera_position)
