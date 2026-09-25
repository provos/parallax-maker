"""Camera pitch: model, card geometry, preview rendering and glTF export.

The hard requirement: at the reference camera (default position and
orientation) the scene reproduces the original image, with or without pitch.
"""

import numpy as np
import pytest

from .camera import MAX_PITCH_DEGREES, Camera
from .gltf import camera_node_rotation
from .segmentation import render_view
from .slice import ImageSlice

W, H = 640, 480


def test_pitch_defaults_to_zero_and_round_trips_through_json():
    cam = Camera(distance=120, max_distance=400, focal_length=50, pitch=7.5)
    restored = Camera.from_json(cam.to_json())
    assert restored.pitch == 7.5
    assert restored == cam

    legacy = cam.to_json()
    del legacy["pitch"]
    assert Camera.from_json(legacy).pitch == 0.0


@pytest.mark.parametrize("bad", [MAX_PITCH_DEGREES + 1, -MAX_PITCH_DEGREES - 1, "5"])
def test_pitch_is_validated(bad):
    with pytest.raises(ValueError):
        Camera(pitch=bad)


def test_horizon_row_and_pitch_are_inverse():
    cam = Camera(focal_length=35)
    assert cam.horizon_row(W, H) == pytest.approx(H / 2)
    cam.pitch = cam.pitch_for_horizon(0.7 * H, W, H)
    assert cam.pitch > 0  # horizon below center -> looking up
    assert cam.horizon_row(W, H) == pytest.approx(0.7 * H)


def test_unpitched_cards_are_the_frame_filling_rectangles():
    cam = Camera(distance=100, max_distance=500, focal_length=100)
    card = ImageSlice(np.zeros((H, W, 4), np.uint8), 128).create_card(H, W, cam)
    z = cam.max_distance * (255 - 128) / 255.0
    half_w = W * (z + 100) / cam.focal_length_px(W) / 2
    half_h = H * (z + 100) / cam.focal_length_px(W) / 2
    expected = [
        [-half_w, -half_h, z],
        [half_w, -half_h, z],
        [half_w, half_h, z],
        [-half_w, half_h, z],
    ]
    np.testing.assert_allclose(card, expected, rtol=1e-5)


def _project(cam, points, position=None):
    """Pinhole projection through ``cam``'s orientation at ``position``."""
    position = cam.reference_position() if position is None else position
    in_camera = (
        np.asarray(points, np.float64) - position
    ) @ cam.rotation_world_to_camera().T
    fl_px = cam.focal_length_px(W)
    return np.stack(
        [
            fl_px * in_camera[:, 0] / in_camera[:, 2] + W / 2,
            fl_px * in_camera[:, 1] / in_camera[:, 2] + H / 2,
        ],
        axis=1,
    )


@pytest.mark.parametrize("pitch", [-12.0, 10.0, 25.0])
def test_pitched_cards_are_vertical_and_project_onto_the_image_corners(pitch):
    cam = Camera(distance=100, max_distance=500, focal_length=50, pitch=pitch)
    for depth in (10, 128, 250):
        card = ImageSlice(np.zeros((H, W, 4), np.uint8), depth).create_card(H, W, cam)
        # Vertical: every corner lies on the same plane z = const.
        assert np.ptp(card[:, 2]) == pytest.approx(0, abs=1e-3)
        np.testing.assert_allclose(
            _project(cam, card), [[0, 0], [W, 0], [W, H], [0, H]], atol=1e-2
        )


def _random_slices(rng, depths):
    slices = []
    for depth in depths:
        image = rng.integers(0, 256, (H, W, 4), dtype=np.uint8)
        image[..., 3] = 255
        slices.append(ImageSlice(image, depth))
    return slices


@pytest.mark.parametrize("pitch", [0.0, 12.0, -9.0])
def test_reference_view_reproduces_the_image(pitch):
    cam = Camera(distance=100, max_distance=500, focal_length=50, pitch=pitch)
    rng = np.random.default_rng(0)
    slices = _random_slices(rng, [40, 200])  # far card first, near card on top
    cards = [s.create_card(H, W, cam) for s in slices]

    rendered = render_view(
        slices,
        cam.camera_matrix(W, H),
        cards,
        cam.reference_position(),
        camera_rotation=cam.rotation_world_to_camera(),
    )

    # The nearest (opaque, frame-filling) card must come out unchanged.
    diff = np.abs(rendered[..., :3].astype(int) - slices[-1].image[..., :3].astype(int))
    assert diff.max() <= 1
    assert diff.mean() < 0.05


def test_camera_node_carries_the_pitch():
    level = camera_node_rotation(Camera())
    np.testing.assert_allclose(np.abs(level), [0, 1, 0, 0], atol=1e-9)

    def forward_elevation(quaternion):
        x, y, z, w = quaternion
        # Camera looks down its local -z; world y is up in glTF.
        forward_y = -(2 * (y * z - x * w))
        return np.degrees(np.arcsin(forward_y))

    assert forward_elevation(camera_node_rotation(Camera(pitch=15.0))) == pytest.approx(
        15.0
    )
    assert forward_elevation(camera_node_rotation(Camera(pitch=-8.0))) == pytest.approx(
        -8.0
    )
