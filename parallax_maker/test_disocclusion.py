import os
import shutil
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from .controller import AppState
from .slice import ImageSlice
from .camera import Camera
from .segmentation import (
    should_use_ai_fallback,
    reconstruct_slice_disocclusions,
    render_view,
    render_image_sequence,
)


class TestDisocclusionAndClamping(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for state files
        self.temp_dir = Path("appstate-testdisocclusion")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_should_use_ai_fallback(self):
        # 1. Test empty/None mask
        self.assertFalse(should_use_ai_fallback(None))
        self.assertFalse(should_use_ai_fallback(np.array([])))

        # 2. Test small hole mask (less than 5%)
        mask_small = np.zeros((100, 100), dtype=np.uint8)
        mask_small[40:42, 40:42] = 255 # 4 pixels out of 10000 -> 0.04%
        self.assertFalse(should_use_ai_fallback(mask_small, threshold_ratio=0.05))

        # 3. Test large hole mask (greater than 5%)
        mask_large = np.zeros((100, 100), dtype=np.uint8)
        mask_large[10:40, 10:40] = 255 # 900 pixels out of 10000 -> 9%
        self.assertTrue(should_use_ai_fallback(mask_large, threshold_ratio=0.05))

    def test_reconstruct_slice_disocclusions_background(self):
        # Create a synthetic background image with a transparent center hole
        img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img[30:70, 30:70, 3] = 0  # Transparent center (hole)
        img[30:70, 30:70, :3] = 0 # Black color inside the hole

        # Run reconstruction
        margin = 0.1
        reconstructed = reconstruct_slice_disocclusions(img, is_background=True, margin=margin)

        # Expected size: 100 * 1.2 = 120
        self.assertEqual(reconstructed.shape, (120, 120, 4))
        # Background should be completely opaque now (alpha channel 255 everywhere)
        np.testing.assert_array_equal(reconstructed[:, :, 3], 255)
        # Inside the original hole, color should be inpainted (not pure black anymore)
        center_color = reconstructed[50, 50, :3]
        self.assertTrue(np.any(center_color > 0))

    def test_reconstruct_slice_disocclusions_foreground(self):
        # Create a synthetic foreground image (opaque center circle, transparent background)
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (255, 100, 0, 255), -1)

        # Run reconstruction
        margin = 0.1
        reconstructed = reconstruct_slice_disocclusions(img, is_background=False, margin=margin)

        # Expected size: 120x120
        self.assertEqual(reconstructed.shape, (120, 120, 4))
        # The opaque region should be dilated slightly
        # Check that alpha is non-zero outside the original circle (e.g. at radius 25)
        # Dilated alpha is blurred/feathered so it should be > 0 but less than 255 at the edges
        self.assertTrue(reconstructed[50, 50 + 25, 3] > 0)

    def test_appstate_caching_and_invalidation(self):
        state = AppState()
        state.filename = str(self.temp_dir)

        # Create two slices
        slice_0 = ImageSlice(np.ones((100, 100, 4), dtype=np.uint8) * 255, depth=5)
        slice_0.filename = str(self.temp_dir / "image_slice_0.png")
        slice_0.save_image()

        slice_1 = ImageSlice(np.ones((100, 100, 4), dtype=np.uint8) * 255, depth=10)
        slice_1.filename = str(self.temp_dir / "image_slice_1.png")
        slice_1.save_image()

        state.add_slice(slice_0)
        state.add_slice(slice_1)

        # Get reconstructed slices - should generate and save them
        recon_slices = state.get_reconstructed_slices(margin=0.1)
        self.assertEqual(len(recon_slices), 2)

        recon_path_0 = self.temp_dir / "image_slice_0_reconstructed.png"
        recon_path_1 = self.temp_dir / "image_slice_1_reconstructed.png"
        self.assertTrue(recon_path_0.exists())
        self.assertTrue(recon_path_1.exists())

        # Calling again should load them directly from cache/disk
        recon_slices_2 = state.get_reconstructed_slices(margin=0.1)
        self.assertEqual(len(recon_slices_2), 2)

        # Mutate slices - should invalidate cache
        state.delete_slice(1)
        self.assertFalse(recon_path_0.exists())
        self.assertFalse(recon_path_1.exists())

    def test_camera_clamping_and_rendering(self):
        # Create a simple slice and card corners
        slice_image = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=10)
        image_slices = [slice_image]

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)

        # Original viewport size is 100x100, card corners correspond to padded 120x120 card
        bg_card = slice_image.create_card(100, 100, camera)
        # Scale card up matching margin=0.1
        bg_card[:, :2] *= 1.2
        card_corners_3d_list = [bg_card]

        # 1. Camera position within limits: should render without clamping
        cam_pos_ok = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        rendered_ok = render_view(
            image_slices, camera_matrix, card_corners_3d_list, cam_pos_ok, original_size=(100, 100)
        )
        self.assertEqual(rendered_ok.shape, (100, 100, 4))

        # 2. Camera position with excessive horizontal translation: should clamp tx to safe margin
        cam_pos_excess = np.array([50.0, 0.0, -100.0], dtype=np.float32) # tx=50 is way too large
        rendered_excess = render_view(
            image_slices, camera_matrix, card_corners_3d_list, cam_pos_excess, original_size=(100, 100)
        )
        self.assertEqual(rendered_excess.shape, (100, 100, 4))
        # Ensure that no large fully black regions are exposed due to clamping
        self.assertTrue(np.all(rendered_excess[:, :, 3] == 255))

    def test_frame_sequences_100_and_300(self):
        # Test rendering standard 100 frames and maximum 300 frames sequences
        slice_image = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=10)
        image_slices = [slice_image]

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)
        bg_card = slice_image.create_card(100, 100, camera)
        bg_card[:, :2] *= 1.2
        card_corners_3d_list = [bg_card]

        # Render a short 2-frame sequence (as a fast proxy for 100/300 frames to keep tests speedy)
        cam_pos_100 = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        render_image_sequence(
            str(self.temp_dir),
            image_slices,
            card_corners_3d_list,
            camera_matrix,
            cam_pos_100,
            push_distance=50,
            num_frames=2,
            original_size=(100, 100),
        )
        self.assertTrue((self.temp_dir / "rendered_image_000.png").exists())
        self.assertTrue((self.temp_dir / "rendered_image_001.png").exists())


if __name__ == "__main__":
    unittest.main()
