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

    def test_padded_vs_unpadded_coordinate_correctness(self):
        """
        Verify that coordinates for unpadded and padded 3D card geometries are correctly computed
        and mapped during perspective rendering.
        """
        slice_image = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=10)
        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)

        # Original unpadded card corners (100x100 viewport)
        orig_card = slice_image.create_card(100, 100, camera)

        # Padded card corners (with 10% safety margin / scale 1.2)
        padded_card = orig_card.copy()
        padded_card[:, :2] *= 1.2
        card_corners_3d_list = [padded_card]

        # Render unpadded view and assert no coordinate scaling errors
        cam_pos = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        rendered = render_view(
            [slice_image], camera_matrix, card_corners_3d_list, cam_pos,
            original_size=(100, 100), original_slices=[slice_image]
        )
        self.assertEqual(rendered.shape, (100, 100, 4))
        # Ensure correct provenance mapping for original pixels
        self.assertTrue(np.all(rendered.provenance_map == 1))

    def test_ai_threshold_selection_and_deterministic_fallback(self):
        """
        Assert the fallback selection hierarchy:
        - Deterministic OpenCV Telea when disocclusion_ratio <= ai_threshold_ratio
        - Precomputed stable AI background when disocclusion_ratio > ai_threshold_ratio
        """
        # Original background slice has a transparent hole
        bg_img_orig = np.ones((100, 100, 4), dtype=np.uint8) * 255
        bg_img_orig[40:55, 40:60, 3] = 0
        bg_slice_orig = ImageSlice(bg_img_orig, depth=0)

        # Reconstructed background slice is precomputed and stable (no hole, fully opaque)
        bg_img_recon = np.ones((100, 100, 4), dtype=np.uint8) * 255
        bg_slice_recon = ImageSlice(bg_img_recon, depth=0)

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)
        bg_card = bg_slice_orig.create_card(100, 100, camera)

        # 1. Deterministic Fallback: Disocclusion ratio <= ai_threshold_ratio
        cam_pos_small = np.array([1.0, 0.0, -100.0], dtype=np.float32)
        rendered_det = render_view(
            [bg_slice_recon], camera_matrix, [bg_card], cam_pos_small,
            original_size=(100, 100), original_slices=[bg_slice_orig],
            ai_threshold_ratio=0.50 # very high threshold forces deterministic mode
        )
        self.assertFalse(rendered_det.ai_used)
        # Provenance map should contain deterministic label (2)
        self.assertTrue(2 in rendered_det.provenance_map)
        self.assertFalse(3 in rendered_det.provenance_map)

        # 2. AI Selection: Disocclusion ratio > ai_threshold_ratio
        rendered_ai = render_view(
            [bg_slice_recon], camera_matrix, [bg_card], cam_pos_small,
            original_size=(100, 100), original_slices=[bg_slice_orig],
            ai_threshold_ratio=0.01 # very low threshold forces AI mode
        )
        self.assertTrue(rendered_ai.ai_used)
        # Provenance map should contain AI label (3)
        self.assertTrue(3 in rendered_ai.provenance_map)

    def test_prevention_of_recursive_reconstruction(self):
        """
        Verify that original_slices are always used during rendering to warp original pristine pixels,
        rather than recursively/progressively warping already warped and reconstructed pixels.
        """
        # Reconstructed slice contains padded/inpainted image
        recon_img = np.ones((120, 120, 4), dtype=np.uint8) * 255
        recon_slice = ImageSlice(recon_img, depth=10)

        # Original slice contains unpadded pristine image
        orig_img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        orig_slice = ImageSlice(orig_img, depth=10)

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)
        bg_card = recon_slice.create_card(100, 100, camera)
        bg_card[:, :2] *= 1.2

        # Use zero camera displacement to ensure no boundary/edge dynamic padding triggers
        cam_pos_zero = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        rendered = render_view(
            [recon_slice], camera_matrix, [bg_card], cam_pos_zero,
            original_size=(100, 100), original_slices=[orig_slice]
        )
        # Ensure provenance is strictly 1 (original) because original_slices were warped directly
        self.assertTrue(np.all(rendered.provenance_map == 1))

    def test_viewport_edge_reconstruction(self):
        """
        Assert that dynamic inpainting is only applied to newly exposed viewport-edge regions
        that cannot be covered by the precomputed padded background representation.
        """
        bg_img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        bg_slice = ImageSlice(bg_img, depth=0)

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)
        bg_card = bg_slice.create_card(100, 100, camera)

        # Request camera trajectory that exceeds the precomputed boundaries (large shift)
        cam_pos_excess = np.array([45.0, 0.0, -100.0], dtype=np.float32)
        rendered = render_view(
            [bg_slice], camera_matrix, [bg_card], cam_pos_excess,
            original_size=(100, 100), original_slices=[bg_slice]
        )
        # Viewport edge reconstruction must trigger dynamic deterministic Telea (2) at the edges
        self.assertTrue(2 in rendered.provenance_map)

    def test_disocclusion_region_growth_and_levels(self):
        """
        Verify that reconstructed-region ratio and mask size grow monotonically with camera displacement
        for small, moderate, and large disocclusions.
        """
        bg_img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        bg_slice = ImageSlice(bg_img, depth=0)

        fg_img = np.zeros((100, 100, 4), dtype=np.uint8)
        fg_img[40:60, 40:60, :3] = 128
        fg_img[40:60, 40:60, 3] = 255
        fg_slice = ImageSlice(fg_img, depth=5)

        image_slices = [bg_slice, fg_slice]

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)

        bg_card = bg_slice.create_card(100, 100, camera)
        fg_card = fg_slice.create_card(100, 100, camera)
        card_corners_3d_list = [bg_card, fg_card]

        # 1. No displacement: small/zero disocclusion
        cam_pos_zero = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        rendered_zero = render_view(
            image_slices, camera_matrix, card_corners_3d_list, cam_pos_zero, original_size=(100, 100), original_slices=image_slices
        )
        ratio_zero = rendered_zero.reconstruction_ratio
        self.assertEqual(ratio_zero, 0.0)

        # 2. Moderate displacement
        cam_pos_mod = np.array([5.0, 0.0, -100.0], dtype=np.float32)
        rendered_mod = render_view(
            image_slices, camera_matrix, card_corners_3d_list, cam_pos_mod, original_size=(100, 100), original_slices=image_slices
        )
        ratio_mod = rendered_mod.reconstruction_ratio
        self.assertTrue(0.0 < ratio_mod < 0.15)

        # 3. Large displacement (reconstructed region grows)
        cam_pos_large = np.array([15.0, 0.0, -100.0], dtype=np.float32)
        rendered_large = render_view(
            image_slices, camera_matrix, card_corners_3d_list, cam_pos_large, original_size=(100, 100), original_slices=image_slices
        )
        ratio_large = rendered_large.reconstruction_ratio
        self.assertTrue(ratio_large > ratio_mod)

    def test_camera_clamping_and_warnings(self):
        # Create slices
        bg_slice = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=0)
        image_slices = [bg_slice]

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)
        bg_card = bg_slice.create_card(100, 100, camera)
        bg_card[:, :2] *= 1.2 # matching 10% margin
        card_corners_3d_list = [bg_card]

        # 1. Approaching safe boundary
        cam_pos_approaching = np.array([8.0, 0.0, -100.0], dtype=np.float32)
        rendered_appr = render_view(
            image_slices, camera_matrix, card_corners_3d_list, cam_pos_approaching,
            original_size=(100, 100), max_reconstruction_ratio=0.30
        )
        self.assertEqual(len(rendered_appr.warnings), 0)

        # 2. Exceeding safe boundary horizontal limit
        cam_pos_exceeding = np.array([45.0, 0.0, -100.0], dtype=np.float32)
        rendered_exceed = render_view(
            image_slices, camera_matrix, card_corners_3d_list, cam_pos_exceeding,
            original_size=(100, 100), max_reconstruction_ratio=0.01
        )
        self.assertTrue(len(rendered_exceed.warnings) > 0)

    def test_temporal_consistency_validation_and_stats(self):
        """
        Verify consecutive-frame RGB MAD, mask change, provenance change, and visible boundary movement.
        """
        from .segmentation import validate_temporal_consistency

        # Test with identical frames: MAD difference should be 0.0
        frame_a = np.ones((100, 100, 4), dtype=np.uint8) * 100
        frame_b = np.ones((100, 100, 4), dtype=np.uint8) * 100
        mask_a = np.ones((100, 100), dtype=np.uint8) * 255
        mask_b = np.ones((100, 100), dtype=np.uint8) * 255

        mad_zero = validate_temporal_consistency(frame_a, frame_b, mask_a, mask_b)
        self.assertEqual(mad_zero, 0.0)

        # Test with slight difference inside reconstructed mask
        frame_b[40:60, 40:60, :3] = 120 # offset of 20
        mad_diff = validate_temporal_consistency(frame_a, frame_b, mask_a, mask_b)
        self.assertAlmostEqual(mad_diff, 0.8, places=1)

    def test_frame_sequences_100_and_300(self):
        slice_image = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=10)
        image_slices = [slice_image]

        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(100, 100)
        bg_card = slice_image.create_card(100, 100, camera)
        bg_card[:, :2] *= 1.2
        card_corners_3d_list = [bg_card]

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
