import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from .upscaler import Upscaler


class TestUpscaler(unittest.TestCase):
    def setUp(self):
        # Initialize Upscaler object without creating a model.
        self.upscaler = Upscaler(model_name="simple")

        # Create a dummy input image
        self.input_image = Image.new("RGBA", (800, 800))
        # Fill the alpha with 1s - so that the bounding box is the full image
        self.input_image.putalpha(255)

        # Create a dummy upscaled tile
        self.upscaled_tile_dummy = Image.new("RGB", (1, 1))
        self.upscaled_tile_dummy_real = Image.new(
            "RGB", (self.upscaler.tile_size * 2, self.upscaler.tile_size * 2)
        )

    def test_upscale_image_tiled_overlap(self):
        # Mocking the creation of the model to skip loading model and processor
        with patch.object(self.upscaler, "create_model"):
            # Mock the upscale_tile function to return our dummy tile
            with patch.object(
                self.upscaler, "upscale_tile", return_value=self.upscaled_tile_dummy
            ) as mock_upscale_tile:
                # Mock the integrate_tile to ensure integral processing of tiles
                with patch.object(self.upscaler, "integrate_tile") as mock_integrate:
                    # Call the function to test
                    upscaled_image = self.upscaler.upscale_image_tiled(
                        self.input_image, overlap=64
                    )

                    # Check that integrate_tile was called the correct number of times (expected 3x3 grid based on sizes).
                    self.assertEqual(mock_integrate.call_count, 4)
                    mock_upscale_tile.assert_called()

    def test_upscale_image_tiled_no_overlap(self):
        # Mocking the creation of the model to skip loading model and processor
        with patch.object(self.upscaler, "create_model"):
            # Mock the upscale_tile function to return our dummy tile
            with patch.object(
                self.upscaler, "upscale_tile", return_value=self.upscaled_tile_dummy
            ):
                # Mock the integrate_tile to ensure integral processing of tiles
                with patch.object(self.upscaler, "integrate_tile") as mock_integrate:
                    # Call the function to test
                    upscaled_image = self.upscaler.upscale_image_tiled(
                        self.input_image, overlap=0
                    )

                    # Check that integrate_tile was called the correct number of times (expected 2x2 grid based on sizes).
                    self.assertEqual(mock_integrate.call_count, 4)

    def test_upscale_image_tiled_no_mock(self):
        # Mocking the creation of the model to skip loading model and processor
        with patch.object(self.upscaler, "create_model"):
            # Mock the upscale_tile function to return our dummy tile
            with patch.object(
                self.upscaler,
                "upscale_tile",
                return_value=self.upscaled_tile_dummy_real,
            ):
                # Call the function to test
                upscaled_image = self.upscaler.upscale_image_tiled(
                    self.input_image, overlap=64
                )

                # Check that integrate_tile was called the correct number of times (expected 3x3 grid based on sizes).
                self.assertEqual(upscaled_image.size, (2 * 800, 2 * 800))


class TestIntegrateTile(unittest.TestCase):

    def setUp(self):
        # Create a sample tile and image for testing
        self.tile = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_integrate_tile_no_overlap(self):
        # Test integrating a tile with no overlap
        Upscaler.integrate_tile(self.tile, self.image, 50, 50, 150, 150, 0, 0, 0)
        self.assertTrue(np.array_equal(self.image[50:150, 50:150], self.tile))

    def test_integrate_tile_left_overlap(self):
        # Test integrating a tile with left overlap
        Upscaler.integrate_tile(self.tile, self.image, 0, 50, 100, 150, 1, 0, 20)
        self.assertTrue(np.all(self.image[50:150, 1:20] > 0))
        self.assertTrue(np.all(self.image[50:150, 20:100] == 255))

    def test_integrate_tile_top_overlap(self):
        # Test integrating a tile with top overlap
        Upscaler.integrate_tile(self.tile, self.image, 50, 0, 150, 100, 0, 1, 20)
        self.assertTrue(np.all(self.image[1:20, 50:150] > 0))
        self.assertTrue(np.all(self.image[20:100, 50:150] == 255))

    def test_integrate_tile_corner_overlap(self):
        # Test integrating a tile with corner overlap
        Upscaler.integrate_tile(self.tile, self.image, 0, 0, 100, 100, 1, 1, 20)
        self.assertTrue(np.all(self.image[2:20, 2:20] > 0))
        self.assertTrue(np.all(self.image[1:20, 20:100] > 0))
        self.assertTrue(np.all(self.image[20:100, 1:20] > 0))
        self.assertTrue(np.all(self.image[20:100, 20:100] == 255))

    def test_integrate_tile_gradient(self):
        # Test if there is a gradient in the overlapping regions
        overlap = 20
        Upscaler.integrate_tile(self.tile, self.image, 0, 0, 100, 100, 1, 1, overlap)

        # Check left overlap gradient
        left_overlap = self.image[:, :overlap]
        self.assertFalse(np.all(left_overlap == left_overlap[0, 0]))
        self.assertTrue(np.all(np.diff(left_overlap, axis=1) >= 0))

        # Check top overlap gradient
        top_overlap = self.image[:overlap, :]
        self.assertFalse(np.all(top_overlap == top_overlap[0, 0]))
        self.assertTrue(np.all(np.diff(top_overlap, axis=0) >= 0))

        # Check corner overlap gradient
        corner_overlap = self.image[:overlap, :overlap]
        self.assertFalse(np.all(corner_overlap == corner_overlap[0, 0]))
        self.assertTrue(np.all(np.diff(corner_overlap, axis=0) >= 0))
        self.assertTrue(np.all(np.diff(corner_overlap, axis=1) >= 0))


if __name__ == "__main__":
    unittest.main()
