import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from upscaler import Upscaler


class TestUpscaler(unittest.TestCase):
    def setUp(self):
        # Initialize Upscaler object without creating a model.
        self.upscaler = Upscaler()

        # Create a dummy input image
        self.input_image = Image.new("RGB", (1024, 1024))

        # Create a dummy upscaled tile
        self.upscaled_tile_dummy = Image.new("RGB", (1, 1))

    def test_upscale_image_tiled_overlap(self):
        # Mocking the creation of the model to skip loading model and processor
        with patch.object(self.upscaler, 'create_model'):
            # Mock the upscale_tile function to return our dummy tile
            with patch.object(self.upscaler, 'upscale_tile', return_value=self.upscaled_tile_dummy):
                # Mock the integrate_tile to ensure integral processing of tiles
                with patch.object(self.upscaler, 'integrate_tile') as mock_integrate:
                    # Call the function to test
                    upscaled_image = self.upscaler.upscale_image_tiled(
                        self.input_image, tile_size=512, overlap=64)

                    # Check that integrate_tile was called the correct number of times (expected 3x3 grid based on sizes).
                    self.assertEqual(mock_integrate.call_count, 9)

    def test_upscale_image_tiled_no_overlap(self):
        # Mocking the creation of the model to skip loading model and processor
        with patch.object(self.upscaler, 'create_model'):
            # Mock the upscale_tile function to return our dummy tile
            with patch.object(self.upscaler, 'upscale_tile', return_value=self.upscaled_tile_dummy):
                # Mock the integrate_tile to ensure integral processing of tiles
                with patch.object(self.upscaler, 'integrate_tile') as mock_integrate:
                    # Call the function to test
                    upscaled_image = self.upscaler.upscale_image_tiled(
                        self.input_image, tile_size=512, overlap=0)

                    # Check that integrate_tile was called the correct number of times (expected 2x2 grid based on sizes).
                    self.assertEqual(mock_integrate.call_count, 4)



if __name__ == '__main__':
    unittest.main()
