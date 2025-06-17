import unittest
from unittest.mock import patch, Mock
from PIL import Image
from io import BytesIO
import requests
from .stabilityai import StabilityAI


class TestStabilityAI(unittest.TestCase):

    @patch("parallax_maker.stabilityai.requests.get")
    def test_validate_key_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"credits": 100}
        mock_get.return_value = mock_response

        ai = StabilityAI(api_key="fake_api_key")
        success, credits = ai.validate_key()

        self.assertTrue(success)
        self.assertEqual(credits, 100)

    @patch("parallax_maker.stabilityai.requests.get")
    def test_validate_key_failure(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        ai = StabilityAI(api_key="fake_api_key")
        success, _ = ai.validate_key()

        self.assertFalse(success)

    @patch("parallax_maker.stabilityai.requests.post")
    def test_generate_image_success(self, mock_post):
        # Prepare mocked response
        image = Image.new("RGB", (100, 100))
        image_data = BytesIO()
        image.save(image_data, format="PNG")
        image_data.seek(0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data.read()
        mock_post.return_value = mock_response

        ai = StabilityAI(api_key="fake_api_key")
        result_image = ai.generate_image(prompt="A dog running in the park")

        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (100, 100))

    @patch("parallax_maker.stabilityai.requests.post")
    def test_generate_image_failure(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_post.return_value = mock_response

        ai = StabilityAI(api_key="fake_api_key")
        with self.assertRaises(Exception):
            ai.generate_image(prompt="A dog running in the park")

    @patch("parallax_maker.stabilityai.requests.post")
    def test_image_to_image_success(self, mock_post):
        image = Image.new("RGB", (100, 100))
        input_image = Image.new("RGB", (100, 100))
        image_data = BytesIO()
        image.save(image_data, format="PNG")
        image_data.seek(0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data.read()
        mock_post.return_value = mock_response

        ai = StabilityAI(api_key="fake_api_key")
        result_image = ai.image_to_image(input_image, prompt="A cat on a sofa")

        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (100, 100))

    @patch("parallax_maker.stabilityai.requests.post")
    def test_image_to_image_invalid_size(self, mock_post):
        small_image = Image.new("RGB", (50, 50))
        ai = StabilityAI(api_key="fake_api_key")

        with self.assertRaises(ValueError):
            ai.image_to_image(small_image, prompt="A cat on a sofa")

    @patch("parallax_maker.stabilityai.requests.post")
    def test_inpaint_image_success(self, mock_post):
        image = Image.new("RGB", (100, 100))
        mask = Image.new("1", (100, 100))
        output_image = Image.new("RGB", (100, 100))
        image_data = BytesIO()
        output_image.save(image_data, format="PNG")
        image_data.seek(0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data.read()
        mock_post.return_value = mock_response

        ai = StabilityAI(api_key="fake_api_key")
        result_image = ai.inpaint_image(
            image, mask, prompt="A beach sunset", strength=0.5
        )

        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (100, 100))

    @patch("parallax_maker.stabilityai.requests.post")
    def test_inpaint_image_invalid_size(self, mock_post):
        small_image = Image.new("RGB", (50, 50))
        mask = Image.new("1", (50, 50))
        ai = StabilityAI(api_key="fake_api_key")

        with self.assertRaises(ValueError):
            ai.inpaint_image(small_image, mask, prompt="A beach sunset")

    @patch("parallax_maker.stabilityai.requests.post")
    def test_upscale_image_success(self, mock_post):
        image = Image.new("RGB", (100, 100))
        output_image = Image.new("RGB", (200, 200))
        image_data = BytesIO()
        output_image.save(image_data, format="PNG")
        image_data.seek(0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data.read()
        mock_post.return_value = mock_response

        ai = StabilityAI(api_key="fake_api_key")
        result_image = ai.upscale_image(image, prompt="Upscale this image")

        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (200, 200))

    @patch("parallax_maker.stabilityai.requests.post")
    def test_upscale_image_invalid_size(self, mock_post):
        large_image = Image.new("RGB", (2000, 2000))
        ai = StabilityAI(api_key="fake_api_key")

        with self.assertRaises(ValueError):
            ai.upscale_image(large_image, prompt="Upscale this image")


if __name__ == "__main__":
    unittest.main()
