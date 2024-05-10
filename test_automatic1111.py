import unittest
from unittest.mock import patch
from PIL import Image
from automatic1111 import create_img2img_payload


class TestCreateImg2ImgPayload(unittest.TestCase):
    def setUp(self):
        self.input_image = Image.new('RGB', (100, 100))
        self.positive_prompt = "Generate a beautiful image"
        self.negative_prompt = "Avoid using bright colors"
        self.mask_image = Image.new('L', (100, 100))
        self.strength = 0.8
        self.steps = 20
        self.cfg_scale = 8.0

    def test_create_img2img_payload_without_mask(self):
        with patch("automatic1111.to_image_url", return_value="data:image/png;base64,") as mock_to_image_url:
            payload = create_img2img_payload(
                self.input_image, self.positive_prompt, self.negative_prompt,
                mask_image=None, strength=self.strength, steps=self.steps, cfg_scale=self.cfg_scale
            )

        self.assertEqual(payload["init_images"], ["data:image/png;base64,"])
        self.assertEqual(payload["prompt"], self.positive_prompt)
        self.assertEqual(payload["negative_prompt"], self.negative_prompt)
        self.assertEqual(payload["denoising_strength"], self.strength)
        self.assertEqual(payload["width"], 100)
        self.assertEqual(payload["height"], 100)
        self.assertEqual(payload["steps"], self.steps)
        self.assertEqual(payload["cfg_scale"], self.cfg_scale)
        self.assertNotIn("mask", payload)

    def test_create_img2img_payload_with_mask(self):
        with patch("automatic1111.to_image_url", return_value="data:image/png;base64,") as mock_to_image_url:
            payload = create_img2img_payload(
                self.input_image, self.positive_prompt, self.negative_prompt,
                mask_image=self.mask_image, strength=self.strength, steps=self.steps, cfg_scale=self.cfg_scale
            )

        self.assertEqual(payload["init_images"], ["data:image/png;base64,"])
        self.assertEqual(payload["prompt"], self.positive_prompt)
        self.assertEqual(payload["negative_prompt"], self.negative_prompt)
        self.assertEqual(payload["denoising_strength"], self.strength)
        self.assertEqual(payload["width"], 100)
        self.assertEqual(payload["height"], 100)
        self.assertEqual(payload["steps"], self.steps)
        self.assertEqual(payload["cfg_scale"], self.cfg_scale)
        self.assertEqual(payload["mask"], "data:image/png;base64,")


if __name__ == '__main__':
    unittest.main()
