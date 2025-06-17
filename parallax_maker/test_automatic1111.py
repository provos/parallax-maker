import unittest
from unittest.mock import patch
from PIL import Image
from .automatic1111 import create_img2img_payload, make_models_request


class TestCreateImg2ImgPayload(unittest.TestCase):
    def setUp(self):
        self.input_image = Image.new("RGB", (100, 100))
        self.positive_prompt = "Generate a beautiful image"
        self.negative_prompt = "Avoid using bright colors"
        self.mask_image = Image.new("L", (100, 100))
        self.strength = 0.8
        self.steps = 20
        self.cfg_scale = 8.0

    def test_create_img2img_payload_without_mask(self):
        with patch(
            "parallax_maker.automatic1111.to_image_url",
            return_value="data:image/png;base64,",
        ) as mock_to_image_url:
            payload = create_img2img_payload(
                self.input_image,
                self.positive_prompt,
                self.negative_prompt,
                mask_image=None,
                strength=self.strength,
                steps=self.steps,
                cfg_scale=self.cfg_scale,
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
        with patch(
            "parallax_maker.automatic1111.to_image_url",
            return_value="data:image/png;base64,",
        ) as mock_to_image_url:
            payload = create_img2img_payload(
                self.input_image,
                self.positive_prompt,
                self.negative_prompt,
                mask_image=self.mask_image,
                strength=self.strength,
                steps=self.steps,
                cfg_scale=self.cfg_scale,
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


class TestMakeModelsRequest(unittest.TestCase):
    @patch("parallax_maker.automatic1111.requests.get")
    def test_make_models_request(self, mock_get):
        mock_response = [
            {
                "title": "sd_xl_base_1.0.safetensors [31e35c80fc]",
                "model_name": "sd_xl_base_1.0",
                "hash": "31e35c80fc",
                "sha256": "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
                "filename": "models/Stable-diffusion/sd_xl_base_1.0.safetensors",
                "config": None,
            },
            {
                "title": "sd_xl_refiner_1.0.safetensors [7440042bbd]",
                "model_name": "sd_xl_refiner_1.0",
                "hash": "7440042bbd",
                "sha256": "7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f",
                "filename": "models/Stable-diffusion/sd_xl_refiner_1.0.safetensors",
                "config": None,
            },
            {
                "title": "v2-1_768-ema-pruned.ckpt [ad2a33c361]",
                "model_name": "v2-1_768-ema-pruned",
                "hash": "ad2a33c361",
                "sha256": "ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0",
                "filename": "models/Stable-diffusion/v2-1_768-ema-pruned.ckpt",
                "config": None,
            },
        ]

        mock_get.return_value.json.return_value = mock_response

        server_address = "localhost:7860"
        models = make_models_request(server_address)

        self.assertEqual(
            models, ["sd_xl_base_1.0", "sd_xl_refiner_1.0", "v2-1_768-ema-pruned"]
        )
        mock_get.assert_called_once_with(
            url="http://localhost:7860/sdapi/v1/sd-models", timeout=3
        )


if __name__ == "__main__":
    unittest.main()
