import unittest
from unittest.mock import patch, MagicMock

import io
import base64
from PIL import Image

import dash
from dash import html

from . import constants as C
import numpy as np
from .components import (
    make_inpainting_container_callbacks,
    make_inpainting_container,
    make_configuration_container,
)
from .slice import ImageSlice


class TestUpdateInpaintingImageDisplay(unittest.TestCase):
    def setUp(self):
        # Setup a minimal Dash app
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div(
            [
                make_inpainting_container(),
                make_configuration_container(),
            ]
        )

        # Save the callback
        self._update_inpainting_image_display = make_inpainting_container_callbacks(
            self.app
        )

    @patch("parallax_maker.components.ctx", new_callable=MagicMock)
    @patch("parallax_maker.components.Image.open")
    @patch("parallax_maker.components.Image.fromarray")
    @patch("parallax_maker.components.Path")
    @patch("parallax_maker.controller.AppState.from_cache")
    @patch("parallax_maker.inpainting.InpaintingModel")
    def test_callback_triggered_comfyui(
        self,
        mock_model,
        mock_from_cache,
        mock_path,
        mock_image_open,
        mock_image_fromarray,
        mock_callback_context,
    ):  # Mock callback context
        mock_callback_context.triggered_id = C.BTN_GENERATE_INPAINTING

        # Setup test data
        n_clicks = 1
        filename = "test_filename"
        model = "comfyui"
        server_address = "http://localhost:8000"
        workflow = "data:filetype;base64,workflow_data"
        positive_prompt = "sky is clear"
        negative_prompt = "cloudy sky"
        strength = 0.7
        guidance_scale = 5.0
        padding = 10
        blur = 5

        # Workflow path mock
        workflow_path = MagicMock()
        workflow_path.exists.return_value = True

        # Set up the AppState mock
        state = MagicMock()
        state.selected_slice = (
            0  # Ensure this matches the state expected by the function
        )
        state.mask_filename.return_value = "mask_0.png"
        state.image_slices = [ImageSlice(np.zeros((100, 100, 4)))]
        state.workflow_path.return_value = workflow_path
        mock_from_cache.return_value = state

        # Mock mask Path exists
        mock_path.return_value.exists.return_value = True

        # Return a mock Image when open succeeds
        mock_img = Image.new("L", (100, 100))
        mock_image_open.return_value = mock_img

        # Setup InpaintingModel mock
        pipeline = MagicMock()
        mock_model.return_value = pipeline
        pipeline.inpaint.return_value = "painted_image"  # Mock the result of inpainting

        # Return the final image
        mock_image_fromarray.return_value = Image.new("RGBA", (100, 100))

        # Call the function under test
        results = self._update_inpainting_image_display(
            n_clicks,
            None,
            None,
            filename,
            model,
            workflow,
            positive_prompt,
            negative_prompt,
            strength,
            guidance_scale,
            padding,
            blur,
        )

        # Verify outputs and behaviors
        pipeline.load_model.assert_called_once()  # Check that model was loaded
        self.assertEqual(len(results[0]), 3)  # Assuming 3 images are expected
        for i, img in enumerate(results[0]):
            id = img.id
            self.assertDictEqual(id, {"type": C.ID_INPAINTING_IMAGE, "index": i})
            img_data = img.src
            # Assuming you have some format to represent the image
            img = Image.open(io.BytesIO(base64.b64decode(img_data.split(",")[1])))
            self.assertEqual(img.size, (100, 100))

        # make sure the second return is a list
        self.assertIsInstance(results[1], list)
        # If this is expected to be an empty list
        self.assertEqual(len(results[1]), 0)

    @patch("parallax_maker.components.ctx", new_callable=MagicMock)
    @patch("parallax_maker.components.Image.open")
    @patch("parallax_maker.components.Image.fromarray")
    @patch("parallax_maker.components.Path")
    @patch("parallax_maker.controller.AppState.from_cache")
    @patch("parallax_maker.inpainting.InpaintingModel")
    def test_callback_triggered_normal(
        self,
        mock_model,
        mock_from_cache,
        mock_path,
        mock_image_open,
        mock_image_fromarray,
        mock_callback_context,
    ):
        # Mock callback context
        mock_callback_context.triggered_id = C.BTN_GENERATE_INPAINTING

        # Setup test data
        n_clicks = 1
        filename = "test_filename"
        model = "other_model"
        server_address = None
        workflow = None
        positive_prompt = "sky is clear"
        negative_prompt = "cloudy sky"
        strength = 0.7
        guidance_scale = 5.0
        padding = 10
        blur = 5

        # Workflow path mock
        workflow_path = MagicMock()
        workflow_path.exists.return_value = False

        # Set up the AppState mock
        state = MagicMock()
        state.selected_slice = (
            0  # Ensure this matches the state expected by the function
        )
        state.mask_filename.return_value = "mask_0.png"
        state.image_slices = [ImageSlice(np.zeros((100, 100, 4)))]
        state.workflow_path.return_value = workflow_path
        mock_from_cache.return_value = state

        # Mock mask Path exists
        mock_path.return_value.exists.return_value = True

        # Return a mock Image when open succeeds
        mock_img = Image.new("L", (100, 100))
        mock_image_open.return_value = mock_img

        # Setup InpaintingModel mock
        pipeline = MagicMock()
        mock_model.return_value = pipeline
        pipeline.inpaint.return_value = "painted_image"  # Mock the result of inpainting

        # Return the final image
        mock_image_fromarray.return_value = Image.new("RGBA", (100, 100))

        # Call the function under test
        results = self._update_inpainting_image_display(
            n_clicks,
            None,
            None,
            filename,
            model,
            workflow,
            positive_prompt,
            negative_prompt,
            strength,
            guidance_scale,
            padding,
            blur,
        )

        # Verify outputs and behaviors
        pipeline.load_model.assert_called_once()  # Check that model was loaded
        self.assertEqual(len(results[0]), 3)  # Assuming 3 images are expected
        for i, img in enumerate(results[0]):
            id = img.id
            self.assertDictEqual(id, {"type": C.ID_INPAINTING_IMAGE, "index": i})
            img_data = img.src
            # Assuming you have some format to represent the image
            img = Image.open(io.BytesIO(base64.b64decode(img_data.split(",")[1])))
            self.assertEqual(img.size, (100, 100))

        # make sure the second return is a list
        self.assertIsInstance(results[1], list)
        # If this is expected to be an empty list
        self.assertEqual(len(results[1]), 0)


if __name__ == "__main__":
    unittest.main()
