import unittest
from unittest.mock import patch, MagicMock
from dash import dcc, html, no_update
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np
from pathlib import Path

from .webui import (
    update_threshold_values,
    click_event,
    copy_to_clipboard,
    export_state_as_gltf,
    slice_upload,
    update_slices,
)
from .controller import AppState
from .utils import to_image_url
from .camera import Camera
from .slice import ImageSlice
from . import constants as C


class TestUpdateThresholds(unittest.TestCase):
    def test_update_threshold_values_boundaries(self):
        # Create a dummy state
        state = AppState()
        state.imgThresholds = [0, 10, 20, 30, 40, 255]

        # set up a fake cache
        filename = "teststate"
        state.cache[filename] = state

        # Call the function
        input_thresholds = [0, 20, 30, 40, 255]
        num_slices = 5
        threshold_values, _ = update_threshold_values(
            input_thresholds, num_slices, filename
        )

        # Assert that the state is updated
        self.assertEqual(state.imgThresholds, [0, 1, 20, 30, 40, 254, 255])

    def test_update_threshold_values_limit(self):
        # Create a dummy state
        state = AppState()
        state.imgThresholds = [0, 10, 20, 30, 40, 255]

        # set up a fake cache
        filename = "teststate"
        state.cache[filename] = state

        # Call the function
        input_thresholds = [255, 255, 255, 255, 255]
        num_slices = 5
        threshold_values, _ = update_threshold_values(
            input_thresholds, num_slices, filename
        )

        # Assert that the state is updated
        self.assertEqual(state.imgThresholds, [0, 255, 256, 257, 258, 254, 255])


class TestClickEvent(unittest.TestCase):

    def setUp(self):
        # Patch objects and methods that aren't the focus of this test
        self.ctx_patch = patch("parallax_maker.webui.ctx")
        self.AppState_patch = patch("parallax_maker.webui.AppState")
        self.find_pixel_patch = patch("parallax_maker.webui.find_pixel_from_event")
        self.SegmentationModel_patch = patch("parallax_maker.webui.SegmentationModel")
        self.no_update_patch = patch("parallax_maker.webui.no_update")

        self.mock_ctx = self.ctx_patch.start()
        self.mock_AppState = self.AppState_patch.start()
        self.mock_find_pixel = self.find_pixel_patch.start()
        self.mock_SegmentationModel = self.SegmentationModel_patch.start()
        self.mock_no_update = self.no_update_patch.start()

        self.mock_state = self.mock_AppState.from_cache.return_value
        self.mock_segmentation_model = self.mock_SegmentationModel.return_value

        # Define default mock return values
        self.mock_state.depth_slice_from_pixel.return_value = (np.ones((100, 100)), 1)
        self.mock_segmentation_model.mask_at_point_blended.return_value = np.ones(
            (100, 100)
        )

        self.mock_image = Image.new("RGB", (100, 100))
        self.mock_mask = np.ones((100, 100))

    def tearDown(self):
        # Stop patches
        patch.stopall()

    def test_click_event_no_filename(self):
        with self.assertRaises(PreventUpdate):
            click_event(None, None, None, None, None, None, None)

    @patch("builtins.print")
    def test_click_event_invalid_trigger(self, mock_print):
        self.mock_ctx.triggered_id = "invalid_trigger"
        with self.assertRaises(ValueError):
            click_event(None, None, None, None, None, "filename", [])

    def test_click_event_no_element_or_data(self):
        self.mock_ctx.triggered_id = "el"
        with self.assertRaises(PreventUpdate):
            click_event(None, None, None, None, None, "filename", [])

    def test_click_event_segment_mode_shift_click(self):
        self.mock_ctx.triggered_id = "el"
        self.mock_state.multi_point_mode = False
        self.mock_state.imgData = self.mock_image
        self.mock_state.slice_mask = self.mock_mask
        self.mock_state.segmentation_model = None
        element = {"shiftKey": True, "ctrlKey": False}
        rect_data = "rect_data"

        self.mock_find_pixel.return_value = (10, 10)

        result = click_event(None, None, element, rect_data, "segment", "filename", [])

        self.mock_state.apply_mask.assert_called_once()
        self.assertEqual(result[0], self.mock_state.serve_main_image.return_value)

    def test_click_event_no_shift_or_ctrl_click(self):
        self.mock_ctx.triggered_id = C.SEG_MULTI_COMMIT
        self.mock_state.multi_point_mode = True
        self.mock_state.points_selected = [((10, 10), False)]
        self.mock_state.imgData = "image_data"
        self.mock_state.segmentation_model = None

        element = {"shiftKey": False, "ctrlKey": False}
        rect_data = "rect_data"

        result = click_event(None, None, element, rect_data, "segment", "filename", [])

        self.assertEqual(
            result[1], ["Committed points [(10, 10)] and [] for Segment Anything"]
        )

    def test_click_event_apply_mask(self):
        self.mock_ctx.triggered_id = "el"
        self.mock_state.multi_point_mode = False
        self.mock_state.imgData = "image_data"
        self.mock_state.slice_mask = self.mock_mask
        element = {"shiftKey": True, "ctrlKey": False}
        rect_data = "rect_data"

        self.mock_find_pixel.return_value = (10, 10)

        result = click_event(None, None, element, rect_data, "mode", "filename", [])

        self.mock_state.apply_mask.assert_called_once()
        self.assertEqual(result[0], self.mock_state.serve_main_image.return_value)

    def test_click_event_with_slice(self):
        self.mock_ctx.triggered_id = "el"
        self.mock_state.multi_point_mode = False
        self.mock_state.imgData = "image_data"
        self.mock_state.slice_mask = self.mock_mask
        self.mock_state.image_slices = [np.ones((100, 100, 4))]
        self.mock_state.selected_slice = 0
        element = {"shiftKey": True, "ctrlKey": False}
        rect_data = "rect_data"

        self.mock_find_pixel.return_value = (10, 10)

        result = click_event(None, None, element, rect_data, "mode", "filename", [])

        self.mock_state.apply_mask.assert_called_once()
        self.assertEqual(result[0], self.mock_state.serve_main_image.return_value)


class TestCopyToClipboard(unittest.TestCase):

    @patch("parallax_maker.webui.AppState.from_cache")
    def test_copy_to_clipboard_no_clicks(self, mock_from_cache):
        # Test when n_clicks is None, should raise PreventUpdate
        with self.assertRaises(PreventUpdate):
            copy_to_clipboard(None, "some_filename", [])

    @patch("parallax_maker.webui.AppState.from_cache")
    def test_copy_to_clipboard_no_filename(self, mock_from_cache):
        # Test when filename is None, should raise PreventUpdate
        with self.assertRaises(PreventUpdate):
            copy_to_clipboard(1, None, [])

    @patch("parallax_maker.webui.AppState.from_cache")
    def test_copy_to_clipboard_no_mask_selected(self, mock_from_cache):
        # Mock AppState with no slice_mask
        mock_state = MagicMock()
        mock_state.slice_mask = None
        mock_from_cache.return_value = mock_state

        # Test when no mask is selected
        logs = []
        result = copy_to_clipboard(1, "some_filename", logs)
        self.assertEqual(result, ["No mask selected"])

    @patch("parallax_maker.webui.AppState.from_cache")
    def test_copy_to_clipboard_with_mask_and_slice(self, mock_from_cache):
        # Mock AppState with a slice_mask and a selected slice
        mock_state = MagicMock()
        mock_state.slice_mask = np.zeros((100, 100))
        mock_state.selected_slice = "selected_slice"
        mock_state.slice_image_composed.return_value = MagicMock(
            convert=lambda mode: Image.new("RGBA", (100, 100))
        )

        mock_from_cache.return_value = mock_state

        logs = []
        result = copy_to_clipboard(1, "some_filename", logs)
        self.assertEqual(result, ["Copied mask to clipboard"])
        self.assertTrue(mock_state.clipboard_image is not None)
        np.testing.assert_array_equal(
            mock_state.clipboard_image[:, :, 3], mock_state.slice_mask
        )

    @patch("parallax_maker.webui.AppState.from_cache")
    def test_copy_to_clipboard_with_mask_no_slice(self, mock_from_cache):
        mock_image = Image.new("RGBA", (100, 100))

        # Mock AppState with a slice_mask and no selected slice
        mock_state = MagicMock()
        mock_state.slice_mask = np.zeros((100, 100))
        mock_state.selected_slice = None
        mock_state.imgData = mock_image

        mock_from_cache.return_value = mock_state

        logs = []
        result = copy_to_clipboard(1, "some_filename", logs)
        self.assertEqual(result, ["Copied mask to clipboard"])
        self.assertTrue(mock_state.clipboard_image is not None)
        np.testing.assert_array_equal(
            mock_state.clipboard_image[:, :, 3], mock_state.slice_mask
        )


class TestExportGltf(unittest.TestCase):
    def setUp(self):
        self.state = MagicMock()
        self.state.image_slices = [
            ImageSlice(np.zeros((100, 100, 4), dtype=np.uint8), i) for i in range(3)
        ]

        # mocking depthmap file requires both exists and return_value
        self.mock_depth_file = MagicMock()
        self.state.depth_filename.return_value = self.mock_depth_file
        self.mock_depth_file.exists.return_value = True

        self.camera = Camera(10, 100, 50)

        self.state.camera_matrix.return_value = self.camera.camera_matrix(100, 100)
        self.state.get_cards.return_value = [
            slice.create_card(100, 100, self.camera)
            for slice in self.state.image_slices
        ]

        self.state.upscaled_filename.return_value = Path("upscaled_file.png")
        self.state.MODEL_FILE = "model.gltf"
        self.state.camera = self.camera

    @patch("parallax_maker.webui.generate_depth_map")
    @patch("parallax_maker.webui.postprocess_depth_map")
    @patch("parallax_maker.webui.export_gltf")
    def test_export_state_as_gltf(
        self, mock_export_gltf, mock_postprocess_depth_map, mock_generate_depth_map
    ):
        # Test case 1: Displacement scale is 0
        state = AppState()
        state.image_slices = self.state.image_slices

        mock_export_gltf.return_value = Path("output.gltf")

        result = export_state_as_gltf(
            self.state, "output_dir", Camera(10, 100, 50), 0, "midas"
        )

        self.assertEqual(result, Path("output.gltf"))
        mock_generate_depth_map.assert_not_called()
        mock_postprocess_depth_map.assert_not_called()

        # Compare individual elements of card_corners_3d_list
        expected_call = mock_export_gltf.call_args_list[0]
        expected_args, expected_kwargs = expected_call
        self.assertEqual(expected_args[0], Path("output_dir/model.gltf"))
        image_slices_filenames = [slice.filename for slice in self.state.image_slices]
        self.assertEqual(expected_args[3], image_slices_filenames)
        self.assertEqual(expected_args[4], [])
        self.assertEqual(expected_kwargs["displacement_scale"], 0)

    @patch("PIL.Image.fromarray")
    @patch("parallax_maker.webui.generate_depth_map")
    @patch("parallax_maker.webui.postprocess_depth_map")
    @patch("parallax_maker.webui.export_gltf")
    def test_export_state_as_gltf_with_displacement(
        self,
        mock_export_gltf,
        mock_postprocess_depth_map,
        mock_generate_depth_map,
        mock_image_fromarray,
    ):
        # Test case 2: Displacement scale is greater than 0
        state = AppState()
        state.image_slices = self.state.image_slices

        mock_export_gltf.return_value = Path("output.gltf")

        mock_generate_depth_map.return_value = np.zeros((100, 100), dtype=np.float32)
        mock_postprocess_depth_map.return_value = np.zeros((100, 100), dtype=np.uint8)

        # path does not exist
        self.mock_depth_file.exists.return_value = False

        # mocking depthmap image saving
        mock_image = MagicMock(spec=Image.Image)
        mock_image_fromarray.return_value = mock_image

        result = export_state_as_gltf(
            self.state, "output_dir", Camera(10, 100, 50), 1, "midas"
        )

        self.assertEqual(result, Path("output.gltf"))
        self.assertEqual(mock_generate_depth_map.call_count, 3)
        self.assertEqual(mock_postprocess_depth_map.call_count, 3)
        self.assertEqual(mock_image_fromarray.call_count, 3)
        mock_image.save.assert_called_with(self.mock_depth_file, compress_level=1)

        # Compare individual elements of card_corners_3d_list
        expected_call = mock_export_gltf.call_args_list[0]
        expected_args, expected_kwargs = expected_call
        self.assertEqual(expected_args[0], Path("output_dir/model.gltf"))

        image_slices_filenames = [slice.filename for slice in self.state.image_slices]
        self.assertEqual(expected_args[3], image_slices_filenames)
        self.assertEqual(expected_args[4], [self.mock_depth_file] * 3)
        self.assertEqual(expected_kwargs["displacement_scale"], 1)

    @patch("parallax_maker.webui.export_gltf")
    def test_export_state_as_gltf_with_upscaled(self, mock_export_gltf):
        # Test case 3: Upscaled slices exist
        state = AppState()
        state.image_slices = self.state.image_slices

        # Pretend the upscaled file exists
        mock_upscaled_file = MagicMock()
        mock_upscaled_file.exists.return_value = True
        self.state.upscaled_filename.return_value = mock_upscaled_file

        result = export_state_as_gltf(
            self.state, "output_dir", Camera(10, 100, 50), 1, "midas"
        )

        # Compare individual elements of card_corners_3d_list
        expected_call = mock_export_gltf.call_args_list[0]
        expected_args, expected_kwargs = expected_call
        self.assertEqual(expected_args[0], Path("output_dir/model.gltf"))
        self.assertEqual(expected_args[3], [mock_upscaled_file] * 3)
        self.assertEqual(expected_args[4], [self.mock_depth_file] * 3)
        self.assertEqual(expected_kwargs["displacement_scale"], 1)

    # TODO: Add more test cases for generate_depth_map, postprocess_depth_map, and export_gltf


class TestSliceUpload(unittest.TestCase):
    @patch("parallax_maker.webui.ctx")
    @patch("parallax_maker.webui.AppState.from_cache")
    def test_filename_none(self, mock_from_cache, mock_ctx):
        with self.assertRaises(PreventUpdate):
            slice_upload(None, None, None)
        mock_from_cache.assert_not_called()

    @patch("parallax_maker.webui.ctx")
    @patch("parallax_maker.webui.AppState.from_cache")
    def test_empty_image_slices(self, mock_from_cache, mock_ctx):
        mock_state = MagicMock(spec=AppState)
        mock_state.image_slices = []
        mock_from_cache.return_value = mock_state

        with self.assertRaises(PreventUpdate):
            slice_upload(None, "appstate-random", None)
        mock_from_cache.assert_called_once_with("appstate-random")

    @patch("parallax_maker.webui.ctx")
    @patch("parallax_maker.webui.AppState.from_cache")
    def test_contents_none(self, mock_from_cache, mock_ctx):
        mock_state = MagicMock(spec=AppState)
        mock_state.image_slices = [np.zeros((100, 100, 4))]
        mock_from_cache.return_value = mock_state
        mock_ctx.triggered_id = {"index": 0}

        with self.assertRaises(PreventUpdate):
            slice_upload([None], "appstate-random", None)
        mock_from_cache.assert_called_once_with("appstate-random")

    @patch("parallax_maker.webui.ctx")
    @patch("parallax_maker.webui.AppState.from_cache")
    @patch("parallax_maker.slice.filename_add_version")
    @patch("parallax_maker.webui.blend_with_alpha")
    @patch("parallax_maker.slice.ImageSlice.save_image")
    def test_valid_upload(
        self,
        mock_imwrite,
        mock_blend,
        mock_filename_add_version,
        mock_from_cache,
        mock_ctx,
    ):
        mock_state = MagicMock(spec=AppState)
        mock_state.image_slices = [
            ImageSlice(np.zeros((100, 100, 4), dtype=np.uint8), filename="slice0.png"),
            ImageSlice(np.ones((100, 100, 4), dtype=np.uint8), filename="slice1.png"),
        ]
        mock_from_cache.return_value = mock_state
        mock_ctx.triggered_id = {"index": 1}
        mock_filename_add_version.return_value = "slice1_v1.png"

        content = to_image_url(np.ones((100, 100, 4), dtype=np.uint8))
        print(content[:50])

        result = slice_upload([None, content], "appstate-random", [])

        mock_imwrite.assert_called()
        self.assertEqual(result[0], True)
        self.assertEqual(len(result[1]), 1)
        self.assertIn(
            "Received image slice upload for slice 1 at slice1_v1.png", result[1][0]
        )
        mock_from_cache.assert_called_once_with("appstate-random")
        mock_filename_add_version.assert_called_once_with("slice1.png")
        mock_blend.assert_called_once()
        mock_state.to_file.assert_called_once_with(
            "appstate-random",
            save_image_slices=False,
            save_depth_map=False,
            save_input_image=False,
        )
        self.assertIsInstance(mock_state.imgData, Image.Image)

    @patch("parallax_maker.webui.ctx")
    @patch("parallax_maker.webui.AppState.from_cache")
    @patch("parallax_maker.slice.filename_add_version")
    @patch("parallax_maker.webui.blend_with_alpha")
    @patch("parallax_maker.slice.ImageSlice.save_image")
    def test_valid_upload_different_ratio(
        self,
        mock_imwrite,
        mock_blend,
        mock_filename_add_version,
        mock_from_cache,
        mock_ctx,
    ):
        mock_state = MagicMock(spec=AppState)
        mock_state.image_slices = [
            ImageSlice(np.zeros((100, 100, 4), dtype=np.uint8), filename="slice0.png"),
            ImageSlice(np.ones((100, 100, 4), dtype=np.uint8), filename="slice1.png"),
        ]
        mock_from_cache.return_value = mock_state
        mock_ctx.triggered_id = {"index": 1}
        mock_filename_add_version.return_value = "slice1_v1.png"

        content = to_image_url(np.ones((110, 99, 4), dtype=np.uint8))

        result = slice_upload([None, content], "appstate-random", [])

        mock_imwrite.assert_called()
        self.assertEqual(result[0], True)
        self.assertEqual(len(result[1]), 2)
        self.assertIn("Fixing aspect ratio from", result[1][0])
        mock_from_cache.assert_called_once_with("appstate-random")
        mock_filename_add_version.assert_called_once_with("slice1.png")
        mock_blend.assert_called_once()
        mock_state.to_file.assert_called_once_with(
            "appstate-random",
            save_image_slices=False,
            save_depth_map=False,
            save_input_image=False,
        )
        self.assertIsInstance(mock_state.imgData, Image.Image)


class TestUpdateSlices(unittest.TestCase):

    def setUp(self):
        # Mocks and AppState setup
        self.filename = "test_file"
        self.mock_state = AppState()

        # Mock the AppState.from_cache method
        self.patcher = patch(
            "parallax_maker.webui.AppState.from_cache", return_value=self.mock_state
        )
        self.mock_from_cache = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_no_filename(self):
        ignored_data = MagicMock()
        # Testing when filename is None
        with self.assertRaises(PreventUpdate):
            update_slices(ignored_data, None)

    def test_no_image_slices(self):
        # Test with no image slices
        self.mock_state.image_slices = []
        ignored_data = MagicMock()
        result = update_slices(ignored_data, self.filename)
        self.assertEqual(result, ([], "", no_update))

    def test_no_depth_map_data(self):
        # Test without depthMapData
        self.mock_state.image_slices = [MagicMock()]
        self.mock_state.depthMapData = None
        ignored_data = MagicMock()
        with self.assertRaises(PreventUpdate):
            update_slices(ignored_data, self.filename)

    @patch(
        "parallax_maker.webui.AppState.serve_slice_image_composed",
        return_value="composed_image_data",
    )
    @patch("parallax_maker.webui.AppState.serve_slice_image", return_value="image_data")
    @patch("parallax_maker.slice.ImageSlice.can_undo", return_value=False)
    def test_full_functionality(
        self, mock_serve_slice_image_composed, mock_can_undo, mock_serve_slice_image
    ):
        # Simulate state with several slices
        self.mock_state.image_slices = [
            ImageSlice(depth=0, filename="slice1.png"),
            ImageSlice(depth=1, filename="slice2.png"),
        ]
        self.mock_state.depthMapData = True
        self.mock_state.selected_slice = 0
        self.mock_state.use_checkerboard = True

        ignored_data = MagicMock()
        result = update_slices(ignored_data, self.filename)

        # Check that the result is correctly generated images and the composed image
        img_container, gen_slice_output, img_data = result
        self.assertEqual(len(img_container), 2)
        self.assertEqual(img_data, "composed_image_data")

    @patch("parallax_maker.webui.AppState.serve_slice_image", return_value="image_data")
    @patch("parallax_maker.slice.ImageSlice.can_undo", return_value=False)
    def test_corner_cases(self, mock_can_undo, mock_serve_slice_image):
        # Simulate corner cases where selected_slice is None
        self.mock_state.image_slices = [ImageSlice(depth=0, filename="slice1.png")]
        self.mock_state.depthMapData = True
        self.mock_state.selected_slice = None

        ignored_data = MagicMock()
        result = update_slices(ignored_data, self.filename)

        # Check that the result correctly handles the corner case and does not update the image data
        img_container, gen_slice_output, img_data = result
        self.assertEqual(len(img_container), 1)
        self.assertEqual(img_data, no_update)


if __name__ == "__main__":
    unittest.main()
