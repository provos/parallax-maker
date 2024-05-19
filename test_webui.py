import unittest

from unittest.mock import patch, MagicMock
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np

import webui
from webui import update_threshold_values, click_event, copy_to_clipboard
from controller import AppState
import constants as C


class TestUpdateThresholds(unittest.TestCase):
    def test_update_threshold_values_boundaries(self):
        # Create a dummy state
        state = AppState()
        state.imgThresholds = [0, 10, 20, 30, 40, 255]

        # set up a fake cache
        filename = 'teststate'
        state.cache[filename] = state

        # Call the function
        input_thresholds = [0, 20, 30, 40, 255]
        num_slices = 5
        threshold_values, _ = update_threshold_values(
            input_thresholds, num_slices, filename)

        # Assert that the state is updated
        self.assertEqual(state.imgThresholds, [0, 1, 20, 30, 40, 254, 255])

    def test_update_threshold_values_limit(self):
        # Create a dummy state
        state = AppState()
        state.imgThresholds = [0, 10, 20, 30, 40, 255]

        # set up a fake cache
        filename = 'teststate'
        state.cache[filename] = state

        # Call the function
        input_thresholds = [255, 255, 255, 255, 255]
        num_slices = 5
        threshold_values, _ = update_threshold_values(
            input_thresholds, num_slices, filename)

        # Assert that the state is updated
        self.assertEqual(state.imgThresholds, [
                         0, 255, 256, 257, 258, 254, 255])


class TestClickEvent(unittest.TestCase):

    def setUp(self):
        # Patch objects and methods that aren't the focus of this test
        self.ctx_patch = patch('webui.ctx')
        self.AppState_patch = patch('webui.AppState')
        self.find_pixel_patch = patch('webui.find_pixel_from_event')
        self.SegmentationModel_patch = patch('webui.SegmentationModel')
        self.no_update_patch = patch('webui.no_update')

        self.mock_ctx = self.ctx_patch.start()
        self.mock_AppState = self.AppState_patch.start()
        self.mock_find_pixel = self.find_pixel_patch.start()
        self.mock_SegmentationModel = self.SegmentationModel_patch.start()
        self.mock_no_update = self.no_update_patch.start()

        self.mock_state = self.mock_AppState.from_cache.return_value
        self.mock_segmentation_model = self.mock_SegmentationModel.return_value

        # Define default mock return values
        self.mock_state.depth_slice_from_pixel.return_value = (
            np.ones((100, 100)), 1)
        self.mock_segmentation_model.mask_at_point_blended.return_value = np.ones(
            (100, 100))

        self.mock_image = Image.new('RGB', (100, 100))
        self.mock_mask = np.ones((100, 100))

    def tearDown(self):
        # Stop patches
        patch.stopall()

    def test_click_event_no_filename(self):
        with self.assertRaises(PreventUpdate):
            click_event(None, None, None, None, None, None, None)

    @patch('builtins.print')
    def test_click_event_invalid_trigger(self, mock_print):
        self.mock_ctx.triggered_id = 'invalid_trigger'
        with self.assertRaises(ValueError):
            click_event(None, None, None, None, None, 'filename', [])

    def test_click_event_no_element_or_data(self):
        self.mock_ctx.triggered_id = 'el'
        with self.assertRaises(PreventUpdate):
            click_event(None, None, None, None, None, 'filename', [])

    def test_click_event_segment_mode_shift_click(self):
        self.mock_ctx.triggered_id = 'el'
        self.mock_state.multi_point_mode = False
        self.mock_state.imgData = self.mock_image
        self.mock_state.slice_mask = self.mock_mask
        self.mock_state.segmentation_model = None
        element = {'shiftKey': True, 'ctrlKey': False}
        rect_data = 'rect_data'

        self.mock_find_pixel.return_value = (10, 10)

        result = click_event(None, None, element, rect_data,
                             'segment', 'filename', [])

        self.mock_state.apply_mask.assert_called_once()
        self.assertEqual(
            result[0], self.mock_state.serve_main_image.return_value)

    def test_click_event_no_shift_or_ctrl_click(self):
        self.mock_ctx.triggered_id = C.SEG_MULTI_COMMIT
        self.mock_state.multi_point_mode = True
        self.mock_state.points_selected = [((10, 10), False)]
        self.mock_state.imgData = 'image_data'
        self.mock_state.segmentation_model = None

        element = {'shiftKey': False, 'ctrlKey': False}
        rect_data = 'rect_data'

        result = click_event(None, None, element, rect_data,
                             'segment', 'filename', [])

        self.assertEqual(
            result[1], ["Committed points [(10, 10)] and [] for Segment Anything"])

    def test_click_event_apply_mask(self):
        self.mock_ctx.triggered_id = 'el'
        self.mock_state.multi_point_mode = False
        self.mock_state.imgData = 'image_data'
        self.mock_state.slice_mask = self.mock_mask
        element = {'shiftKey': True, 'ctrlKey': False}
        rect_data = 'rect_data'

        self.mock_find_pixel.return_value = (10, 10)

        result = click_event(None, None, element,
                             rect_data, 'mode', 'filename', [])

        self.mock_state.apply_mask.assert_called_once()
        self.assertEqual(
            result[0], self.mock_state.serve_main_image.return_value)


class TestCopyToClipboard(unittest.TestCase):

    @patch('webui.AppState.from_cache')
    def test_copy_to_clipboard_no_clicks(self, mock_from_cache):
        # Test when n_clicks is None, should raise PreventUpdate
        with self.assertRaises(PreventUpdate):
            copy_to_clipboard(None, 'some_filename', [])

    @patch('webui.AppState.from_cache')
    def test_copy_to_clipboard_no_filename(self, mock_from_cache):
        # Test when filename is None, should raise PreventUpdate
        with self.assertRaises(PreventUpdate):
            copy_to_clipboard(1, None, [])

    @patch('webui.AppState.from_cache')
    def test_copy_to_clipboard_no_mask_selected(self, mock_from_cache):
        # Mock AppState with no slice_mask
        mock_state = MagicMock()
        mock_state.slice_mask = None
        mock_from_cache.return_value = mock_state

        # Test when no mask is selected
        logs = []
        result = copy_to_clipboard(1, 'some_filename', logs)
        self.assertEqual(result, ["No mask selected"])

    @patch('webui.AppState.from_cache')
    def test_copy_to_clipboard_with_mask_and_slice(self, mock_from_cache):
        # Mock AppState with a slice_mask and a selected slice
        mock_state = MagicMock()
        mock_state.slice_mask = np.zeros((100, 100))
        mock_state.selected_slice = 'selected_slice'
        mock_state.slice_image_composed.return_value = MagicMock(
            convert=lambda mode: Image.new('RGBA', (100, 100)))

        mock_from_cache.return_value = mock_state

        logs = []
        result = copy_to_clipboard(1, 'some_filename', logs)
        self.assertEqual(result, ["Copied mask to clipboard"])
        self.assertTrue(mock_state.clipboard_image is not None)
        np.testing.assert_array_equal(mock_state.clipboard_image[:, :, 3], mock_state.slice_mask)
        
    @patch('webui.AppState.from_cache')
    def test_copy_to_clipboard_with_mask_no_slice(self, mock_from_cache):
        mock_image = Image.new('RGBA', (100, 100))
        
        # Mock AppState with a slice_mask and no selected slice
        mock_state = MagicMock()
        mock_state.slice_mask = np.zeros((100, 100))
        mock_state.selected_slice = None
        mock_state.imgData = mock_image

        mock_from_cache.return_value = mock_state

        logs = []
        result = copy_to_clipboard(1, 'some_filename', logs)
        self.assertEqual(result, ["Copied mask to clipboard"])
        self.assertTrue(mock_state.clipboard_image is not None)
        np.testing.assert_array_equal(mock_state.clipboard_image[:, : , 3], mock_state.slice_mask)


if __name__ == '__main__':
    unittest.main()
