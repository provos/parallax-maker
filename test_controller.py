import numpy as np
import unittest
from unittest.mock import patch, MagicMock

from utils import encode_string_with_nonce, decode_string_with_nonce
from controller import AppState


class TestAddSlice(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = 'test'

    def test_add_slice_normal(self):
        # Create a mock slice image and depth value
        slice_image1 = 'image-1'
        depth1 = 5

        # Call the function
        insert1 = self.state.add_slice(slice_image1, depth1)

        self.assertEqual(insert1, 0)
        self.assertEqual(len(self.state.image_slices), 1)
        self.assertEqual(self.state.image_depths[0], depth1)

        slice_image2 = 'image-2'
        depth2 = 10

        # Call the function
        insert2 = self.state.add_slice(slice_image2, depth2)

        # Check whether this slice was added before the first slice
        self.assertEqual(insert2, 1)
        self.assertEqual(len(self.state.image_slices), 2)
        self.assertEqual(self.state.image_depths[0], depth1)
        self.assertEqual(self.state.image_depths[1], depth2)

    def test_add_slice_reverse(self):
        # Create a mock slice image and depth value
        slice_image1 = 'image-1'
        depth1 = 5

        # Call the function
        insert1 = self.state.add_slice(slice_image1, depth1)

        self.assertEqual(insert1, 0)
        self.assertEqual(len(self.state.image_slices), 1)
        self.assertEqual(self.state.image_depths[0], depth1)

        slice_image2 = 'image-2'
        depth2 = 1

        # Call the function
        insert2 = self.state.add_slice(slice_image2, depth2)

        # Check whether this slice was added before the first slice
        self.assertEqual(insert2, 0)
        self.assertEqual(len(self.state.image_slices), 2)
        self.assertEqual(self.state.image_depths[0], depth2)
        self.assertEqual(self.state.image_depths[1], depth1)

        slice_image3 = 'image-2'
        depth3 = 1

        # Call the function
        insert3 = self.state.add_slice(slice_image3, depth3)

        # Check that the slice was added and depth increased to avoid duplicates
        self.assertEqual(insert3, 1)
        self.assertEqual(len(self.state.image_slices), 3)
        self.assertEqual(self.state.image_depths[0], depth2)
        self.assertEqual(self.state.image_depths[1], depth3+1)

        # check that all the filenames are accurate
        expected = [
            'test/image_slice_1.png',
            'test/image_slice_2.png',
            'test/image_slice_0.png'
        ]
        self.assertEqual(self.state.image_slices_filenames, expected)


class TestCheckPathnames(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = 'appstate-test'

    def test_check_pathnames_valid(self):
        # Set up valid pathnames
        self.state.image_slices_filenames = [
            'appstate-test/image_slice_1.png',
            'appstate-test/image_slice_2.png',
            'appstate-test/image_slice_0.png'
        ]

        # Call the function
        self.state.check_pathnames()

    def test_check_pathnames_invalid_root(self):
        # Set up invalid root pathname
        self.state.filename = 'invalid_root/../something'
        self.state.image_slices_filenames = [
            'appstate-test/image_slice_2.png',
            'appstate-test/image_slice_0.png'
        ]

        # Call the function and expect an AssertionError
        with self.assertRaises(AssertionError):
            self.state.check_pathnames()

    def test_check_pathnames_invalid_filenames(self):
        # Set up invalid filenames
        self.state.image_slices_filenames = [
            'appstate-test/image_slice_1.png',
            'appstate-test/image_slice_2.png',
            'appstate-test/../../../invalid_filename.png'
        ]

        # Call the function and expect an AssertionError
        with self.assertRaises(AssertionError):
            self.state.check_pathnames()


class TestChangeSliceDepth(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = 'test'

        # Add some initial slices
        self.state.add_slice(
            'image-1', 5, positive_prompt="one", negative_prompt="two")
        self.state.add_slice(
            'image-2', 10, positive_prompt="three", negative_prompt="four")
        self.state.add_slice(
            'image-3', 15, positive_prompt="blue", negative_prompt="green")

    def test_change_slice_depth_same_depth(self):
        # Get the initial state
        initial_slices = self.state.image_slices.copy()
        initial_depths = self.state.image_depths.copy()

        # Change the depth of the second slice to the same depth
        slice_index = 1
        depth = 10
        new_index = self.state.change_slice_depth(slice_index, depth)

        # Check that the slice index remains the same
        self.assertEqual(new_index, slice_index)

        # Check that the slices and depths remain unchanged
        self.assertEqual(self.state.image_slices, initial_slices)
        self.assertEqual(self.state.image_depths, initial_depths)

    def test_change_slice_depth_valid(self):
        # Get the initial state
        initial_slices = self.state.image_slices.copy()
        initial_depths = self.state.image_depths.copy()

        # Change the depth of the second slice to a new depth
        slice_index = 1
        depth = 3
        new_index = self.state.change_slice_depth(slice_index, depth)

        # Check that the new index is returned
        self.assertEqual(new_index, 0)

        # Check that the depth of the second slice is updated
        self.assertEqual(self.state.image_depths[new_index], depth)
        self.assertEqual(self.state.positive_prompts[new_index], "three")
        self.assertEqual(self.state.negative_prompts[new_index], "four")

        # Check that the other slices and depths remain unchanged
        self.assertEqual(self.state.image_slices[1], initial_slices[0])
        self.assertEqual(self.state.image_slices[2], initial_slices[2])
        self.assertEqual(self.state.image_depths[1], initial_depths[0])
        self.assertEqual(self.state.image_depths[2], initial_depths[2])

    def test_change_slice_depth_invalid_index(self):
        # Get the initial state
        initial_slices = self.state.image_slices.copy()
        initial_depths = self.state.image_depths.copy()

        # Try to change the depth of an invalid slice index
        slice_index = 3
        depth = 12
        with self.assertRaises(AssertionError):
            self.state.change_slice_depth(slice_index, depth)

        # Check that the slices and depths remain unchanged
        self.assertEqual(self.state.image_slices, initial_slices)
        self.assertEqual(self.state.image_depths, initial_depths)


class TestFromJson(unittest.TestCase):
    def test_from_json_empty(self):
        json_data = None
        state = AppState.from_json(json_data)
        self.assertIsInstance(state, AppState)
        self.assertEqual(state.filename, None)
        self.assertEqual(state.num_slices, 5)
        self.assertEqual(state.imgThresholds, None)
        self.assertEqual(state.image_depths, [])
        self.assertEqual(state.image_slices_filenames, [])
        self.assertEqual(state.depth_model_name, None)
        self.assertEqual(state.inpainting_model_name, None)
        self.assertEqual(state.positive_prompts, [])
        self.assertEqual(state.negative_prompts, [])
        self.assertEqual(state.server_address, None)
        self.assertEqual(state.api_key, None)

    def test_from_json_full(self):
        encoded = encode_string_with_nonce('sk-somesecretkey', 'appstate-test')
        json_data = '''
        {
            "filename": "appstate-test",
            "num_slices": 10,
            "imgThresholds": [0, 255],
            "image_depths": [0, 1, 2],
            "image_slices_filenames": [
                "appstate-test/image1.png",
                "appstate-test/image2.png",
                "appstate-test/image3.png"],
            "depth_model_name": "model",
            "inpainting_model_name": "inpainting",
            "positive_prompts": ["one fish", "two fish", "red fish"],
            "negative_prompts": ["blue fish", "new fish", "old fish"],
            "server_address": "http://localhost:8000",
            "api_key": "ENCODED",
            "camera_distance": 1.0,
            "max_distance": 2.0,
            "focal_length": 3.0,
            "mesh_displacement": 4.0
        }
        '''
        json_data = json_data.replace('ENCODED', encoded)
        
        state = AppState.from_json(json_data)
        self.assertIsInstance(state, AppState)
        self.assertEqual(state.filename, 'appstate-test')
        self.assertEqual(state.num_slices, 10)
        self.assertEqual(state.imgThresholds, [0, 255])
        self.assertEqual(state.image_depths, [0, 1, 2])
        self.assertEqual(state.image_slices_filenames, [
                         "appstate-test/image1.png",
                         "appstate-test/image2.png",
                         "appstate-test/image3.png"])
        self.assertEqual(state.depth_model_name, "model")
        self.assertEqual(state.inpainting_model_name, "inpainting")
        self.assertEqual(state.positive_prompts, [
            "one fish", "two fish", "red fish"])
        self.assertEqual(state.negative_prompts, [
            "blue fish", "new fish", "old fish"])
        self.assertEqual(state.server_address, "http://localhost:8000")
        self.assertEqual(state.api_key, "sk-somesecretkey")
        self.assertEqual(state.camera_distance, 1.0)
        self.assertEqual(state.max_distance, 2.0)
        self.assertEqual(state.focal_length, 3.0)
        self.assertEqual(state.mesh_displacement, 4.0)

    def test_from_json_partial(self):
        json_data = '''
            {
                "filename": "appstate-test",
                "num_slices": 10,
                "imgThresholds": [0, 255],
                "image_depths": [0, 1, 2],
                "image_slices_filenames": [
                    "appstate-test/image1.png",
                    "appstate-test/image2.png",
                    "appstate-test/image3.png"]
            }
            '''
        state = AppState.from_json(json_data)
        self.assertIsInstance(state, AppState)
        self.assertEqual(state.filename, 'appstate-test')
        self.assertEqual(state.num_slices, 10)
        self.assertEqual(state.imgThresholds, [0, 255])
        self.assertEqual(state.image_depths, [0, 1, 2])
        self.assertEqual(state.image_slices_filenames, [
            "appstate-test/image1.png",
            "appstate-test/image2.png",
            "appstate-test/image3.png"])
        self.assertEqual(state.depth_model_name, None)
        self.assertEqual(state.inpainting_model_name, None)
        self.assertEqual(state.positive_prompts, [""]*3)
        self.assertEqual(state.negative_prompts, [""]*3)
        self.assertEqual(state.server_address, None)
        self.assertEqual(state.api_key, None)


class TestUpscaling(unittest.TestCase):
    def setUp(self):
        self.state = AppState()
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_create_upscaler_default(self):
        self.state.pipeline_spec = None
        self.state._create_upscaler()
        self.assertIsNotNone(self.state.upscaler)

    @patch('controller.StabilityAI')
    def test_create_upscaler_stabilityai(self, mock_stabilityai):
        self.state.pipeline_spec = MagicMock()
        self.state.inpainting_model_name = 'stabilityai'
        self.state.api_key = 'test_api_key'
        self.state._create_upscaler()
        self.assertIsNotNone(self.state.upscaler)
        mock_stabilityai.assert_called_with('test_api_key')

    def test_create_upscaler_inpainting(self):
        self.state.pipeline_spec = MagicMock()
        self.state.inpainting_model_name = 'inpainting'
        self.state._create_upscaler()
        self.assertIsNotNone(self.state.upscaler)
        self.state.pipeline_spec.load_model.assert_called_once()

    @patch('controller.Upscaler')
    def test_upscale_image(self, mock_upscaler):
        mock_upscaler_instance = MagicMock()
        mock_upscaler.return_value = mock_upscaler_instance
        upscaled_image = self.state.upscale_image(self.image)
        mock_upscaler_instance.upscale_image_tiled.assert_called_with(
            self.image, overlap=64, prompt='', negative_prompt=''
        )
        self.assertIsNotNone(upscaled_image)

    @patch('controller.Upscaler')
    def test_upscale_image_with_prompts(self, mock_upscaler):
        mock_upscaler_instance = MagicMock()
        mock_upscaler.return_value = mock_upscaler_instance
        prompt = 'test prompt'
        negative_prompt = 'test negative prompt'
        self.state.upscale_image(self.image, prompt, negative_prompt)
        mock_upscaler_instance.upscale_image_tiled.assert_called_with(
            self.image, overlap=64, prompt=prompt, negative_prompt=negative_prompt
        )


class TestSaveImageSlices(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = 'appstate-random'

        # Add some initial slices
        self.state.add_slice(
            np.zeros((10, 10, 4), np.uint8), 5, positive_prompt="one", negative_prompt="two")
        self.state.add_slice(
            np.zeros((10, 10, 4), np.uint8), 10, positive_prompt="three", negative_prompt="four")
        self.state.add_slice(
            np.zeros((10, 10, 4), np.uint8), 15, positive_prompt="blue", negative_prompt="green")

    def test_save_image_slices(self):
        # Mock the image saving function
        with patch('controller.Image.Image.save') as mock_imwrite:
            self.state.save_image_slices(self.state.filename)

            # Check that the image slices were saved
            self.assertEqual(mock_imwrite.call_count, 3)
            mock_imwrite.assert_any_call('appstate-random/image_slice_0.png')
            mock_imwrite.assert_any_call('appstate-random/image_slice_1.png')
            mock_imwrite.assert_any_call('appstate-random/image_slice_2.png')


if __name__ == '__main__':
    unittest.main()
