import numpy as np
import unittest
from unittest.mock import mock_open, patch, MagicMock
from pathlib import Path
from PIL import Image
import numpy as np

from .utils import encode_string_with_nonce, decode_string_with_nonce
from .controller import AppState
from .slice import ImageSlice


class TestAddSlice(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = "test"

    def test_add_slice_normal(self):
        # Create a mock slice image and depth value
        slice_image1 = np.ones((10, 10, 4), np.uint8)
        depth1 = 5

        image_slice = ImageSlice(slice_image1, depth1)

        # Call the function
        insert1 = self.state.add_slice(image_slice)

        self.assertEqual(insert1, 0)
        self.assertEqual(len(self.state.image_slices), 1)
        self.assertEqual(self.state.image_slices[0].depth, depth1)

        slice_image2 = np.ones((10, 10, 4), np.uint8)
        depth2 = 10
        image_slice = ImageSlice(slice_image2, depth2)

        # Call the function
        insert2 = self.state.add_slice(image_slice)

        # Check whether this slice was added before the first slice
        self.assertEqual(insert2, 1)
        self.assertEqual(len(self.state.image_slices), 2)
        self.assertEqual(self.state.image_slices[0].depth, depth1)
        self.assertEqual(self.state.image_slices[1].depth, depth2)

    def test_add_slice_reverse(self):
        # Create a mock slice image and depth value
        slice_image1 = np.ones((10, 10, 4), np.uint8)
        depth1 = 5

        image_slice1 = ImageSlice(slice_image1, depth1)

        # Call the function
        insert1 = self.state.add_slice(image_slice1)

        self.assertEqual(insert1, 0)
        self.assertEqual(len(self.state.image_slices), 1)
        self.assertEqual(self.state.image_slices[0].depth, depth1)

        slice_image2 = np.ones((10, 10, 4), np.uint8)
        depth2 = 1

        image_slice2 = ImageSlice(slice_image2, depth2)

        # Call the function
        insert2 = self.state.add_slice(image_slice2)

        # Check whether this slice was added before the first slice
        self.assertEqual(insert2, 0)
        self.assertEqual(len(self.state.image_slices), 2)
        self.assertEqual(self.state.image_slices[0].depth, depth2)
        self.assertEqual(self.state.image_slices[1].depth, depth1)

        slice_image3 = np.ones((10, 10, 4), np.uint8)
        depth3 = 1

        image_slice3 = ImageSlice(slice_image3, depth3)

        # Call the function
        insert3 = self.state.add_slice(image_slice3)

        # Check that the slice was added and depth increased to avoid duplicates
        self.assertEqual(insert3, 1)
        self.assertEqual(len(self.state.image_slices), 3)
        self.assertEqual(self.state.image_slices[0].depth, depth2)
        self.assertEqual(self.state.image_slices[1].depth, depth3 + 1)

        # check that all the filenames are accurate
        expected = [
            "test/image_slice_1.png",
            "test/image_slice_2.png",
            "test/image_slice_0.png",
        ]

        image_slices_filenames = [slice.filename for slice in self.state.image_slices]

        self.assertEqual(image_slices_filenames, expected)


class TestCheckPathnames(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = "appstate-test"

    def test_check_pathnames_valid(self):
        # Set up valid pathnames
        image_slices_filenames = [
            "appstate-test/image_slice_1.png",
            "appstate-test/image_slice_2.png",
            "appstate-test/image_slice_0.png",
        ]

        self.state.image_slices = [
            ImageSlice(filename=filename) for filename in image_slices_filenames
        ]

        # Call the function
        self.state.check_pathnames()

    def test_check_pathnames_invalid_root(self):
        # Set up invalid root pathname
        self.state.filename = "invalid_root/../something"
        image_slices_filenames = [
            "appstate-test/image_slice_2.png",
            "appstate-test/image_slice_0.png",
        ]

        self.state.image_slices = [
            ImageSlice(filename=filename) for filename in image_slices_filenames
        ]

        # Call the function and expect an AssertionError
        with self.assertRaises(AssertionError):
            self.state.check_pathnames()

    def test_check_pathnames_invalid_filenames(self):
        # Set up invalid filenames
        image_slices_filenames = [
            "appstate-test/image_slice_1.png",
            "appstate-test/image_slice_2.png",
            "appstate-test/../../../invalid_filename.png",
        ]

        self.state.image_slices = [
            ImageSlice(filename=filename) for filename in image_slices_filenames
        ]

        # Call the function and expect an AssertionError
        with self.assertRaises(AssertionError):
            self.state.check_pathnames()


class TestChangeSliceDepth(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = "test"

        # Add some initial slices
        self.state.add_slice(
            ImageSlice(
                np.ones((10, 10, 4), np.uint8),
                5,
                positive_prompt="one",
                negative_prompt="two",
            )
        )
        self.state.add_slice(
            ImageSlice(
                np.ones((10, 10, 4), np.uint8),
                10,
                positive_prompt="three",
                negative_prompt="four",
            )
        )
        self.state.add_slice(
            ImageSlice(
                np.ones((10, 10, 4), np.uint8),
                15,
                positive_prompt="blue",
                negative_prompt="green",
            )
        )

    def test_change_slice_depth_same_depth(self):
        # Get the initial state
        initial_depths = [image_slice.depth for image_slice in self.state.image_slices]

        # Change the depth of the second slice to the same depth
        slice_index = 1
        depth = 10
        new_index = self.state.change_slice_depth(slice_index, depth)

        # Check that the slice index remains the same
        self.assertEqual(new_index, slice_index)

        # Check that the slices and depths remain unchanged
        new_depths = [image_slice.depth for image_slice in self.state.image_slices]
        self.assertEqual(new_depths, initial_depths)

    def test_change_slice_depth_valid(self):
        # Get the initial state
        initial_slices = self.state.image_slices.copy()

        # Change the depth of the second slice to a new depth
        slice_index = 1
        depth = 3
        new_index = self.state.change_slice_depth(slice_index, depth)

        # Check that the new index is returned
        self.assertEqual(new_index, 0)

        # Check that the depth of the second slice is updated
        image_slice = self.state.image_slices[new_index]
        self.assertEqual(image_slice.depth, depth)
        self.assertEqual(image_slice.positive_prompt, "three")
        self.assertEqual(image_slice.negative_prompt, "four")

        # Check that the other slices and depths remain unchanged
        self.assertEqual(self.state.image_slices[1], initial_slices[0])
        self.assertEqual(self.state.image_slices[2], initial_slices[2])

    def test_change_slice_depth_invalid_index(self):
        # Get the initial state
        initial_slices = self.state.image_slices.copy()

        # Try to change the depth of an invalid slice index
        slice_index = 3
        depth = 12
        with self.assertRaises(AssertionError):
            self.state.change_slice_depth(slice_index, depth)

        # Check that the slices and depths remain unchanged
        self.assertEqual(self.state.image_slices, initial_slices)


class TestFromJson(unittest.TestCase):
    def test_from_json_empty(self):
        json_data = None
        state = AppState.from_json(json_data)
        self.assertIsInstance(state, AppState)
        self.assertEqual(state.image_slices, [])
        self.assertEqual(state.filename, None)
        self.assertEqual(state.num_slices, 5)
        self.assertEqual(state.imgThresholds, None)
        self.assertEqual(state.depth_model_name, None)
        self.assertEqual(state.inpainting_model_name, None)
        self.assertEqual(state.server_address, None)
        self.assertEqual(state.api_key, None)

    def test_from_json_full(self):
        encoded = encode_string_with_nonce("sk-somesecretkey", "appstate-test")
        json_data = """
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
        """
        json_data = json_data.replace("ENCODED", encoded)

        state = AppState.from_json(json_data)
        self.assertIsInstance(state, AppState)
        self.assertEqual(state.filename, "appstate-test")
        self.assertEqual(state.num_slices, 10)
        self.assertEqual(state.imgThresholds, [0, 255])

        image_depths = [image_slice.depth for image_slice in state.image_slices]
        self.assertEqual(image_depths, [0, 1, 2])

        image_slices_filenames = [
            image_slice.filename for image_slice in state.image_slices
        ]
        self.assertEqual(
            image_slices_filenames,
            [
                "appstate-test/image1.png",
                "appstate-test/image2.png",
                "appstate-test/image3.png",
            ],
        )
        self.assertEqual(state.depth_model_name, "model")
        self.assertEqual(state.inpainting_model_name, "inpainting")

        positive_prompts = [
            image_slice.positive_prompt for image_slice in state.image_slices
        ]
        self.assertEqual(positive_prompts, ["one fish", "two fish", "red fish"])

        negative_prompts = [
            image_slice.negative_prompt for image_slice in state.image_slices
        ]
        self.assertEqual(negative_prompts, ["blue fish", "new fish", "old fish"])

        self.assertEqual(state.server_address, "http://localhost:8000")
        self.assertEqual(state.api_key, "sk-somesecretkey")
        self.assertEqual(state.camera.camera_distance, 1.0)
        self.assertEqual(state.camera.max_distance, 2.0)
        self.assertEqual(state.camera.focal_length, 3.0)
        self.assertEqual(state.mesh_displacement, 4.0)

    def test_from_json_partial(self):
        json_data = """
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
            """
        state = AppState.from_json(json_data)
        self.assertIsInstance(state, AppState)
        self.assertEqual(state.filename, "appstate-test")
        self.assertEqual(state.num_slices, 10)
        self.assertEqual(state.imgThresholds, [0, 255])

        image_depths = [image_slice.depth for image_slice in state.image_slices]
        self.assertEqual(image_depths, [0, 1, 2])

        image_slices_filenames = [
            image_slice.filename for image_slice in state.image_slices
        ]
        self.assertEqual(
            image_slices_filenames,
            [
                "appstate-test/image1.png",
                "appstate-test/image2.png",
                "appstate-test/image3.png",
            ],
        )
        self.assertEqual(state.depth_model_name, None)
        self.assertEqual(state.inpainting_model_name, None)

        positive_prompts = [
            image_slice.positive_prompt for image_slice in state.image_slices
        ]
        self.assertEqual(positive_prompts, [""] * 3)

        negative_prompts = [
            image_slice.negative_prompt for image_slice in state.image_slices
        ]
        self.assertEqual(negative_prompts, [""] * 3)
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

    @patch("parallax_maker.controller.StabilityAI")
    def test_create_upscaler_stabilityai(self, mock_stabilityai):
        self.state.pipeline_spec = MagicMock()
        self.state.inpainting_model_name = "stabilityai"
        self.state.api_key = "test_api_key"
        self.state._create_upscaler()
        self.assertIsNotNone(self.state.upscaler)
        mock_stabilityai.assert_called_with("test_api_key")

    def test_create_upscaler_inpainting(self):
        self.state.pipeline_spec = MagicMock()
        self.state.inpainting_model_name = "inpainting"
        self.state._create_upscaler()
        self.assertIsNotNone(self.state.upscaler)
        self.state.pipeline_spec.load_model.assert_called_once()

    @patch("parallax_maker.controller.Upscaler")
    def test_upscale_image(self, mock_upscaler):
        mock_upscaler_instance = MagicMock()
        mock_upscaler.return_value = mock_upscaler_instance
        upscaled_image = self.state.upscale_image(self.image)
        mock_upscaler_instance.upscale_image_tiled.assert_called_with(
            self.image, overlap=64, prompt="", negative_prompt=""
        )
        self.assertIsNotNone(upscaled_image)

    @patch("parallax_maker.controller.Upscaler")
    def test_upscale_image_with_prompts(self, mock_upscaler):
        mock_upscaler_instance = MagicMock()
        mock_upscaler.return_value = mock_upscaler_instance
        prompt = "test prompt"
        negative_prompt = "test negative prompt"
        self.state.upscale_image(self.image, prompt, negative_prompt)
        mock_upscaler_instance.upscale_image_tiled.assert_called_with(
            self.image, overlap=64, prompt=prompt, negative_prompt=negative_prompt
        )


class TestSaveImageSlices(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.state = AppState()
        self.state.filename = "appstate-random"

        # Add some initial slices
        self.state.add_slice(
            ImageSlice(
                np.zeros((10, 10, 4), np.uint8),
                5,
                positive_prompt="one",
                negative_prompt="two",
            )
        )
        self.state.add_slice(
            ImageSlice(
                np.zeros((10, 10, 4), np.uint8),
                10,
                positive_prompt="three",
                negative_prompt="four",
            )
        )
        self.state.add_slice(
            ImageSlice(
                np.zeros((10, 10, 4), np.uint8),
                15,
                positive_prompt="blue",
                negative_prompt="green",
            )
        )

    def test_save_image_slices(self):
        # Mock the image saving function
        with patch("parallax_maker.slice.Image.Image.save") as mock_imwrite:
            self.state.save_image_slices(self.state.filename)

            # Check that the image slices were saved
            self.assertEqual(mock_imwrite.call_count, 3)
            mock_imwrite.assert_any_call("appstate-random/image_slice_0.png")
            mock_imwrite.assert_any_call("appstate-random/image_slice_1.png")
            mock_imwrite.assert_any_call("appstate-random/image_slice_2.png")


class TestToFile(unittest.TestCase):

    def setUp(self):
        # Creating a mock AppState instance
        self.app_state = AppState()

        self.file_path = Path("appstate-random")

        # Mocking methods directly on the instance
        self.img_data_mock = MagicMock()
        self.app_state.imgData = self.img_data_mock

        self.depth_map_data_mock = MagicMock()
        self.app_state.depthMapData = self.depth_map_data_mock

    @patch("parallax_maker.controller.Image.fromarray")
    @patch("parallax_maker.controller.Path.mkdir")
    @patch("parallax_maker.controller.Path.exists")
    @patch("parallax_maker.controller.shutil.move")
    @patch("parallax_maker.controller.open", new_callable=mock_open)
    @patch("parallax_maker.controller.Path.unlink")
    @patch.object(AppState, "to_json", return_value='{"state": "dummy state"}')
    def test_to_file(
        self,
        mock_to_json,
        mock_unlink,
        mock_open_func,
        mock_shutil_move,
        mock_exists,
        mock_mkdir,
        mock_fromarray,
    ):
        mock_mkdir.return_value = None
        # file_path, state_file, backup_file exists
        mock_exists.side_effect = [True, True, True]
        mock_fromarray.return_value = MagicMock()

        self.app_state.to_file(self.file_path)

        # Check if the directory was created
        mock_mkdir.assert_called_once_with()

        # Check if files were saved
        self.img_data_mock.save.assert_called_once_with(
            str(self.file_path / AppState.IMAGE_FILE)
        )
        mock_fromarray.return_value.save.assert_called_once_with(
            str(self.file_path / AppState.DEPTH_MAP_FILE)
        )

        # Check if state file operations were performed correctly
        state_file = self.file_path / AppState.STATE_FILE
        temp_file = state_file.with_suffix(".tmp")
        backup_file = state_file.with_suffix(".bak")
        mock_open_func.assert_called_once_with(temp_file, "w", encoding="utf-8")

        # Read the written content
        handle = mock_open_func()
        handle.write.assert_called_once_with('{"state": "dummy state"}')

        mock_shutil_move.assert_any_call(state_file, backup_file)
        mock_shutil_move.assert_any_call(temp_file, state_file)
        mock_unlink.assert_called_once()

    @patch("parallax_maker.controller.Image.fromarray")
    @patch("parallax_maker.controller.Path.mkdir")
    @patch("parallax_maker.controller.Path.exists")
    @patch("parallax_maker.controller.shutil.move")
    @patch("parallax_maker.controller.open", new_callable=mock_open)
    @patch("parallax_maker.controller.Path.unlink")
    @patch.object(AppState, "to_json", return_value='{"state": "dummy state"}')
    def test_to_file_restore_backup_on_error(
        self,
        mock_to_json,
        mock_unlink,
        mock_open_func,
        mock_shutil_move,
        mock_exists,
        mock_mkdir,
        mock_fromarray,
    ):
        mock_mkdir.return_value = None
        # state_file, no backup file
        mock_exists.side_effect = [True, True, False]
        # First call succeeds, second (inside state write) fails
        mock_fromarray.return_value = MagicMock()

        # Configure mock_open to work with 'with' statement
        mock_file_handle = mock_open_func.return_value
        mock_file_handle.__enter__.return_value = mock_file_handle
        mock_open_func.side_effect = [IOError]

        with self.assertRaises(IOError):
            self.app_state.to_file(self.file_path)

        # Check if appropriate cleanup was done
        temp_file = (self.file_path / AppState.STATE_FILE).with_suffix(".tmp")
        backup_file = (self.file_path / AppState.STATE_FILE).with_suffix(".bak")

        mock_unlink.assert_called_once()
        mock_shutil_move.assert_called_once_with(
            backup_file, self.file_path / AppState.STATE_FILE
        )


class TestApplyMask(unittest.TestCase):
    def setUp(self):
        self.state = AppState()
        self.state.imgData = Image.new("RGB", (100, 100))
        self.state.result_tinted = Image.new("RGB", (100, 100), (255, 0, 0))
        self.state.grayscale_tinted = Image.new("RGB", (100, 100), (128, 128, 128))

    def test_apply_mask_with_pil_image(self):
        mask = Image.new("L", (100, 100), 128)

        result = self.state.apply_mask(self.state.imgData, mask)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.state.imgData.size)

    def test_apply_mask_with_numpy_array(self):
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = self.state.apply_mask(self.state.imgData, mask)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.state.imgData.size)

    def test_apply_mask_with_none_image(self):
        image = None
        mask = Image.new("L", (100, 100))

        with self.assertRaises(AttributeError):
            self.state.apply_mask(image, mask)

    def test_apply_mask_creates_tints_when_needed(self):
        mask = Image.new("L", (100, 100), 128)
        self.state.result_tinted = None
        self.state.grayscale_tinted = None

        result = self.state.apply_mask(self.state.imgData, mask)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.state.imgData.size)

    def test_apply_mask_uses_precomputed_tints(self):
        mask = Image.new("L", (100, 100), 128)

        result = self.state.apply_mask(self.state.imgData, mask)

        self.assertNotEqual(self.state.result_tinted, None)
        self.assertNotEqual(self.state.grayscale_tinted, None)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.state.imgData.size)


if __name__ == "__main__":
    unittest.main()
