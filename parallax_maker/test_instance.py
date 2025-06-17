import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from .instance import SegmentationModel


class TestSegmentationModel(unittest.TestCase):

    def setUp(self):
        self.model = SegmentationModel("sam")
        self.mock_image = Image.new("RGB", (120, 90), color="white")
        self.mock_point = (50, 50)
        self.mock_points = [(50, 50), (60, 60), (70, 70)]
        self.mock_mask = np.zeros((90, 120), np.uint8)
        self.mock_mask[40:60, 40:60] = 255

        patcher = patch.object(
            SegmentationModel, "segment_image_mask2former", return_value=self.mock_mask
        )
        self.mock_segment_image = patcher.start()
        self.addCleanup(patcher.stop)

        patcher2 = patch.object(
            SegmentationModel,
            "_get_mask_at_point_function",
            return_value=lambda point: self.mock_mask,
        )
        self.mock_get_mask_at_point_function = patcher2.start()
        self.addCleanup(patcher2.stop)

    def test_transform(self):
        transformations = [
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "flip_h",
            "flip_v",
        ]
        for transformation in transformations:
            transformed_image = self.model._transform_image(
                self.mock_image, transformation
            )
            transformed_point = self.model._transform_point(
                self.mock_point, transformation, self.mock_image.size
            )
            self.assertIsInstance(transformed_image, Image.Image)
            self.assertIsInstance(transformed_point, tuple)
            self.assertEqual(len(transformed_point), 2)

    def test_rotate_point_single(self):
        angles = [0, 90, 180, 270]
        image_size = (120, 90)
        for angle in angles:
            rotated_point = self.model._rotate_point(self.mock_point, angle, image_size)
            self.assertIsInstance(rotated_point, tuple)
            self.assertEqual(len(rotated_point), 2)

    def test_transform_point_multiple(self):
        transformations = [
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "flip_h",
            "flip_v",
        ]
        image_size = (120, 90)
        for transformation in transformations:
            rotated_point = self.model._transform_point(
                self.mock_points, transformation, image_size
            )
            self.assertIsInstance(rotated_point, list)
            for point in rotated_point:
                self.assertIsInstance(point, tuple)
                self.assertEqual(len(point), 2)

    def test_inverse_transform(self):
        transformations = [
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "flip_h",
            "flip_v",
        ]
        for transformation in transformations:
            transformed_image = self.model._transform_image(
                self.mock_image, transformation
            )
            inverse_transformed_mask = self.model._inverse_transform(
                np.array(transformed_image), transformation
            )
            self.assertIsInstance(inverse_transformed_mask, np.ndarray)
            it_mask = Image.fromarray(inverse_transformed_mask)
            self.assertEqual(self.mock_image.size, it_mask.size)

    def test_filter_mask(self):
        masks = []
        for _ in range(5):
            m = np.zeros((90, 120), np.uint8)
            m[40:60, 40:60] = 255
            masks.append(Image.fromarray(m))

        filtered_masks = self.model._filter_mask(masks)
        self.assertIsInstance(filtered_masks, list)
        self.assertGreater(len(filtered_masks), 0)

    @patch.object(SegmentationModel, "_get_mask_at_point_function")
    def test_mask_at_point_blended(self, mock_get_mask_at_point_function):
        def mock_mask_function(point):
            width, height = self.model.image.size
            return np.zeros((height, width), np.uint8)

        mock_get_mask_at_point_function.return_value = mock_mask_function

        self.model.image = self.mock_image
        blended_mask = self.model.mask_at_point_blended(self.mock_point)
        self.assertIsInstance(blended_mask, np.ndarray)


if __name__ == "__main__":
    unittest.main()
