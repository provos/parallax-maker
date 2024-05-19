import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from instance import SegmentationModel


class TestSegmentationModel(unittest.TestCase):

    def setUp(self):
        self.model = SegmentationModel("sam")
        self.mock_image = Image.new('RGB', (100, 100), color='white')
        self.mock_point = (50, 50)
        self.mock_mask = np.zeros((100, 100), np.uint8)
        self.mock_mask[40:60, 40:60] = 255

        patcher = patch.object(
            SegmentationModel, 'segment_image', return_value=self.mock_mask)
        self.mock_segment_image = patcher.start()
        self.addCleanup(patcher.stop)

        patcher2 = patch.object(
            SegmentationModel, '_get_mask_at_point_function', return_value=lambda point: self.mock_mask)
        self.mock_get_mask_at_point_function = patcher2.start()
        self.addCleanup(patcher2.stop)

    def test_transform(self):
        transformations = ['identity', 'rotate_90',
                           'rotate_180', 'rotate_270', 'flip_h', 'flip_v']
        for transformation in transformations:
            transformed_image, transformed_point = self.model._transform(
                self.mock_image, self.mock_point, transformation)
            self.assertIsInstance(transformed_image, Image.Image)
            self.assertIsInstance(transformed_point, tuple)
            self.assertEqual(len(transformed_point), 2)

    def test_rotate_point(self):
        angles = [0, 90, 180, 270]
        image_size = (100, 100)
        for angle in angles:
            rotated_point = self.model._rotate_point(
                self.mock_point, angle, image_size)
            self.assertIsInstance(rotated_point, tuple)
            self.assertEqual(len(rotated_point), 2)

    def test_inverse_transform(self):
        transformations = ['identity', 'rotate_90',
                           'rotate_180', 'rotate_270', 'flip_h', 'flip_v']
        for transformation in transformations:
            inverse_transformed_mask = self.model._inverse_transform(
                self.mock_mask, transformation)
            self.assertIsInstance(inverse_transformed_mask, np.ndarray)

    def test_filter_mask(self):
        masks = []
        for _ in range(5):
            m = np.zeros((100, 100), np.uint8)
            m[40:60, 40:60] = 255
            masks.append(Image.fromarray(m))

        filtered_masks = self.model._filter_mask(masks)
        self.assertIsInstance(filtered_masks, list)
        self.assertGreater(len(filtered_masks), 0)

    @patch.object(SegmentationModel, '_get_mask_at_point_function', return_value=lambda point: np.zeros((100, 100), np.uint8))
    def test_mask_at_point_blended(self, mock_get_mask_at_point_function):
        self.model.image = self.mock_image
        blended_mask = self.model.mask_at_point_blended(self.mock_point)
        self.assertIsInstance(blended_mask, np.ndarray)


if __name__ == '__main__':
    unittest.main()
