import unittest
from .segmentation import (
    analyze_depth_histogram,
    blend_with_alpha,
    generate_simple_thresholds,
)
import numpy as np


class TestAnalyzeDepthHistogram(unittest.TestCase):
    def test_analyze_depth_histogram(self):
        # Create a dummy depth map
        depth_map = np.array([[0, 20, 30], [40, 50, 60], [70, 80, 255]])

        # Define the number of slices
        num_slices = 5

        # Call the function
        result = analyze_depth_histogram(depth_map, num_slices=num_slices)

        # Define the expected result
        expected_result = [0, 30, 40, 60, 70, 255]

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_analyze_depth_histogram_with_different_slices(self):
        # Create a dummy depth map
        depth_map = np.array([[0, 20, 30], [40, 50, 60], [70, 80, 255]])

        # Define the number of slices
        num_slices = 3

        # Call the function
        result = analyze_depth_histogram(depth_map, num_slices=num_slices)

        # Define the expected result
        expected_result = [0, 40, 70, 255]

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_analyze_depth_histogram_with_one_slice(self):
        # Create a dummy depth map
        depth_map = np.array([[0, 20, 30], [40, 50, 60], [70, 80, 255]])

        # Define the number of slices
        num_slices = 1

        # Call the function
        result = analyze_depth_histogram(depth_map, num_slices=num_slices)

        # Define the expected result
        expected_result = [0, 255]

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)


class TestGenerateSimpleThresholds(unittest.TestCase):
    def test_generate_simple_thresholds(self):
        # Create a dummy image
        image = np.array([[0, 20, 30], [40, 50, 60], [70, 80, 255]])

        # Define the number of slices
        num_slices = 5

        # Call the function
        result = generate_simple_thresholds(image, num_slices=num_slices)

        # Define the expected result
        expected_result = np.array([0, 51, 102, 153, 204, 255])

        # Assert that the result is as expected
        np.testing.assert_array_equal(result, expected_result)


class TestBlendWithAlpha(unittest.TestCase):
    def test_blend_with_alpha(self):
        # Create target image
        target_image = np.array(
            [
                [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
                [[255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255]],
                [[255, 255, 255, 255], [0, 0, 0, 255], [128, 128, 128, 255]],
            ]
        )

        # Create merge image
        merge_image = np.array(
            [
                [[0, 0, 0, 128], [128, 128, 128, 128], [255, 255, 255, 128]],
                [[128, 0, 0, 128], [0, 128, 0, 128], [0, 0, 128, 128]],
                [[128, 128, 0, 128], [128, 0, 128, 128], [0, 128, 128, 128]],
            ]
        )

        # Call the function
        blend_with_alpha(target_image, merge_image)

        # Define the expected result
        expected_result = np.array(
            [
                [[127, 0, 0, 255], [64, 191, 64, 255], [128, 128, 255, 255]],
                [[191, 127, 0, 255], [127, 64, 127, 255], [0, 127, 191, 255]],
                [[191, 191, 127, 255], [64, 0, 64, 255], [63, 128, 128, 255]],
            ]
        )

        # Assert that the result is as expected
        np.testing.assert_array_equal(target_image, expected_result)


if __name__ == "__main__":
    unittest.main()
