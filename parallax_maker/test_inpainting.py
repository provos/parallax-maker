import numpy as np
import unittest

from .inpainting import find_nearest_alpha


class TestFindNearestAlpha(unittest.TestCase):
    def test_nearest_alpha_with_full_alpha(self):
        alpha = np.ones((3, 3)) * 255
        nearest_alpha = np.full((3, 3, 4, 2), -1, dtype=int)
        expected = np.zeros((3, 3, 4, 2))
        for i in range(3):
            for j in range(3):
                expected[i, j, :, :] = [[i, j], [i, j], [i, j], [i, j]]
        find_nearest_alpha(alpha, nearest_alpha)
        np.testing.assert_array_equal(nearest_alpha, expected)

    def test_nearest_alpha_with_partial_alpha(self):
        alpha = np.array([[255, 0, 255], [255, 255, 0], [0, 255, 255]])
        nearest_alpha = np.full((3, 3, 4, 2), -1, dtype=int)
        expected = np.zeros((3, 3, 4, 2))
        expected[0, 0, :, :] = [[0, 0], [0, 0], [0, 0], [0, 0]]
        expected[0, 1, :, :] = [[-1, -1], [0, 0], [1, 1], [0, 2]]
        expected[0, 2, :, :] = [[0, 2], [0, 2], [0, 2], [0, 2]]
        expected[1, 0, :, :] = [[1, 0], [1, 0], [1, 0], [1, 0]]
        expected[1, 1, :, :] = [[1, 1], [1, 1], [1, 1], [1, 1]]
        expected[1, 2, :, :] = [[0, 2], [1, 1], [2, 2], [-1, -1]]
        expected[2, 0, :, :] = [[1, 0], [-1, -1], [-1, -1], [2, 1]]
        expected[2, 1, :, :] = [[2, 1], [2, 1], [2, 1], [2, 1]]
        expected[2, 2, :, :] = [[2, 2], [2, 2], [2, 2], [2, 2]]
        find_nearest_alpha(alpha, nearest_alpha)
        np.testing.assert_array_equal(nearest_alpha, expected)


if __name__ == "__main__":
    unittest.main()
