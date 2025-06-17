import numpy as np
import unittest
from .gltf import subdivide_geometry, triangle_indices_from_grid


class TestSubdivideGeometry(unittest.TestCase):
    def test_subdivide_2d(self):
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        subdivisions = 2
        dimension = 2
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [0.5, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        result = subdivide_geometry(coords, subdivisions, dimension)
        np.testing.assert_array_equal(result, expected)

    def test_subdivide_3d(self):
        coords = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        subdivisions = 2
        dimension = 3
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        result = subdivide_geometry(coords, subdivisions, dimension)
        np.testing.assert_array_equal(result, expected)


class TestTriangleIndicesFromGrid(unittest.TestCase):
    def test_triangle_indices_from_grid(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        expected = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)
        result = triangle_indices_from_grid(vertices)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
