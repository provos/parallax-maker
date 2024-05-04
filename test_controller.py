import numpy as np
import unittest

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
        self.state.image_slices_filenames = [
            'invalid_root/image_slice_1.png',
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
            'invalid_filename.png'
        ]

        # Call the function and expect an AssertionError
        with self.assertRaises(AssertionError):
            self.state.check_pathnames()

if __name__ == '__main__':
    unittest.main()