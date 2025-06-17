import unittest
from PIL import Image
from .utils import (
    find_bounding_box,
    find_square_from_bounding_box,
    find_square_bounding_box,
    filename_add_version,
    filename_previous_version,
    highlight_selected_element,
    encode_string_with_nonce,
    decode_string_with_nonce,
)


class TestFindBoundingBox(unittest.TestCase):
    def test_find_bounding_box(self):
        # Create a mask image with a bounding box
        mask_image = Image.new("L", (100, 100))
        mask_image.paste(255, (20, 30, 80, 70))

        # Call the function
        result = find_bounding_box(mask_image, padding=0)

        # Define the expected result
        expected_result = (20, 30, 79, 69)

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_find_bounding_box_with_padding(self):
        # Create a mask image with a bounding box
        mask_image = Image.new("L", (100, 100))
        mask_image.paste(255, (20, 30, 80, 70))

        # Call the function with padding
        result = find_bounding_box(mask_image, padding=10)

        # Define the expected result
        expected_result = (10, 20, 89, 79)

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_find_bounding_box_with_empty_mask(self):
        # Create an empty mask image
        mask_image = Image.new("L", (100, 100))

        # Call the function
        result = find_bounding_box(mask_image)

        # Define the expected result
        expected_result = (0, 0, 0, 0)

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_find_square_from_bounding_box(self):
        # Define the bounding box coordinates
        xmin, ymin, xmax, ymax = 20, 30, 80, 70

        # Call the function
        result = find_square_from_bounding_box(xmin, ymin, xmax, ymax)

        # Define the expected result
        expected_result = (20, 20, 80, 80)

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_find_square_bounding_box(self):
        # Create an empty mask image
        mask_image = Image.new("L", (100, 100))
        mask_image.paste(255, (20, 30, 80, 70))

        result = find_square_bounding_box(mask_image, padding=10)

        expected_result = (11, 11, 89, 89)

        self.assertEqual(result, expected_result)

    def test_find_square_bounding_box_outside(self):
        # Create an empty mask image
        mask_image = Image.new("L", (100, 100))
        mask_image.paste(255, (20, 30, 80, 70))

        result = find_square_bounding_box(mask_image, padding=30)

        expected_result = (0, 0, 100, 100)

        self.assertEqual(result, expected_result)


class TestFilenameAddVersion(unittest.TestCase):
    def test_filename_add_version(self):
        # Define the input filename
        filename = "/path/to/image.png"

        # Call the function
        result = filename_add_version(filename)

        # Define the expected result
        expected_result = "/path/to/image_v2.png"

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

        # Call the function
        result = filename_add_version(result)

        # Define the expected result
        expected_result = "/path/to/image_v3.png"

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_filename_previous_version(self):
        # Define the input filename
        filename = "/path/to/image_v3.png"

        # Call the function
        result = filename_previous_version(filename)

        # Define the expected result
        expected_result = "/path/to/image_v2.png"

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

        # Call the function
        result = filename_previous_version(result)

        # Define the expected result
        expected_result = "/path/to/image.png"

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

        # Call the function with a filename without version
        filename = "/path/to/image.png"
        result = filename_previous_version(filename)

        # Define the expected result
        expected_result = None

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)


class TestHighLightSelectedElement(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.classnames = [
            "text-white",
            "text-brown",
            "text-black",
        ]
        self.highlight_classname = "color-is-selected"

    def test_highlight_selected_element(self):
        result = highlight_selected_element(
            self.classnames, 0, self.highlight_classname
        )

        expected_result = self.classnames.copy()
        expected_result[0] += f" {self.highlight_classname}"

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)

    def test_highlight_selected_element_remove(self):
        classnames = self.classnames.copy()
        expected_result = self.classnames
        classnames[1] += f" {self.highlight_classname}"

        result = highlight_selected_element(classnames, None, self.highlight_classname)

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)


class TestEncodeStringWithNonce(unittest.TestCase):
    def test_encode_string_with_nonce(self):
        # Define the plaintext and nonce
        plaintext = "Hello, World!"
        nonce = "unique-filename"

        # Call the function
        result = encode_string_with_nonce(plaintext, nonce)

        self.assertNotEqual(result, plaintext)

        decoded = decode_string_with_nonce(result, nonce)

        self.assertEqual(decoded, plaintext)

        decoded = decode_string_with_nonce(result, "wrong-nonce")

        self.assertNotEqual(decoded, plaintext)

    def test_decode_with_string_bad_data(self):
        nonce = "unique-filename"
        input = "bad-data"

        decoded = decode_string_with_nonce(input, nonce)

        self.assertEqual(decoded, None)


if __name__ == "__main__":
    unittest.main()
