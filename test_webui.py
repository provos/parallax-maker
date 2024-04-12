import unittest
from webui import filename_add_version
from pathlib import Path


class TestFilenameAddVersion(unittest.TestCase):
    def test_without_version(self):
        filename = "image.png"
        expected = "image_v2.png"
        self.assertEqual(filename_add_version(filename), expected)

    def test_with_version(self):
        filename = "image_v2.png"
        expected = "image_v3.png"
        self.assertEqual(filename_add_version(filename), expected)
        
    def test_with_directory(self):
        filename = "dir/image.png"
        expected = "dir/image_v2.png"
        self.assertEqual(filename_add_version(filename), expected)


if __name__ == '__main__':
    unittest.main()
