import cv2
from pathlib import Path
from PIL import Image
import numpy as np

from utils import filename_add_version, filename_previous_version
from camera import Camera


class ImageSlice:
    __slots__ = ('image', '_depth', '_filename',
                 'positive_prompt', 'negative_prompt')

    def __init__(self, image=None, depth=-1, filename=None, positive_prompt='', negative_prompt=''):
        self.image = image
        self._depth = depth
        self._filename = filename

        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        if not isinstance(value, (float, int)) or value < -1:
            raise ValueError(
                "depth must be a non-negative number or -1 for unknown depth")
        self._depth = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, Path) and not isinstance(value, str) and value is not None:
            raise ValueError("filename must be a Path, str object or None")
        self._filename = value

    def __eq__(self, other):
        if not isinstance(other, ImageSlice):
            return False
        return (self.image == other.image and
                self.depth == other.depth and
                self.filename == other.filename)

    def create_card(self, image_height: int, image_width: int, cam: Camera):
        z = cam.max_distance * ((255 - self.depth) / 255.0)

        fl_px = cam.focal_length_px(image_width)

        # Calculate the 3D points of the card corners
        card_width = (image_width * (z + cam.camera_distance)) / fl_px
        card_height = (image_height * (z + cam.camera_distance)) / fl_px

        card_corners_3d = np.array([
            [-card_width / 2, -card_height / 2, z],
            [card_width / 2, -card_height / 2, z],
            [card_width / 2, card_height / 2, z],
            [-card_width / 2, card_height / 2, z]
        ], dtype=np.float32)

        return card_corners_3d

    def save_image(self):
        slice_image = self.image
        if not isinstance(slice_image, Image.Image):
            slice_image = Image.fromarray(slice_image, mode='RGBA')
        output_image_path = self.filename
        print(f"Saving image slice: {output_image_path}")
        slice_image.save(str(output_image_path))

    def read_image(self):
        img = cv2.imread(str(self.filename), cv2.IMREAD_UNCHANGED)
        self.image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    def new_version(self, image=None, save=True):
        """
        Creates a new version of the image and saves it if specified.

        Args:
            image (np.ndarray, optional): The new image to be set. Defaults to None.
            save (bool, optional): Whether to save the new image. Defaults to True.

        Returns:
            str: The filename of the new version of the image.
        """
        assert isinstance(image, np.ndarray)
        if image is not None:
            self.image = image
        self.filename = filename_add_version(self.filename)
        if save:
            self.save_image()
        return self.filename

    def can_undo(self, forward=False):
        """
        Check if it is possible to undo the specified image slice.

        Args:
            forward (bool, optional): If True, check for the next version of the slice. 
                                      If False, check for the previous version. Defaults to False.

        Returns:
            bool: True if the specified slice version exists, False otherwise.
        """
        if forward:
            filename = filename_add_version(self.filename)
        else:
            filename = filename_previous_version(self.filename)

        if filename is None:
            return False

        return Path(filename).exists()

    def undo(self, forward=False):
        """
        Undo the specified image slice.

        Args:
            forward (bool, optional): If True, undo the next version of the slice. 
                          If False, undo the previous version. Defaults to False.

        Returns:
            bool: True if the undo operation is successful, False otherwise.
        """
        if not self.can_undo(forward):
            return False

        if forward:
            filename = filename_add_version(self.filename)
        else:
            filename = filename_previous_version(self.filename)

        if filename is None:
            return False

        self.filename = filename
        self.read_image()
        return True
