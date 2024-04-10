# (c) 2024 Niels Provos

import base64
import json
import random
import string
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from pathlib import Path


class AppState:
    STATE_FILE = 'appstate.json'

    cache = {}

    def __init__(self):
        self.imgData = None
        self.imgThresholds = None
        self.depthMapData = None
        self.image_slices_filenames = []
        self.image_slices = []  # does not require JSON serialization

    def read_image_slices(self, file_path):
        self.image_slices = [
            cv2.cvtColor(cv2.imread(str(image_slice), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            for image_slice in self.image_slices_filenames]
        print(f"Loaded {len(self.image_slices)} image slices")

    def save_image_slices(self, file_path):
        """
        Save the image slices to the specified file path.

        Args:
            file_path (str): The file path to save the image slices.
        """
        file_path = Path(file_path)
        if len(self.image_slices_filenames) == 0:
            self.image_slices_filenames = [
                str(file_path / f"image_slice_{i}.png") for i in range(len(self.image_slices))]
        assert len(self.image_slices) == len(self.image_slices_filenames)
        for i, slice_image in enumerate(self.image_slices):
            output_image_path = self.image_slices_filenames[i]
            print(f"Saving image slice: {output_image_path}")
            cv2.imwrite(str(output_image_path), cv2.cvtColor(
                slice_image, cv2.COLOR_RGBA2BGRA))

    def to_file(self, file_path):
        """
        Save the current state to a file.

        Args:
            file_path (str): The file path to save the state.
        """
        file_path = Path(file_path)
        # check if file path is a directory. if not create it
        if not file_path.is_dir():
            file_path.mkdir()

        # save the image slices
        self.save_image_slices(file_path)

        state_file = file_path / AppState.STATE_FILE
        with open(str(state_file), 'w', encoding='utf-8') as file:
            file.write(self.to_json())

    @staticmethod
    def from_file(file_path):
        """
        Load the state from a file.

        Args:
            file_path (str): The file path to load the state from.

        Returns:
            AppState: The loaded state.
        """
        state_file = Path(file_path) / AppState.STATE_FILE
        with open(state_file, 'r', encoding='utf-8') as file:
            state = AppState.from_json(file.read())

        state.read_image_slices(file_path)

        return state

    @staticmethod
    def from_cache(file_path):
        """
        Retrieve the state from the cache or load it from a file.

        Args:
            file_path (str): The file path to load the state from.

        Returns:
            AppState: The loaded state.
        """
        if file_path in AppState.cache:
            return AppState.cache[file_path]
        state = AppState.from_file(file_path)
        AppState.cache[file_path] = state
        return state

    @staticmethod
    def from_file_or_new(file_path):
        """
        Load the state from a file or create a new state if the file path is None.

        Args:
            file_path (str): The file path to load the state from.

        Returns:
            tuple: A tuple containing the loaded state and the file path.
        """
        if file_path is None:
            state = AppState()
            file_path = AppState.random_filename()
        else:
            state = AppState.from_file(file_path)
        AppState.cache[file_path] = state
        return state, file_path

    @staticmethod
    def random_filename():
        """
        Generate a random filename.

        Returns:
            str: The random filename.
        """
        return 'appstate-'+''.join(random.choices(string.ascii_letters + string.digits, k=8))

    def to_json(self):
        """
        Convert the state to a JSON string.

        Returns:
            str: The JSON string representation of the state.
        """
        data = {
            'imgData': self._serialize_image(self.imgData),
            'imgThresholds': self.imgThresholds,
            'depthMapData': self._serialize_ndarray(self.depthMapData),
            'image_slices_filenames': self.image_slices_filenames
        }
        return json.dumps(data)

    @staticmethod
    def from_json(json_data):
        """
        Load the state from a JSON string.

        Args:
            json_data (str): The JSON string representation of the state.

        Returns:
            AppState: The loaded state.
        """
        state = AppState()
        if json_data is None:
            return state

        data = json.loads(json_data)

        state.imgData = AppState._deserialize_image(data['imgData'])
        state.imgThresholds = data['imgThresholds']
        state.depthMapData = AppState._deserialize_ndarray(data['depthMapData'])
        state.image_slices_filenames = data['image_slices_filenames']
        return state

    @staticmethod
    def _serialize_image(image):
        """
        Serialize the image to a base64-encoded string.

        Args:
            image (PIL.Image.Image): The image to serialize.

        Returns:
            str: The base64-encoded string representation of the image.
        """
        if image is None:
            return None
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    @staticmethod
    def _deserialize_image(image_base64):
        """
        Deserialize the image from a base64-encoded string.

        Args:
            image_base64 (str): The base64-encoded string representation of the image.

        Returns:
            PIL.Image.Image: The deserialized image.
        """
        if image_base64 is None:
            return None
        image_bytes = base64.b64decode(image_base64.encode('utf-8'))
        image = Image.open(BytesIO(image_bytes))
        return image

    @staticmethod
    def _serialize_ndarray(arr):
        """
        Serialize the ndarray to a list.

        Args:
            arr (numpy.ndarray): The ndarray to serialize.

        Returns:
            list: The serialized list representation of the ndarray.
        """
        if arr is None:
            return None
        return arr.tolist()

    @staticmethod
    def _deserialize_ndarray(array_list):
        """
        Deserialize the ndarray from a list.

        Args:
            array_list (list): The list representation of the ndarray.

        Returns:
            numpy.ndarray: The deserialized ndarray.
        """
        if array_list is None:
            return None
        return np.array(array_list)
