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
    IMAGE_FILE = 'input_image.png'
    DEPTH_MAP_FILE = 'depth_map.png'

    cache = {}

    def __init__(self):
        self.filename = None
        self.num_slices = 5
        self.imgData = None # PIL image
        self.imgThresholds = None
        self.depthMapData = None # numpy array
        self.image_slices_filenames = []
        
        self.positive_prompt = ""
        self.negative_prompt = ""
        
        # no JSON serialization for items below
        self.image_slices = []
        self.selected_slice = 0
        self.pipeline_spec = None # PipelineSpec()
        self.selected_inpainting = None
        
    def mask_filename(self, slice_index):
        """Returns the mask filename for the specified slice index."""
        assert slice_index >=0 and slice_index < len(self.image_slices_filenames)
        file_path = Path(self.image_slices_filenames[slice_index])
        return file_path.parent / f"{file_path.stem}_mask.png"
        
    def save_image_mask(self, slice_index, mask):
        """Saves the PIL image mask to a png file for the specified slice index."""
        mask_path = self.mask_filename(slice_index)
        mask.save(str(mask_path))
        
        return str(mask_path)

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
            if not isinstance(slice_image, Image.Image):
                slice_image = Image.fromarray(slice_image, mode='RGBA')
            output_image_path = self.image_slices_filenames[i]
            print(f"Saving image slice: {output_image_path}")
            slice_image.save(str(output_image_path))

    def to_file(self, file_path, save_image_slices=True):
        """
        Save the current state to a file.

        Args:
            file_path (str): The file path to save the state.
            save_image_slices (bool, optional): Whether to save the image slices. Defaults to True.
        """
        file_path = Path(file_path)
        # check if file path is a directory. if not create it
        if not file_path.is_dir():
            file_path.mkdir()
            
        # save the input image
        img_file = file_path / AppState.IMAGE_FILE
        self.imgData.save(str(img_file))
        
        # save the depth map
        if self.depthMapData is not None:
            depth_map_file = file_path / AppState.DEPTH_MAP_FILE
            image = Image.fromarray(self.depthMapData)
            image.save(str(depth_map_file))

        # save the image slices
        if save_image_slices:
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

        state.fill_from_files(file_path)
        
        return state
    
    def fill_from_files(self, file_path):
        """
        This method reads the input image and depth map files from the specified file path and updates the state
        with the corresponding image data and depth map data.

        Args:
            file_path (str): The file path to load the state from.

        Returns:
            None
        """
        # read the input image
        img_file = Path(file_path) / AppState.IMAGE_FILE
        self.imgData = Image.open(img_file)
        self.read_image_slices(file_path)

        # read the depth map and turn it into a numpy array
        depth_map_file = Path(file_path) / AppState.DEPTH_MAP_FILE
        if depth_map_file.exists():
            self.depthMapData = np.array(Image.open(depth_map_file))

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
            state.filename = file_path
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
            'filename': self.filename,
            'num_slices': self.num_slices,
            'imgThresholds': self.imgThresholds,
            'image_slices_filenames': self.image_slices_filenames,
            'positive_prompt': self.positive_prompt,
            'negative_prompt': self.negative_prompt,
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

        state.filename = data['filename']
        state.num_slices = data['num_slices'] if 'num_slices' in data else 5
        state.imgThresholds = data['imgThresholds']
        state.image_slices_filenames = data['image_slices_filenames']
        state.positive_prompt = data['positive_prompt'] if 'positive_prompt' in data else ''
        state.negative_prompt = data['negative_prompt'] if 'negative_prompt' in data else ''
        return state

