# (c) 2024 Niels Provos

import os
import json
import random
import string
import time
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from pathlib import Path
from utils import filename_previous_version, filename_add_version, apply_color_tint
from segmentation import mask_from_depth
from upscaler import Upscaler


class AppState:
    SRV_DIR = 'tmp-images'
    STATE_FILE = 'appstate.json'
    IMAGE_FILE = 'input_image.png'
    DEPTH_MAP_FILE = 'depth_map.png'
    MAIN_IMAGE = 'main_image.bmp'
    MODEL_FILE = 'model.gltf'
    WORKFLOW = 'workflow.json'

    cache = {}

    def __init__(self):
        self.filename = None
        self.num_slices = 5
        self.imgData = None  # PIL image
        self.imgThresholds = None
        self.depthMapData = None  # numpy array
        self.image_depths = []
        self.image_slices_filenames = []

        self.depth_model_name = None

        self.positive_prompt = ""
        self.negative_prompt = ""
        
        self.server_address = None

        # no JSON serialization for items below
        self.image_slices = []
        self.selected_slice = None
        self.pipeline_spec = None  # PipelineSpec() for inpainting
        self.depth_estimation_model = None # DepthEstimationModel() for depth estimation
        self.segmentation_model = None # SegmentationModel() for segmentation
        self.selected_inpainting = None
        self.result_tinted = None
        self.grayscale_tinted = None
        self.slice_pixel = None
        self.slice_pixel_depth = None
        self.slice_mask = None
        self.upscaler = None

        # XXX - make this configurable
        self.camera_position = np.array([0, 0, -100], dtype=np.float32)
        self.camera_distance = 100.0
        self.max_distance = 500.0
        self.focal_length = 100.0

    def create_tints(self):
        """Precomputes the tings for visualizing slices."""
        # Apply the main color tint to the original image
        self.result_tinted = apply_color_tint(self.imgData, (150, 255, 0), 0.3)

        # Convert the image to grayscale and back to RGB
        grayscale = self.imgData.convert('L').convert('RGB')
        self.grayscale_tinted = apply_color_tint(grayscale, (0, 0, 150), 0.1)

    def apply_mask(self, image, mask):
        # Prepare masks; make sure they are the right size and mode
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)
        mask = mask.convert('L')
        
        if image is not self.imgData:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert('RGB')
            result_tinted = apply_color_tint(image, (150, 255, 0), 0.3)
            grayscale_tinted = apply_color_tint(image.convert('L').convert('RGB'), (0, 0, 150), 0.1)
        else:
            if self.result_tinted is None or self.grayscale_tinted is None:
                self.create_tints()
            result_tinted = self.result_tinted
            grayscale_tinted = self.grayscale_tinted

        # Combine the tinted and the grayscale image
        final_result = Image.composite(
            result_tinted, grayscale_tinted, mask)

        return final_result
    
    def change_slice_depth(self, slice_index, depth):
        """Changes the depth of the slice at the specified index."""
        assert slice_index >= 0 and slice_index < len(self.image_slices)
        
        if depth == self.image_depths[slice_index]:
            return slice_index
        
        filename = self.image_slices_filenames[slice_index]
        image = self.image_slices[slice_index]
        
        # remove it from the lists
        self.image_depths.pop(slice_index)
        self.image_slices_filenames.pop(slice_index)
        self.image_slices.pop(slice_index)
        
        return self.add_slice(image, depth, filename=filename)
    
    def add_slice(self, slice_image, depth, filename=None):
        """Adds the image as a new slice at the provided depth."""
        if filename is None:
            filename = str(Path(self.filename) / f"image_slice_{len(self.image_slices)}.png")
        # find the index where the depth should be inserted
        index = len(self.image_depths)
        for i, d in enumerate(self.image_depths):
            if depth < d:
                index = i
                break
        self.image_depths.insert(index, depth)
        self.image_slices_filenames.insert(index, filename)
        self.image_slices.insert(index, slice_image)
        
        if index > 0:
            # make sure the depth values are all unique
            for i in range(index, len(self.image_slices)):
                if self.image_depths[i] == self.image_depths[i-1]:
                    self.image_depths[i] += 1
                    
        return index
    
    def delete_slice(self, slice_index):
        """Deletes the slice at the specified index."""
        if slice_index < 0 or slice_index >= len(self.image_slices):
            return False
        self.image_slices.pop(slice_index)
        self.image_depths.pop(slice_index)
        self.image_slices_filenames.pop(slice_index)
        self.selected_slice = None
        self.slice_pixel = None
        self.slice_pixel_depth = None
        self.slice_mask = None
        
        # XXX - decide whether to delete the corresponding files
        return True
    
    def reset_image_slices(self):
        self.image_slices = []
        self.image_depths = []
        self.image_slices_filenames = []
        self.selected_slice = None
        self.selected_inpainting = None
        self.slice_pixel = None
        self.slice_mask = None
        self.slice_pixel_depth = None
        
    def balance_slices_depths(self):
        """Equally distribute the depths of the image slices."""
        self.image_depths[0] = 0
        self.image_depths[1:] = [int(i * 255 / (self.num_slices - 1)) for i in range(1, self.num_slices)]

    def depth_slice_from_pixel(self, pixel_x, pixel_y):
        depth = -1  # for log below
        if self.depthMapData is not None and self.imgThresholds is not None:
            depth = int(self.depthMapData[pixel_y, pixel_x])
            # find the depth that is bracketed by imgThresholds
            for i, threshold in enumerate(self.imgThresholds):
                if depth <= threshold:
                    threshold_min = int(self.imgThresholds[i-1])
                    threshold_max = int(threshold)
                    break
            mask = mask_from_depth(
                self.depthMapData, threshold_min, threshold_max)

        return mask, depth

    def set_img_data(self, img_data):
        self.imgData = img_data.convert('RGB')
        self.depthMapData = None
        self.selected_slice = None
        self.selected_inpainting = None
        self.image_slices = []
        self.slice_pixel = None

        self.create_tints()

    def serve_model_file(self):
        """Serves the gltf model file."""
        model_path = Path(self.SRV_DIR) / Path(self.filename) / self.MODEL_FILE
        unique_id = int(time.time())
        return f'/{str(model_path)}?v={unique_id}'

    def serve_slice_image(self, slice_index):
        """Serves the image slice with the specified index."""
        assert slice_index >= 0 and slice_index < len(
            self.image_slices_filenames)
        image_path = self.image_slices_filenames[slice_index]
        image_path = Path(self.SRV_DIR) / image_path
        unique_id = int(time.time())
        return f'/{str(image_path)}?v={unique_id}'
    
    def slice_image_composed(self, slice_index, grayscale=True):
        """Composes the slice image over the main image."""
        assert slice_index >= 0 and slice_index < len(
            self.image_slices_filenames)
        slice_image = self.image_slices[slice_index]
        if not isinstance(slice_image, Image.Image):
            slice_image = Image.fromarray(slice_image)
        if self.grayscale_tinted is None:
            self.create_tints()
        full_image = Image.composite(
            slice_image, self.grayscale_tinted if grayscale else self.imgData, slice_image.getchannel('A'))
        return full_image
    
    def serve_slice_image_composed(self, slice_index):
        """Serves the slice image composed over the gray main image."""
        full_image = self.slice_image_composed(slice_index)
        return self.serve_main_image(full_image)
    
    def serve_input_image(self):
        """Serves the input image from the state directory."""
        filename = Path(self.filename) / self.IMAGE_FILE
        if not filename.exists():
            if not Path(self.filename).exists():
                Path(self.filename).mkdir()
            self.imgData.save(filename, compress_level=1)
        filename = Path(self.SRV_DIR) / filename
        unique_id = int(time.time())
        return f'/{str(filename)}?v={unique_id}'

    def serve_main_image(self, image):
        """Serves the image using a temporary directory."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        output_dir = Path(self.filename)
        if not output_dir.exists():
            output_dir.mkdir()
        save_path = Path(self.filename) / self.MAIN_IMAGE
        image.save(save_path)
        image_path = Path(self.SRV_DIR) / save_path
        unique_id = int(time.time())
        return f'/{str(image_path)}?v={unique_id}'
    
    def workflow_path(self):
        """Returns the workflow path."""
        return Path(self.filename) / self.WORKFLOW

    def _make_filename(self, slice_index, suffix):
        assert slice_index >= 0 and slice_index < len(
            self.image_slices_filenames)
        file_path = Path(self.image_slices_filenames[slice_index])
        return file_path.parent / f"{file_path.stem}_{suffix}.png"

    def depth_filename(self, slice_index):
        """Returns the depth filename for the specified slice index."""
        return self._make_filename(slice_index, 'depth')

    def mask_filename(self, slice_index):
        """Returns the mask filename for the specified slice index."""
        return self._make_filename(slice_index, 'mask')
    
    def upscaled_filename(self, slice_index):
        """Returns the upscaled filename for the specified slice index."""
        return self._make_filename(slice_index, 'upscaled')

    def save_image_mask(self, slice_index, mask):
        """Saves the PIL image mask to a png file for the specified slice index."""
        mask_path = self.mask_filename(slice_index)
        mask.save(str(mask_path))

        return str(mask_path)

    def _read_image_slice(self, slice_index):
        img = cv2.imread(
            str(self.image_slices_filenames[slice_index]), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img

    def read_image_slices(self):
        self.image_slices = [None] * len(self.image_slices_filenames)
        for i in range(len(self.image_slices_filenames)):
            self.image_slices[i] = self._read_image_slice(i)
        print(f"Loaded {len(self.image_slices)} image slices")
        assert len(self.image_slices) == len(self.image_slices_filenames)

    def can_undo(self, slice_index, forward=False):
        """
        Check if it is possible to undo the specified slice index.

        Args:
            slice_index (int): The index of the slice to check.
            forward (bool, optional): If True, check for the next version of the slice. 
                                      If False, check for the previous version. Defaults to False.

        Returns:
            bool: True if the specified slice version exists, False otherwise.
        """
        assert slice_index >= 0 and slice_index < len(
            self.image_slices_filenames)
        if forward:
            filename = filename_add_version(
                self.image_slices_filenames[slice_index])
        else:
            filename = filename_previous_version(
                self.image_slices_filenames[slice_index])

        if filename is None:
            return False

        return Path(filename).exists()

    def undo(self, slice_index, forward=False):
        """
        Undo the specified slice index.

        Args:
            slice_index (int): The index of the slice to undo.
            forward (bool, optional): If True, undo the next version of the slice. 
                          If False, undo the previous version. Defaults to False.

        Returns:
            bool: True if the undo operation is successful, False otherwise.
        """
        assert slice_index >= 0 and slice_index < len(
            self.image_slices_filenames)

        if not self.can_undo(slice_index, forward):
            return False

        if forward:
            filename = filename_add_version(
                self.image_slices_filenames[slice_index])
        else:
            filename = filename_previous_version(
                self.image_slices_filenames[slice_index])

        if filename is None:
            return False

        self.image_slices_filenames[slice_index] = filename
        self.image_slices[slice_index] = self._read_image_slice(slice_index)
        return True

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
            
    def upscale_slices(self):
        if self.upscaler is None:
            self.upscaler = Upscaler()
            
        for i, slice_image in enumerate(self.image_slices):
            filename = self.upscaled_filename(i)
            if not Path(filename).exists():
                print(f"Upscaling image slice: {filename}")
                upscaled_image = self.upscaler.upscale_image_tiled(slice_image, tile_size=512, overlap=64)
                upscaled_image.save(filename)
                print(f"Saved upscaled image slice: {filename}")

    def to_file(self, file_path, save_image_slices=True, save_depth_map=True, save_input_image=True):
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
        if save_input_image:
            img_file = file_path / AppState.IMAGE_FILE
            self.imgData.save(str(img_file))

        # save the depth map
        if self.depthMapData is not None and save_depth_map:
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
        self.read_image_slices()

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
            'image_depths': self.image_depths,
            'image_slices_filenames': self.image_slices_filenames,
            'positive_prompt': self.positive_prompt,
            'negative_prompt': self.negative_prompt,
        }
        if self.depth_model_name is not None:
            data['depth_model_name'] = self.depth_model_name
        if self.server_address is not None:
            data['server_address'] = self.server_address
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
        state.image_depths = data['image_depths'] if 'image_depths' in data else state.imgThresholds[1:]
        state.image_slices_filenames = data['image_slices_filenames']
        
        state.depth_model_name = data['depth_model_name'] if 'depth_model_name' in data else None
        
        state.positive_prompt = data['positive_prompt'] if 'positive_prompt' in data else ''
        state.negative_prompt = data['negative_prompt'] if 'negative_prompt' in data else ''
        
        state.server_address = data['server_address'] if 'server_address' in data else None
        
        # check that all filenames start with the filename as prefix
        state.check_pathnames()
            
        return state
    
    def check_pathnames(self):
        """Check that all pathnames are valid."""
        
        cwd = Path().cwd()
        root = Path(self.filename).resolve()
        assert str(root).startswith(str(cwd / 'appstate-')), f"Invalid filename: {self.filename}"
        
        for filename in self.image_slices_filenames:
            filename = Path(filename).resolve()
            assert str(filename).startswith(str(root)), f"Invalid filename: {filename}"
