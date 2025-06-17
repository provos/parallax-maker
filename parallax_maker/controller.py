# (c) 2024 Niels Provos

import shutil
import threading
import json
import random
import string
import time
from typing import List
from PIL import Image
from io import BytesIO
from enum import Enum
import numpy as np
import cv2
from pathlib import Path
from .utils import (
    filename_previous_version,
    filename_add_version,
    apply_color_tint,
    create_checkerboard,
    encode_string_with_nonce,
    decode_string_with_nonce,
)
from .segmentation import mask_from_depth
from .upscaler import Upscaler
from .stabilityai import StabilityAI
from .camera import Camera
from .slice import ImageSlice


class CompositeMode(Enum):
    GRAYSCALE = 1
    CHECKERBOARD = 2
    NONE = 3


class AppState:
    __slots__ = (
        "_lock",
        "filename",
        "num_slices",
        "imgData",
        "imgThresholds",
        "depthMapData",
        "depth_model_name",
        "inpainting_model_name",
        "server_address",
        "api_key",
        "dark_mode",
        "image_slices",
        "selected_slice",
        "pipeline_spec",
        "depth_estimation_model",
        "segmentation_model",
        "selected_inpainting",
        "result_tinted",
        "grayscale_tinted",
        "checkerboard",
        "slice_pixel",
        "slice_pixel_depth",
        "slice_mask",
        "upscaler",
        "clipboard_image",
        "use_checkerboard",
        "multi_point_mode",
        "points_selected",
        "_camera",
        "_mesh_displacement",
    )
    SRV_DIR = "tmp-images"
    STATE_FILE = "appstate.json"
    IMAGE_FILE = "input_image.png"
    DEPTH_MAP_FILE = "depth_map.png"
    MAIN_IMAGE = "main_image.bmp"
    MODEL_FILE = "model.gltf"
    WORKFLOW = "workflow.json"

    cache = {}

    def __init__(self):
        # prevent concurrent writes
        self._lock = threading.Lock()

        self.filename = None
        self.num_slices = 5
        self.imgData = None  # PIL image
        self.imgThresholds = None
        self.depthMapData = None  # numpy array

        self.depth_model_name = None
        self.inpainting_model_name = None

        self.server_address = None
        self.api_key = None

        self.dark_mode = False

        # no JSON serialization for items below
        self.image_slices: List[ImageSlice] = []
        self.selected_slice = None
        self.pipeline_spec = None  # PipelineSpec() for inpainting
        self.depth_estimation_model = (
            None  # DepthEstimationModel() for depth estimation
        )
        self.segmentation_model = None  # SegmentationModel() for segmentation
        self.selected_inpainting = None
        self.result_tinted = None
        self.grayscale_tinted = None
        self.checkerboard = None
        self.slice_pixel = None
        self.slice_pixel_depth = None
        self.slice_mask = None
        self.upscaler = None
        self.clipboard_image = None

        self.use_checkerboard = False
        self.multi_point_mode = False
        self.points_selected = []

        self._camera = Camera(100.0, 500.0, 100.0)
        self._camera.camera_position = np.array([0.0, 0.0, -100.0], dtype=np.float32)

        self._mesh_displacement = 0.0

    @property
    def camera(self):
        return self._camera

    @property
    def mesh_displacement(self):
        return self._mesh_displacement

    @mesh_displacement.setter
    def mesh_displacement(self, value):
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("mesh_displacement must be a non-negative number")
        self._mesh_displacement = value

    def camera_matrix(self):
        image_width, image_height = self.imgData.size
        return self._camera.camera_matrix(image_width, image_height)

    def get_cards(self):
        image_width, image_height = self.imgData.size
        cards = []
        for i, image_slice in enumerate(self.image_slices):
            card = image_slice.create_card(image_height, image_width, self.camera)
            cards.append(card)
        return cards

    def create_tints(self):
        """Precomputes the tings for visualizing slices."""
        # Apply the main color tint to the original image
        self.result_tinted = apply_color_tint(self.imgData, (150, 255, 0), 0.3)

        # Convert the image to grayscale and back to RGB
        grayscale = self.imgData.convert("L").convert("RGB")
        self.grayscale_tinted = apply_color_tint(grayscale, (0, 0, 150), 0.1)

    def apply_mask(self, image, mask):
        # Prepare masks; make sure they are the right size and mode
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)
        mask = mask.convert("L")

        if image is not self.imgData:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")
            result_tinted = apply_color_tint(image, (150, 255, 0), 0.3)
            grayscale_tinted = apply_color_tint(
                image.convert("L").convert("RGB"), (0, 0, 150), 0.1
            )
        else:
            if self.result_tinted is None or self.grayscale_tinted is None:
                self.create_tints()
            result_tinted = self.result_tinted
            grayscale_tinted = self.grayscale_tinted

        # Combine the tinted and the grayscale image
        final_result = Image.composite(result_tinted, grayscale_tinted, mask)

        return final_result

    def change_slice_depth(self, slice_index, depth):
        """Changes the depth of the slice at the specified index."""
        assert slice_index >= 0 and slice_index < len(self.image_slices)

        if depth == self.image_slices[slice_index].depth:
            return slice_index

        # remove it from the lists
        image_slice = self.image_slices.pop(slice_index)
        image_slice.depth = depth

        return self.add_slice(image_slice)

    def add_slice(self, image_slice: ImageSlice):
        """Adds the image as a new slice at the provided depth."""
        if image_slice.filename is None:
            image_slice.filename = str(
                Path(self.filename) / f"image_slice_{len(self.image_slices)}.png"
            )
        # find the index where the depth should be inserted
        index = len(self.image_slices)
        for i, cur_slice in enumerate(self.image_slices):
            if image_slice.depth < cur_slice.depth:
                index = i
                break
        self.image_slices.insert(index, image_slice)

        if index > 0:
            # make sure the depth values are all unique
            for i in range(index, len(self.image_slices)):
                if self.image_slices[i].depth == self.image_slices[i - 1].depth:
                    self.image_slices[i].depth += 1

        return index

    def delete_slice(self, slice_index):
        """Deletes the slice at the specified index."""
        if slice_index < 0 or slice_index >= len(self.image_slices):
            return False
        self.image_slices.pop(slice_index)
        self.selected_slice = None
        self.slice_pixel = None
        self.slice_pixel_depth = None
        self.slice_mask = None

        return True

    def reset_image_slices(self):
        self.image_slices = []
        self.selected_slice = None
        self.selected_inpainting = None
        self.slice_pixel = None
        self.slice_mask = None
        self.slice_pixel_depth = None

    def balance_slices_depths(self):
        """Equally distribute the depths of the image slices."""
        depths = [
            int(i * 255 / (len(self.image_slices) - 1))
            for i in range(len(self.image_slices))
        ]
        for i in len(self.image_slices):
            self.image_slices[i].depth = depths[i]

    def depth_slice_from_pixel(self, pixel_x, pixel_y):
        depth = -1  # for log below
        if self.depthMapData is not None and self.imgThresholds is not None:
            depth = int(self.depthMapData[pixel_y, pixel_x])
            # find the depth that is bracketed by imgThresholds
            for i, threshold in enumerate(self.imgThresholds):
                if depth <= threshold:
                    threshold_min = int(self.imgThresholds[i - 1])
                    threshold_max = int(threshold)
                    break
            mask = mask_from_depth(self.depthMapData, threshold_min, threshold_max)

        return mask, depth

    def set_img_data(self, img_data):
        self.imgData = img_data.convert("RGB")
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
        return f"/{str(model_path)}?v={unique_id}"

    def serve_slice_image(self, slice_index):
        """Serves the image slice with the specified index."""
        assert slice_index >= 0 and slice_index < len(self.image_slices)
        image_path = self.checkerboard_filename(slice_index)
        if not image_path.exists():
            image = self.slice_image_composed(
                slice_index, mode=CompositeMode.CHECKERBOARD
            )
            image.save(image_path)
        image_path = Path(self.SRV_DIR) / image_path
        return f"/{str(image_path)}"

    def slice_image_composed(
        self, slice_index, mode: CompositeMode = CompositeMode.NONE
    ):
        """Composes the slice image over the main image."""
        assert slice_index >= 0 and slice_index < len(self.image_slices)
        slice_image = self.image_slices[slice_index].image
        if not isinstance(slice_image, Image.Image):
            slice_image = Image.fromarray(slice_image)

        composite = self.imgData
        if mode == CompositeMode.GRAYSCALE:
            if self.grayscale_tinted is None:
                self.create_tints()
            composite = self.grayscale_tinted
        elif mode == CompositeMode.CHECKERBOARD:
            if self.checkerboard is None:
                self.checkerboard = create_checkerboard(
                    slice_image.size[1], slice_image.size[0], 32
                )
            composite = Image.fromarray(self.checkerboard)

        full_image = Image.composite(
            slice_image, composite, slice_image.getchannel("A")
        )
        return full_image

    def serve_slice_image_composed(self, slice_index, mode: CompositeMode):
        """Serves the slice image composed over the gray main image."""
        full_image = self.slice_image_composed(slice_index, mode=mode)
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
        return f"/{str(filename)}?v={unique_id}"

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
        return f"/{str(image_path)}?v={unique_id}"

    def workflow_path(self):
        """Returns the workflow path."""
        return Path(self.filename) / self.WORKFLOW

    def _make_filename(self, slice_index, suffix):
        assert slice_index >= 0 and slice_index < len(self.image_slices)
        file_path = Path(self.image_slices[slice_index].filename)
        return file_path.parent / f"{file_path.stem}_{suffix}.png"

    def checkerboard_filename(self, slice_index):
        """Returns the checkerboard filename for the specified slice index."""
        return self._make_filename(slice_index, "checkerboard")

    def depth_filename(self, slice_index):
        """Returns the depth filename for the specified slice index."""
        return self._make_filename(slice_index, "depth")

    def mask_filename(self, slice_index):
        """Returns the mask filename for the specified slice index."""
        return self._make_filename(slice_index, "mask")

    def upscaled_filename(self, slice_index):
        """Returns the upscaled filename for the specified slice index."""
        return self._make_filename(slice_index, "upscaled")

    def save_image_mask(self, slice_index, mask):
        """Saves the PIL image mask to a png file for the specified slice index."""
        mask_path = self.mask_filename(slice_index)
        mask.save(str(mask_path))

        return str(mask_path)

    def _read_image_slices(self):
        for i, image_slice in enumerate(self.image_slices):
            image_slice.read_image()
        print(f"Loaded {len(self.image_slices)} image slices")

    def save_image_slices(self, file_path):
        """
        Save the image slices to the specified file path.

        Args:
            file_path (str): The file path to save the image slices.
        """
        file_path = Path(file_path)
        for i, slice_image in enumerate(self.image_slices):
            if slice_image.filename is None:
                slice_image.filename = str(file_path / f"image_slice_{i}.png")
            slice_image.save_image()

    def _create_upscaler(self):
        if self.pipeline_spec is None:
            self.upscaler = Upscaler()
            return
        if self.inpainting_model_name == "stabilityai":
            model_name = "stabilityai"
            model = StabilityAI(self.api_key)
        else:
            model_name = "inpainting"
            model = self.pipeline_spec
            self.pipeline_spec.load_model()
        self.upscaler = Upscaler(model_name=model_name, external_model=model)

    def upscale_image(self, image, prompt="", negative_prompt=""):
        if self.upscaler is None:
            self._create_upscaler()
        upscaled_image = self.upscaler.upscale_image_tiled(
            image, overlap=64, prompt=prompt, negative_prompt=negative_prompt
        )
        return upscaled_image

    def upscale_slices(self):
        for i, slice_image in enumerate(self.image_slices):
            filename = self.upscaled_filename(i)
            if not Path(filename).exists():
                prompt = slice_image.positive_prompt
                negative_prompt = slice_image.negative_prompt
                print(
                    f"Upscaling image slice: {filename} with '{prompt}'/'{negative_prompt}'"
                )
                upscaled_image = self.upscale_image(
                    slice_image.image, prompt=prompt, negative_prompt=negative_prompt
                )
                upscaled_image.save(filename)
                print(f"Saved upscaled image slice: {filename}")

    def to_file(
        self,
        file_path,
        save_image_slices=True,
        save_depth_map=True,
        save_input_image=True,
    ):
        """
        Save the current state to a file.

        Args:
            file_path (str): The file path to save the state.
            save_image_slices (bool, optional): Whether to save the image slices. Defaults to True.
        """
        with self._lock:
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
            temp_file = state_file.with_suffix(".tmp")
            backup_file = state_file.with_suffix(".bak")

            try:
                # Write to temporary file
                with open(temp_file, "w", encoding="utf-8") as file:
                    file.write(self.to_json())

                # Create a backup of the current state file if it exists
                if state_file.exists():
                    shutil.move(state_file, backup_file)

                # Move the temporary file to the state file
                shutil.move(temp_file, state_file)

                # Optionally, remove the backup file if everything went fine
                if backup_file.exists():
                    backup_file.unlink()

            except Exception as e:
                # Clean up temporary and backup files if any error occurs
                if temp_file.exists():
                    temp_file.unlink()
                if backup_file.exists() and not state_file.exists():
                    # Restore from backup if the original state file is missing
                    shutil.move(backup_file, state_file)
                raise e  # Re-raise the error after cleanup

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
        with open(state_file, "r", encoding="utf-8") as file:
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
        self._read_image_slices()

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
        return "appstate-" + "".join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )

    def to_json(self):
        """
        Convert the state to a JSON string.

        Returns:
            str: The JSON string representation of the state.
        """

        image_depths = [image_slice.depth for image_slice in self.image_slices]
        image_slices_filenames = [
            image_slice.filename for image_slice in self.image_slices
        ]
        positive_prompts = [
            image_slice.positive_prompt for image_slice in self.image_slices
        ]
        negative_prompts = [
            image_slice.negative_prompt for image_slice in self.image_slices
        ]

        data = {
            "filename": self.filename,
            "num_slices": self.num_slices,
            "imgThresholds": self.imgThresholds,
            "image_depths": image_depths,
            "image_slices_filenames": image_slices_filenames,
            "positive_prompts": positive_prompts,
            "negative_prompts": negative_prompts,
            "dark_mode": self.dark_mode,
            "mesh_displacement": self._mesh_displacement,
        }

        # merge the camera data
        camera_dict = self._camera.to_json()
        data.update(camera_dict)

        if self.depth_model_name is not None:
            data["depth_model_name"] = self.depth_model_name
        if self.inpainting_model_name is not None:
            data["inpainting_model_name"] = self.inpainting_model_name
        if self.server_address is not None:
            data["server_address"] = self.server_address
        if self.api_key is not None:
            data["api_key"] = encode_string_with_nonce(self.api_key, self.filename)
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

        state.filename = data["filename"]
        state.num_slices = data["num_slices"] if "num_slices" in data else 5
        state.imgThresholds = data["imgThresholds"]

        image_depths = (
            data["image_depths"] if "image_depths" in data else state.imgThresholds[1:]
        )
        image_slices_filenames = data["image_slices_filenames"]

        state.depth_model_name = (
            data["depth_model_name"] if "depth_model_name" in data else None
        )
        state.inpainting_model_name = (
            data["inpainting_model_name"] if "inpainting_model_name" in data else None
        )

        empty = [""] * len(image_slices_filenames)
        positive_prompts = (
            data["positive_prompts"] if "positive_prompts" in data else empty
        )
        negative_prompts = (
            data["negative_prompts"] if "negative_prompts" in data else empty
        )

        # check dats structures have consistent lengths
        assert len(image_slices_filenames) == len(image_depths)
        assert len(image_slices_filenames) == len(positive_prompts)
        assert len(image_slices_filenames) == len(negative_prompts)

        state.image_slices = [
            ImageSlice(
                depth=depth,
                filename=filename,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
            )
            for depth, filename, positive_prompt, negative_prompt in zip(
                image_depths, image_slices_filenames, positive_prompts, negative_prompts
            )
        ]

        state.server_address = (
            data["server_address"] if "server_address" in data else None
        )
        state.api_key = (
            decode_string_with_nonce(data["api_key"], state.filename)
            if "api_key" in data
            else None
        )

        state.dark_mode = data["dark_mode"] if "dark_mode" in data else False

        if "mesh_displacement" in data:
            state.mesh_displacement = data["mesh_displacement"]

        state._camera = Camera.from_json(data)

        # check that all filenames start with the filename as prefix
        state.check_pathnames()

        return state

    def check_pathnames(self):
        """Check that all pathnames are valid."""

        cwd = Path().cwd()
        root = Path(self.filename).resolve()
        assert str(root).startswith(
            str(cwd / "appstate-")
        ), f"Invalid filename: {self.filename}"

        for image_slice in self.image_slices:
            filename = Path(image_slice.filename).resolve()
            assert str(filename).startswith(str(root)), f"Invalid filename: {filename}"
