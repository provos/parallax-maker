#!/usr/bin/env python
# (c) 2024 Niels Provos
#
"""
Upscale Image Tiling

This module provides functionality to upscale an image using a pre-trained deep learning model,
specifically Swin2SR for Super Resolution. The upscaling process breaks the input image into smaller
tiles.  Each tile is individually upscaled using the model, and then these upscaled tiles are reassembled
into the final image. The tile-based approach helps manage memory usage and can handle larger
images by processing small parts one at a time.

For better integration of the tiles into the resultant upscaled image, tiles are overlapped and
blended to ensure smoother transitions between tiles.
"""

from PIL import Image
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
import numpy as np
from scipy.ndimage import zoom

from utils import torch_get_device, premultiply_alpha_numpy


class Upscaler:
    def __init__(self, model_name="swin2sr"):
        assert model_name in ["swin2sr", "simple"]
        self.model_name = model_name
        self.model = None
        self.image_processor = None

    def create_model(self):
        if self.model_name == "swin2sr":
            self.model, self.image_processor = self.load_swin2sr_model()
        elif self.model_name == "simple":
            self.model, self.image_processor = None, None

    def __eq__(self, other):
        if not isinstance(other, Upscaler):
            return False
        return self.model_name == other.model_name

    def load_swin2sr_model(self):
        # Initialize the image processor and model
        image_processor = AutoImageProcessor.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64")
        model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64")
        model.to(torch_get_device())

        return model, image_processor

    def upscale_image_tiled(self, image, tile_size=512, overlap=64):
        """
        Upscales an image using a tiled approach.

        Args:
            image (np.ndarray): The input image array.
            tile_size (int, optional): The size of each tile. Defaults to 512.
            overlap (int, optional): The overlap between adjacent tiles. Defaults to 64.

        Returns:
            np.ndarray: The upscaled image array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        alpha = None
        if image.shape[2] == 4:  # RGBA image
            alpha = image[:, :, 3]
            # double the size of the alpha channel using bicubic interpolation
            alpha = zoom(alpha, 2, order=3)
            image = image[:, :, :3]  # Remove alpha channel

        if self.model is None:
            self.create_model()

        # Ensure the overlap is an even number
        if overlap % 2 != 0:
            overlap += 1

        # Calculate the number of tiles
        height, width, _ = image.shape
        step_size = tile_size - overlap // 2
        num_tiles_x = (width + step_size - 1) // step_size
        num_tiles_y = (height + step_size - 1) // step_size

        # Create a new array to store the upscaled result
        upscaled_height = height * 2
        upscaled_width = width * 2
        upscaled_image = np.zeros(
            (upscaled_height, upscaled_width, 3), dtype=np.uint8)

        # Iterate over the tiles
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate the coordinates of the current tile
                left = x * step_size
                top = y * step_size
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)

                print(
                    f"Processing tile ({y}, {x}) with coordinates ({left}, {top}, {right}, {bottom})")

                # Extract the current tile from the image
                tile = image[top:bottom, left:right]
                tile = Image.fromarray(tile) # XXX - revisit whether we can keep this as a numpy array
                upscaled_tile = self.upscale_tile(tile)
                upscaled_tile = np.array(upscaled_tile)

                # Calculate the coordinates to paste the upscaled tile
                place_left = x * step_size * 2
                place_top = y * step_size * 2
                place_right = place_left + upscaled_tile.shape[1]
                place_bottom = place_top + upscaled_tile.shape[0]

                self.integrate_tile(upscaled_tile, upscaled_image, place_left,
                                    place_top, place_right, place_bottom, x, y, overlap)

        # Combine the upscaled image with the alpha channel if present
        if alpha is not None:
            upscaled_image = np.dstack((upscaled_image, alpha))
            upscaled_image = premultiply_alpha_numpy(upscaled_image)

        return upscaled_image

    @staticmethod
    def integrate_tile(tile, image, left, top, right, bottom, tile_x, tile_y, overlap):
        height, width, _ = tile.shape

        # Create an alpha channel for the tile
        alpha = np.ones((height, width), dtype=np.float32)
        if tile_x > 0 and tile_y > 0:
            alpha[:overlap, :overlap] = np.outer(
                np.linspace(0, 1, overlap), np.linspace(0, 1, overlap))
        elif tile_x > 0:
            alpha[:, :overlap] = np.linspace(0, 1, overlap)
        elif tile_y > 0:
            alpha[:overlap, :] = np.linspace(0, 1, overlap).reshape(-1, 1)

        # Reshape the alpha channel to match the tile shape
        alpha = alpha.reshape(height, width, 1)

        # Apply the tile with alpha blending to the image
        image[top:bottom, left:right] = (
            alpha * tile + (1 - alpha) * image[top:bottom, left:right]).astype(np.uint8)

    def upscale_tile(self, tile):
        if self.model_name == "swin2sr":
            return self._upscale_tile_swin2sr(tile)
        elif self.model_name == "simple":
            return self._upscale_tile_simple(tile)
        
    def _upscale_tile_simple(self, tile):
        """
        Upscales a tile using a simple bicubic interpolation.

        Args:
            tile: The input tile to be upscaled.

        Returns:
            An Image object representing the upscaled tile.
        """
        return tile.resize((tile.width * 2, tile.height * 2), Image.BICUBIC)

    def _upscale_tile_swin2sr(self, tile):
        """
        Upscales a tile using a given model and image processor.

        Args:
            model: The model used for upscaling the tile.
            image_processor: The image processor used to preprocess the tile.
            tile: The input tile to be upscaled.

        Returns:
            An Image object representing the upscaled tile.
        """
        inputs = self.image_processor(tile, return_tensors="pt")
        inputs = {name: tensor.to(self.model.device)
                  for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        output = outputs.reconstruction.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)

        return Image.fromarray(output)


if __name__ == "__main__":
    upscaler = Upscaler(model_name="simple")
    image = Image.open("appstate-feBWVeXR/image_slice_1_v2.png")
    upscaled_image = upscaler.upscale_image_tiled(image, tile_size=512, overlap=64)
    upscaled_image.save("upscaled_image.png")
