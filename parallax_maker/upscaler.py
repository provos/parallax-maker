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

from .utils import torch_get_device, premultiply_alpha_numpy, find_bounding_box
import argparse


class Upscaler:
    def __init__(self, model_name="swin2sr", external_model=None):
        assert model_name in ["swin2sr", "simple", "inpainting", "stabilityai"]
        self.model_name = model_name
        self.model = None
        self.image_processor = None
        self.external_model = external_model
        self.tile_size = 512
        self.scale_factor = 2

    def create_model(self):
        if self.model_name == "swin2sr":
            self.model, self.image_processor = self.load_swin2sr_model()
        elif self.model_name == "simple":
            self.model, self.image_processor = None, None
            self.tile_size = 512
        elif self.model_name == "inpainting":
            assert self.external_model is not None
            assert self.external_model.get_dimension() is not None
            self.model, self.image_processor = self.external_model, None
            self.tile_size = self.external_model.get_dimension() // 2
        elif self.model_name == "stabilityai":
            self.model, self.image_processor = self.external_model, None
            self.tile_size = 1024
            self.scale_factor = 4

    def __eq__(self, other):
        if not isinstance(other, Upscaler):
            return False
        return self.model_name == other.model_name

    def load_swin2sr_model(self):
        # Initialize the image processor and model
        image_processor = AutoImageProcessor.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64"
        )
        model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64"
        )
        model.to(torch_get_device())

        return model, image_processor

    def upscale_image_tiled(self, image, overlap=64, prompt=None, negative_prompt=None):
        """
        Upscales an image using a tiled approach.

        Args:
            image (PIL.Image or np.ndarray): The input image array.
            overlap (int, optional): The overlap between adjacent tiles. Defaults to 64.

        Returns:
            np.ndarray: The upscaled image array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # initializes parameters we need below
        if self.model is None:
            self.create_model()

        alpha = None
        bounding_box = None
        if image.shape[2] == 4:  # RGBA image
            alpha = image[:, :, 3]
            bounding_box = find_bounding_box(alpha, padding=0)

            # scale the size of the alpha channel using bicubic interpolation
            alpha = zoom(alpha, self.scale_factor, order=3)
            image = image[:, :, :3]  # Remove alpha channel

            # crop the image to the bounding box
            orig_image = image
            image = image[
                bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]
            ]

        # Ensure the overlap can be divided by the scale factor
        if overlap % self.scale_factor != 0:
            overlap = (overlap // self.scale_factor + 1) * self.scale_factor

        # Calculate the number of tiles
        height, width, _ = image.shape
        step_size = self.tile_size - overlap // self.scale_factor
        num_tiles_x = (width + step_size - 1) // step_size
        num_tiles_y = (height + step_size - 1) // step_size

        # Create a new array to store the upscaled result
        upscaled_height = height * self.scale_factor
        upscaled_width = width * self.scale_factor
        upscaled_image = np.zeros((upscaled_height, upscaled_width, 3), dtype=np.uint8)

        # Iterate over the tiles
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate the coordinates of the current tile
                left = x * step_size
                top = y * step_size
                right = min(left + self.tile_size, width)
                bottom = min(top + self.tile_size, height)

                # make sure we process a full tile
                if x > 0 and right - left < self.tile_size:
                    left = right - self.tile_size
                    assert left >= 0
                if y > 0 and bottom - top < self.tile_size:
                    top = bottom - self.tile_size
                    assert top >= 0

                print(
                    f"Processing tile ({y}, {x}) with coordinates ({left}, {top}, {right}, {bottom})"
                )

                # Extract the current tile from the image
                tile = image[top:bottom, left:right]
                # XXX - revisit whether we can keep this as a numpy array
                tile = Image.fromarray(tile)

                cur_width, cur_height = tile.size
                if cur_width % 64 != 0 or cur_height % 64 != 0:
                    new_width = (
                        cur_width + (64 - cur_width % 64)
                        if cur_width % 64 != 0
                        else cur_width
                    )
                    new_height = (
                        cur_height + (64 - cur_height % 64)
                        if cur_height % 64 != 0
                        else cur_height
                    )
                    tile = tile.resize((new_width, new_height))
                upscaled_tile = self.upscale_tile(tile, prompt, negative_prompt)
                if tile.size != (cur_width, cur_height):
                    upscaled_tile = upscaled_tile.resize(
                        (cur_width * self.scale_factor, cur_height * self.scale_factor)
                    )
                upscaled_tile = np.array(upscaled_tile)

                # Calculate the coordinates to paste the upscaled tile
                place_left = left * self.scale_factor
                place_top = top * self.scale_factor
                place_right = place_left + upscaled_tile.shape[1]
                place_bottom = place_top + upscaled_tile.shape[0]

                self.integrate_tile(
                    upscaled_tile,
                    upscaled_image,
                    place_left,
                    place_top,
                    place_right,
                    place_bottom,
                    x,
                    y,
                    overlap,
                )

        # Combine the upscaled image with the alpha channel if present
        if alpha is not None:
            image = zoom(orig_image, (self.scale_factor, self.scale_factor, 1), order=3)
            bounding_box = [coord * self.scale_factor for coord in bounding_box]
            image[
                bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]
            ] = upscaled_image
            upscaled_image = np.dstack((image, alpha))
            upscaled_image = premultiply_alpha_numpy(upscaled_image)
        else:
            upscaled_image = Image.fromarray(upscaled_image)

        return upscaled_image

    @staticmethod
    def integrate_tile(tile, image, left, top, right, bottom, tile_x, tile_y, overlap):
        height, width, _ = tile.shape

        # xxx - should be move before upscaling
        if overlap >= height or overlap >= width:
            return

        # Create an alpha channel for the tile
        alpha = np.ones((height, width), dtype=np.float32)
        if tile_x > 0 and tile_y > 0:
            alpha[:overlap, :overlap] = np.outer(
                np.linspace(0, 1, overlap), np.linspace(0, 1, overlap)
            )
        if tile_x > 0:
            new_alpha = np.tile(np.linspace(0, 1, overlap), (height, 1))
            alpha[:, :overlap] = np.minimum(alpha[:, :overlap], new_alpha)

        if tile_y > 0:
            new_alpha = np.tile(np.linspace(0, 1, overlap).reshape(-1, 1), (1, width))
            alpha[:overlap, :] = np.minimum(alpha[:overlap, :], new_alpha)

        # Reshape the alpha channel to match the tile shape
        alpha = alpha.reshape(height, width, 1)

        # Apply the tile with alpha blending to the image
        image[top:bottom, left:right] = (
            alpha * tile + (1 - alpha) * image[top:bottom, left:right]
        ).astype(np.uint8)

    def upscale_tile(self, tile, prompt=None, negative_prompt=None):
        if self.model_name == "swin2sr":
            return self._upscale_tile_swin2sr(tile)
        elif self.model_name == "simple":
            return self._upscale_tile_simple(tile)
        elif self.model_name == "inpainting":
            return self._upscale_tile_inpainting(tile, prompt, negative_prompt)
        elif self.model_name == "stabilityai":
            return self._upscale_tile_stabilityai(tile, prompt, negative_prompt)

    def _upscale_tile_stabilityai(self, tile, prompt, negative_prompt):
        upscaled_image = self.external_model.upscale_image(
            tile, prompt, negative_prompt=negative_prompt
        )
        width, height = tile.size
        upscaled_image = upscaled_image.resize(
            (width * self.scale_factor, height * self.scale_factor), Image.LANCZOS
        )
        return upscaled_image

    def _upscale_tile_inpainting(self, tile, prompt, negative_prompt):
        rescaled_tile = tile.resize((tile.size[0] * 2, tile.size[1] * 2), Image.LANCZOS)
        rescaled_tile = np.array(rescaled_tile)
        mask_image = np.ones(rescaled_tile.shape[:2], dtype=np.uint8)
        mask_image *= 255
        scale = 2 if len(prompt) or len(negative_prompt) else 0
        tile = self.external_model.inpaint(
            prompt,
            negative_prompt,
            rescaled_tile,
            mask_image,
            strength=0.25,
            guidance_scale=scale,
            num_inference_steps=75,
            padding=0,
            blur_radius=0,
        )
        tile = tile.convert("RGB")
        return tile

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
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        output = outputs.reconstruction.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)

        return Image.fromarray(output)


if __name__ == "__main__":
    from inpainting import InpaintingModel
    from stabilityai import StabilityAI

    parser = argparse.ArgumentParser(description="Image Upscaling")
    parser.add_argument(
        "-i", "--input", type=str, default="input.jpg", help="Input image path"
    )
    parser.add_argument(
        "-p", "--prompt", type=str, default="a sci-fi robot in a futuristic laboratory"
    )
    parser.add_argument("--api-key", type=str)
    args = parser.parse_args()

    if args.api_key:
        model_name = "stabilityai"
        model = StabilityAI(args.api_key)
    else:
        model_name = "inpainting"
        model = InpaintingModel()
        model.load_model()
    upscaler = Upscaler(model_name=model_name, external_model=model)
    image = Image.open(args.input)
    upscaled_image = upscaler.upscale_image_tiled(
        image,
        overlap=64,
        prompt=args.prompt,
    )
    upscaled_image.save("upscaled_image.png")
