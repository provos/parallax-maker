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

from utils import torch_get_device


class Upscaler:
    def __init__(self, model_name="swin2sr"):
        assert model_name in ["swin2sr"]
        self.model_name = model_name
        self.model = None
        self.image_processor = None

    def create_model(self):
        if self.model_name == "swin2sr":
            self.model, self.image_processor = self.load_swin2sr_model()

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
            image (PIL.Image.Image): The input image file.
            tile_size (int, optional): The size of each tile. Defaults to 512.
            overlap (int, optional): The overlap between adjacent tiles. Defaults to 64.

        Returns:
            PIL.Image.Image: The upscaled image.
        """
        # Convert the image
        alpha = None
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            # double the size of the alpha image
            double_size = (image.size[0] * 2, image.size[1] * 2) 
            alpha = alpha.resize(double_size, Image.BICUBIC)
            
        image = image.convert("RGB")

        if self.model is None:
            self.create_model()

        # Calculate the number of tiles
        width, height = image.size
        step_size = tile_size - overlap
        num_tiles_x = (width + step_size - 1) // step_size
        num_tiles_y = (height + step_size - 1) // step_size

        # Create a new image to store the upscaled result
        upscaled_width = width * 2
        upscaled_height = height * 2
        upscaled_image = Image.new("RGB", (upscaled_width, upscaled_height))

        # Iterate over the tiles
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate the coordinates of the current tile
                left = x * step_size
                top = y * step_size
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)

                print(
                    f"Processing tile ({y}, {x} with coordinates ({left}, {top}, {right}, {bottom})")

                # Extract the current tile from the image
                tile = image.crop((left, top, right, bottom))
                upscaled_tile = self.upscale_tile(tile)

                # Calculate the coordinates to paste the upscaled tile
                place_left = x * step_size * 2
                place_top = y * step_size * 2
                place_right = place_left + upscaled_tile.width
                place_bottom = place_top + upscaled_tile.height

                self.integrate_tile(upscaled_tile, upscaled_image, place_left,
                                    place_top, place_right, place_bottom, x, y, overlap)

        # Save the upscaled image
        if alpha:
            upscaled_image.putalpha(alpha)
        
        return upscaled_image

    def upscale_tile(self, tile):
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

    @staticmethod
    def integrate_tile(tile, image, left, top, right, bottom, tile_x, tile_y, overlap):
        if tile_x > 0:
            left_strip = tile.crop((0, 0, overlap, tile.height))
            blended = Image.blend(left_strip, image.crop(
                (left, top, left + overlap, bottom)), 0.5)
            image.paste(blended, (left, top, left + overlap, bottom))
            if overlap > tile.width:
                # Skip the tile if the overlap is greater than the tile width
                return
            tile = tile.crop((overlap, 0, tile.width, tile.height))
            left += overlap
        if tile_y > 0:
            top_strip = tile.crop((0, 0, tile.width, overlap))
            blended = Image.blend(top_strip, image.crop(
                (left, top, right, top + overlap)), 0.5)
            image.paste(blended, (left, top, right, top + overlap))
            if overlap > tile.height:
                # Skip the tile if the overlap is greater than the tile height
                return
            tile = tile.crop((0, overlap, tile.width, tile.height))
            top += overlap
        # Paste the upscaled tile onto the upscaled image
        image.paste(tile, (left, top, right, bottom))


if __name__ == "__main__":
    upscaler = Upscaler()
    image = Image.open("appstate-feBWVeXR/image_slice_1_v2.png")
    upscaled_image = upscaler.upscale_image_tiled(image, tile_size=512, overlap=64)
    upscaled_image.save("upscaled_image.png")
