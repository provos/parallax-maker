#!/usr/bin/env python
# (c) 2024 Niels Provos
#

import argparse

import numba as nb
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid
import cv2
import numpy as np
from PIL import Image

from utils import torch_get_device, find_square_bounding_box, feather_mask, timeit

# Possible pretrained models:
# pretrained_model = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
# pretrained_model = "runwayml/stable-diffusion-v1-5"
# pretrained_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


class PipelineSpec:
    def __init__(self,
                 pretrained_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                 variant="fp16",
                 dimension=1024):
        self.pretrained_model = pretrained_model
        self.variant = variant
        self.dimension = dimension
        self.pipeline = None

    def __eq__(self, other):
        if not isinstance(other, PipelineSpec):
            return False
        return self.pretrained_model == other.pretrained_model and \
            self.variant == other.variant and \
            self.dimension == other.dimension

    def get_dimension(self):
        return self.dimension

    def create_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        kwargs = {}
        if self.variant:
            kwargs['variant'] = self.variant

        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            self.pretrained_model, torch_dtype=torch.float16, **kwargs
        )

        device = torch_get_device()
        self.pipeline.to(device)

        return self.pipeline


def pipelinespec_from_model(model):
    """
    Create a PipelineSpec object based on the given model.

    Args:
        model (str): The name of the model.

    Returns:
        PipelineSpec: The PipelineSpec object with the specified model, variant, and dimension.
    """
    dimension = 512
    variant = ''
    if model == "kandinsky-community/kandinsky-2-2-decoder-inpaint":
        dimension = 768
    elif model == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1":
        dimension = 1024
        variant = "fp16"
    return PipelineSpec(pretrained_model=model, variant=variant, dimension=dimension)


def inpaint(pipelinespec, prompt, negative_prompt, init_image, mask_image,
            strength=0.7, guidance_scale=8, num_inference_steps=50, padding=50, blur_radius=50,
            crop=False, seed=-1):
    original_init_image = init_image.copy()

    if crop:
        bounding_box = find_square_bounding_box(mask_image, padding=padding)
        init_image = init_image.crop(bounding_box)
        mask_image = mask_image.crop(bounding_box)

    blurred_mask = feather_mask(mask_image, num_expand=blur_radius)

    # resize images to match the model input size
    pipeline = pipelinespec.pipeline
    dimension = pipelinespec.get_dimension()
    resize_init_image = init_image.convert(
        'RGB').resize((dimension, dimension))
    resize_mask_image = blurred_mask.resize((dimension, dimension))

    if seed == -1:
        seed = np.random.randint(0, 2**32)
    generator = torch.Generator(torch_get_device()).manual_seed(seed)
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt,
                     image=resize_init_image, mask_image=resize_mask_image, generator=generator,
                     strength=strength, guidance_scale=guidance_scale,
                     num_inference_steps=num_inference_steps).images[0]

    image = image.resize(init_image.size, resample=Image.BICUBIC)
    image = image.convert('RGBA')

    image.putalpha(blurred_mask)

    image = Image.alpha_composite(init_image, image)

    if crop:
        original_init_image.paste(image, bounding_box[:2])
        image = original_init_image

    return image


def load_image_from_file(image_path, mode='RGB'):
    image = Image.open(image_path)
    image = image.convert(mode)
    return image


@nb.jit(nopython=True, cache=True)
def find_nearest_alpha(alpha, near_a):
    """
    Finds the nearest alpha values in pixel indices for each pixel in the given alpha image.

    Args:
        alpha (numpy.ndarray): The alpha channel represented as a 2D numpy array.
        near_a (numpy.ndarray): The array to store the nearest alpha values.

    Returns:
        None: This function modifies the `nearest_alpha` array in-place.

    Description:
        Uses dynamic programming to find the nearest alpha values for each pixel in the alpha image.

        The `near_a` array is modified in-place to store the nearest alpha values for each pixel.
    """
    height, width = alpha.shape
    for i in range(height):
        for j in range(width):
            if alpha[i, j] == 255:
                near_a[i, j, 0, 0] = i
                near_a[i, j, 0, 1] = j
                near_a[i, j, 1, 0] = i
                near_a[i, j, 1, 1] = j
                near_a[i, j, 2, 0] = i
                near_a[i, j, 2, 1] = j
                near_a[i, j, 3, 0] = i
                near_a[i, j, 3, 1] = j
                
    for i in range(height):
        for j in range(width):
            if alpha[i, j] != 255:
                if i > 0:
                    near_a[i, j, 0] = near_a[i - 1, j, 0]
                if j > 0:
                    near_a[i, j, 1] = near_a[i, j - 1, 1]
            if alpha[height - i - 1, width - j - 1] != 255:
                if i > 0:
                    near_a[height - i - 1, width - j - 1, 2] = near_a[height - i, width - j - 1, 2]
                if j > 0:
                    near_a[height - i - 1, width - j - 1, 3] = near_a[height - i - 1, width - j, 3]


@nb.jit(nopython=True, parallel=True)
def patch_pixels(image, mask_image, nearest_alpha, patch_indices):
    """
    Fills in the missing pixels in the image using a patch-based inpainting algorithm.

    Args:
        image (ndarray): The input image with missing pixels.
        mask_image (ndarray): The mask image indicating the locations of missing pixels.
        nearest_alpha (ndarray): The nearest alpha values for each pixel in the image.
        patch_indices (ndarray): The indices of the pixels to be inpainted.

    Returns:
        None. The input image is modified in-place.

    """
    for idx in nb.prange(len(patch_indices)):
        i, j = patch_indices[idx]
        surrounding_pixels = np.zeros((4, 3))
        distances = np.zeros(4)
        count = 0

        for direction in range(4):
            test_y, test_x = nearest_alpha[i, j, direction]
            if test_y != -1 and test_x != -1:
                distance = max(abs(i - test_y), abs(j - test_x))
                surrounding_pixels[count] = image[test_y, test_x, :3]
                distances[count] = distance
                count += 1

        if count > 0:
            weights = 1 / (distances[:count] + 1)
            average_color = np.sum(
                surrounding_pixels[:count] * weights.reshape(-1, 1), axis=0) / np.sum(weights)
            image[i, j, :3] = average_color
            image[i, j, 3] = max(mask_image[i, j], image[i, j, 3])


@timeit
def patch_image(image, mask_image):
    """
    Finds the area of the image that has no alpha channel and is overlapped by the mask image.
    It patches those pixels by approximating the color by finding the average color of the nearest pixels that
    have alpha channel.

    Args:
        image (np.array): The image with alpha channel to patch. Modified in place.
        mask_image (np.array): The mask image with a single channel.

    Returns:
        np.array: The patched image with the same shape as the input image.

    Raises:
        ValueError: If the input image does not have an alpha channel.
    """
    if image.shape[2] != 4:
        raise ValueError("The input image must have an alpha channel.")

    height, width, _ = image.shape

    alpha = image[:, :, 3].copy()
    alpha = cv2.blur(alpha, (25, 25))

    # Create a boolean mask indicating pixels that need patching
    patch_mask = (alpha < 255) & (mask_image > 0)

    # Find the indices of pixels that need patching
    patch_indices = np.argwhere(patch_mask)

    # Initialize the nearest_alpha array with -1 (indicating no nearest alpha pixel)
    nearest_alpha = np.full((height, width, 4, 2), -1, dtype=int)

    find_nearest_alpha(alpha, nearest_alpha)

    patch_pixels(image, mask_image, nearest_alpha, patch_indices)

    return image


def main():
    parser = argparse.ArgumentParser(
        description='Inpain images based on a mask and prompt.')
    parser.add_argument('-i', '--image', type=str,
                        help='Path to the input image')
    parser.add_argument('-m', '--mask', type=str,
                        help='Path to the mask image')
    parser.add_argument('-o', '--output', type=str,
                        default='inpainted_image.png', help='Path to the output filename')
    parser.add_argument('-a', '--patch', action='store_true',
                        help='Patch the image using the fast inpainting algorithm.')
    parser.add_argument('-p', '--prompt', type=str,
                        default='a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k',
                        help='The positive prompt to use for inpainting')
    parser.add_argument('-n', '--negative-prompt', type=str, default='bad anatomy, deformed, ugly, disfigured',
                        help='The negative prompt to use for inpainting')
    args = parser.parse_args()

    init_image = load_image_from_file(args.image, mode='RGBA')
    mask_image = load_image_from_file(args.mask, mode='L')

    init_image = patch_image(np.array(init_image), np.array(mask_image))
    init_image = Image.fromarray(init_image)

    if not args.patch:
        pipeline = PipelineSpec(
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            variant='fp16',
            dimension=1024)

        pipeline.create_pipeline()

        new_image = inpaint(pipeline, args.prompt,
                            args.negative_prompt, init_image, mask_image, crop=True)
    else:
        new_image = init_image

    new_image.save(args.output, format='PNG')

    image = make_image_grid(
        [init_image, mask_image, new_image], rows=1, cols=3)

    image = np.array(image)

    cv2.imshow("Inpainting", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
