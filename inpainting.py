#!/usr/bin/env python
# (c) 2024 Niels Provos
#

import argparse
import base64
from io import BytesIO
import numba as nb
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid
import cv2
import numpy as np
from PIL import Image

from utils import torch_get_device, find_square_bounding_box, feather_mask, timeit
from automatic1111 import create_img2img_payload, make_img2img_request
from comfyui import inpainting_comfyui

# Possible pretrained models:
# pretrained_model = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
# pretrained_model = "runwayml/stable-diffusion-v1-5"
# pretrained_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


class InpaintingModel:
    MODELS = [
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        "runwayml/stable-diffusion-v1-5",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    ]

    def __init__(self,
                 model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                 **kwargs):
        """
        Initializes the Inpainting class.

        Args:
            model (str): The name of the model to be used for inpainting.
                Defaults to "diffusers/stable-diffusion-xl-1.0-inpainting-0.1". Can also
                be "automatic1111" or "comfyui" to use an external API server.
            **kwargs: Additional keyword arguments to be passed to the class.
                At the moment, automatic1111 requires a "server_address" argument and
                ComfyUI requires a "server_address" and "workflow_path" argument.

        Attributes:
            model (str): The name of the model used for inpainting.
            kwargs (dict): Additional keyword arguments passed to the class.
        """
        self.model = model
        self.variant = ''
        self.dimension = 512
        self.kwargs = kwargs
        self.pipeline = None
        self.server_address = None
        self.workflow_path = None

    def __eq__(self, other):
        if not isinstance(other, InpaintingModel):
            return False
        return self.model == other.model and \
            self.variant == other.variant and \
            self.dimension == other.dimension

    def load_diffusers_model(self):
        self.dimension = 512
        self.variant = ''
        if self.model == "kandinsky-community/kandinsky-2-2-decoder-inpaint":
            self.dimension = 768
        elif self.model == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1":
            self.dimension = 1024
            self.variant = "fp16"

        kwargs = {}
        if self.variant:
            kwargs['variant'] = self.variant

        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            self.model, torch_dtype=torch.float16, **kwargs
        )

        device = torch_get_device()
        self.pipeline.to(device)

        return self.pipeline

    def load_external_model(self):
        assert 'server_address' in self.kwargs
        self.dimension = 1024
        self.pipeline = self.kwargs['server_address']
        self.server_address = self.kwargs['server_address']
        if self.model == "comfyui":
            assert 'workflow_path' in self.kwargs
            self.workflow_path = self.kwargs['workflow_path']
        return self.pipeline

    def get_dimension(self):
        return self.dimension

    def load_model(self):
        if self.pipeline is not None:
            return self.pipeline

        # diffusers models
        if self.model in self.MODELS:
            return self.load_diffusers_model()

        if self.model in ["automatic1111", "comfyui"]:
            return self.load_external_model()

        return None

    def inpaint(self, prompt, negative_prompt, init_image, mask_image,
                strength=0.7, guidance_scale=8, num_inference_steps=50, padding=50, blur_radius=50,
                crop=False, seed=-1):
        original_init_image = init_image.copy()

        if crop:
            bounding_box = find_square_bounding_box(
                mask_image, padding=padding)
            init_image = init_image.crop(bounding_box)
            mask_image = mask_image.crop(bounding_box)

        blurred_mask = feather_mask(mask_image, num_expand=blur_radius)

        # resize images to match the model input size
        dimension = self.get_dimension()
        resize_init_image = init_image.convert(
            'RGB').resize((dimension, dimension))
        resize_mask_image = blurred_mask.resize((dimension, dimension))

        if self.model in self.MODELS:
            image = self.inpaint_diffusers(
                resize_init_image, resize_mask_image,
                prompt, negative_prompt,
                strength, guidance_scale, num_inference_steps, seed)
        elif self.model == "automatic1111":
            image = self.inpaint_automatic1111(
                resize_init_image, resize_mask_image,
                prompt, negative_prompt,
                strength, guidance_scale, num_inference_steps, seed)
        elif self.model == "comfyui":
            image = inpainting_comfyui(
                self.server_address,
                self.workflow_path,
                resize_init_image, resize_mask_image,
                prompt, negative_prompt,
                strength=strength, cfg_scale=guidance_scale, steps=num_inference_steps,
                seed=seed)
        else:
            raise ValueError(f"Model {self.model} is not supported.")

        image = image.resize(init_image.size, resample=Image.BICUBIC)
        image = image.convert('RGBA')

        image.putalpha(blurred_mask)

        image = Image.alpha_composite(init_image, image)

        if crop:
            original_init_image.paste(image, bounding_box[:2])
            image = original_init_image

        return image

    def inpaint_automatic1111(self,
                              resize_init_image, resize_mask_image,
                              prompt, negative_prompt,
                              strength, guidance_scale, num_inference_steps, seed):
        server_address = self.kwargs['server_address']
        payload = create_img2img_payload(
            resize_init_image, prompt, negative_prompt,
            mask_image=resize_mask_image,
            strength=strength,
            steps=num_inference_steps,
            cfg_scale=guidance_scale
        )
        images = make_img2img_request(server_address, payload)
        if len(images) != 1:
            raise ValueError("Expected one image from the server.")
        return Image.open(BytesIO(base64.b64decode(images[0])))

    def inpaint_diffusers(self,
                          resize_init_image, resize_mask_image,
                          prompt, negative_prompt,
                          strength, guidance_scale, num_inference_steps, seed):
        if seed == -1:
            seed = np.random.randint(0, 2**32)
        generator = torch.Generator(torch_get_device()).manual_seed(seed)
        pipeline = self.pipeline
        image = pipeline(prompt=prompt, negative_prompt=negative_prompt,
                         image=resize_init_image, mask_image=resize_mask_image, generator=generator,
                         strength=strength, guidance_scale=guidance_scale,
                         num_inference_steps=num_inference_steps).images[0]

        return image


def prefetch_models(fetch_all_models=False):
    """
    Prefetch all models to avoid loading them multiple times.

    Args:
        fetch_all_models (bool): Whether to fetch all models.

    Returns:
        None
    """
    if fetch_all_models:
        for model in InpaintingModel.MODELS:
            InpaintingModel(model).load_model()
    else:
        InpaintingModel().load_model()


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
                    near_a[height - i - 1, width - j - 1,
                           2] = near_a[height - i, width - j - 1, 2]
                if j > 0:
                    near_a[height - i - 1, width - j - 1,
                           3] = near_a[height - i - 1, width - j, 3]


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
            # alpha blend the patched color with the original color
            alpha = image[i, j, 3]
            orig_color = image[i, j, :3]
            image[i, j, :3] = alpha / 255.0 * orig_color + (255.0 - alpha)/255.0 * average_color
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
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

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
    parser.add_argument('-s', '--strength', type=float, default=0.7,
                        help='The strength of diffusion applied to the image')
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
        #pipeline = InpaintingModel(
        #    'diffusers/stable-diffusion-xl-1.0-inpainting-0.1')
        # pipeline = InpaintingModel(
        #    'automatic1111', server_address='localhost:7860')
        pipeline = InpaintingModel(
            'comfyui',
            workflow_path='workflow.json',
            server_address='localhost:8188')

        pipeline.load_model()

        new_image = pipeline.inpaint(
            args.prompt, args.negative_prompt, init_image, mask_image,
            strength=args.strength, crop=True)
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
