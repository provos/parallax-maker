#!/usr/bin/env python
# (c) 2024 Niels Provos
#

import argparse
import base64
from io import BytesIO
import numba as nb
import torch
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image
from diffusers.utils import make_image_grid
import cv2
import numpy as np
from PIL import Image, ImageFilter

from .utils import torch_get_device, find_square_bounding_box, feather_mask, timeit
from .automatic1111 import create_img2img_payload, make_img2img_request
from .comfyui import inpainting_comfyui
from .stabilityai import StabilityAI


class InpaintingModel:
    # local models running on the local GPU
    MODELS = [
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        "runwayml/stable-diffusion-v1-5",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "stabilityai/stable-diffusion-3-medium-diffusers",
    ]

    # reached via API end points
    EXTERNAL_MODELS = ["automatic1111", "comfyui", "stabilityai"]

    def __init__(
        self, model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", **kwargs
    ):
        """
        Initializes the Inpainting class.

        Args:
            model (str): The name of the model to be used for inpainting.
                Defaults to "diffusers/stable-diffusion-xl-1.0-inpainting-0.1". Can also
                be "automatic1111" or "comfyui" to use an external API server.
            **kwargs: Additional keyword arguments to be passed to the class.
                At the moment, automatic1111 requires a "server_address" argument and
                ComfyUI requires a "server_address" and "workflow_path" argument.
                StabilityAI requires an "api_key" argument.

        Attributes:
            model (str): The name of the model used for inpainting.
            kwargs (dict): Additional keyword arguments passed to the class.
        """
        assert (
            model in self.MODELS or model in self.EXTERNAL_MODELS
        ), f"Model {model} is not supported."
        self.model = model
        self.variant = ""
        self._dimension = 512
        self._supports_feathering = True
        self.kwargs = kwargs
        self.pipeline = None
        self.server_address = None
        self.workflow_path = None  # use for comfyui

    def __eq__(self, other):
        if not isinstance(other, InpaintingModel):
            return False
        return (
            self.model == other.model
            and self.variant == other.variant
            and self._dimension == other._dimension
        )

    def load_model(self):
        if self.pipeline is not None:
            return self.pipeline

        # diffusers models
        if self.model in self.MODELS:
            return self.load_diffusers_model()

        if self.model in ["automatic1111", "comfyui"]:
            return self.load_external_model()

        if self.model == "stabilityai":
            return self.load_stability_model()

        return None

    def load_diffusers_model(self):
        models = {
            "kandinsky-community/kandinsky-2-2-decoder-inpaint": {
                "dimension": 768,
                "variant": "",
                "pipeline": AutoPipelineForInpainting.from_pretrained,
            },
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": {
                "dimension": 1024,
                "variant": "fp16",
                "pipeline": AutoPipelineForInpainting.from_pretrained,
            },
            "stabilityai/stable-diffusion-3-medium-diffusers": {
                "dimension": 1024,
                "variant": "",
                "pipeline": AutoPipelineForImage2Image.from_pretrained,
            },
        }

        assert self.model in models
        model_info = models[self.model]
        self._dimension = model_info["dimension"]
        self.variant = model_info["variant"]

        kwargs = {}
        if self.variant:
            kwargs["variant"] = self.variant

        self.pipeline = model_info["pipeline"](
            self.model, torch_dtype=torch.float16, **kwargs
        )

        device = torch_get_device()
        self.pipeline.to(device)

        return self.pipeline

    def load_external_model(self):
        assert "server_address" in self.kwargs
        self._dimension = 1024
        self.pipeline = self.kwargs["server_address"]
        self.server_address = self.kwargs["server_address"]
        if self.model == "comfyui":
            assert "workflow_path" in self.kwargs
            self.workflow_path = self.kwargs["workflow_path"]
        return self.pipeline

    def load_stability_model(self):
        assert "api_key" in self.kwargs
        self.pipeline = StabilityAI(self.kwargs["api_key"])
        assert self.pipeline.validate_key()
        self._dimension = None  # the model will handle resizing
        return self.pipeline

    def get_dimension(self):
        return self._dimension

    def inpaint(
        self,
        prompt,
        negative_prompt,
        init_image,
        mask_image,
        strength=0.7,
        guidance_scale=8,
        num_inference_steps=50,
        padding=50,
        blur_radius=50,
        crop=False,
        seed=-1,
    ):
        """
        Perform inpainting on an image using the specified model.

        Args:
            prompt (str): The prompt for the inpainting process.
            negative_prompt (str): The negative prompt for the inpainting process.
            init_image (PIL.Image.Image or np.array): The initial image to be inpainted.
            mask_image (PIL.Image.Image or np.array): The mask image indicating the areas to be inpainted.
            strength (float, optional): The strength of the inpainting process. Defaults to 0.7.
            guidance_scale (int, optional): The guidance scale for the inpainting process. Defaults to 8.
            num_inference_steps (int, optional): The number of inference steps for the inpainting process. Defaults to 50.
            padding (int, optional): The padding value for cropping the image. Defaults to 50.
            blur_radius (int, optional): The blur radius for feathering the mask image. Defaults to 50.
            crop (bool, optional): Whether to crop the image after inpainting. Defaults to False.
            seed (int, optional): The seed value for the inpainting process. Defaults to -1.

        Returns:
            PIL.Image.Image: The inpainted image.
        """
        if not isinstance(init_image, Image.Image):
            init_image = Image.fromarray(init_image)
        if not isinstance(mask_image, Image.Image):
            mask_image = Image.fromarray(mask_image)

        # Convert images to RGBA mode
        if init_image.mode != "RGBA":
            init_image = init_image.convert("RGBA")

        original_init_image = init_image.copy()

        if crop:
            bounding_box = find_square_bounding_box(mask_image, padding=padding)
            init_image = init_image.crop(bounding_box)
            mask_image = mask_image.crop(bounding_box)

        if blur_radius > 0 and self._supports_feathering:
            blurred_mask = feather_mask(mask_image, num_expand=blur_radius)
        else:
            blurred_mask = mask_image

        # resize images to match the model input size
        dimension = self.get_dimension()
        if dimension:
            resize_init_image = init_image.convert("RGB").resize((dimension, dimension))
            resize_mask_image = blurred_mask.resize((dimension, dimension))
        else:
            resize_init_image = init_image.convert("RGB")
            resize_mask_image = blurred_mask

        # XXX - refactor this to make it more readable
        if self.model == "stabilityai/stable-diffusion-3-medium-diffusers":
            image = self.inpaint_image2image(
                resize_init_image,
                prompt,
                negative_prompt,
                strength,
                guidance_scale,
                num_inference_steps,
                seed,
            )
        elif self.model in self.MODELS:
            image = self.inpaint_diffusers(
                resize_init_image,
                resize_mask_image,
                prompt,
                negative_prompt,
                strength,
                guidance_scale,
                num_inference_steps,
                seed,
            )
        elif self.model == "automatic1111":
            image = self.inpaint_automatic1111(
                resize_init_image,
                resize_mask_image,
                prompt,
                negative_prompt,
                strength,
                guidance_scale,
                num_inference_steps,
                seed,
            )
        elif self.model == "comfyui":
            image = inpainting_comfyui(
                self.server_address,
                self.workflow_path,
                resize_init_image,
                resize_mask_image,
                prompt,
                negative_prompt,
                strength=strength,
                cfg_scale=guidance_scale,
                steps=num_inference_steps,
                seed=seed,
            )
        elif self.model == "stabilityai":
            image = self.pipeline.image_to_image(
                resize_init_image,
                prompt,
                negative_prompt=negative_prompt,
                strength=strength,
            )
        else:
            raise ValueError(f"Model {self.model} is not supported.")

        image = image.resize(init_image.size, resample=Image.LANCZOS)
        image = image.convert("RGBA")

        image.putalpha(blurred_mask)

        image = Image.alpha_composite(init_image, image)

        if crop:
            original_init_image.paste(image, bounding_box[:2])
            image = original_init_image

        return image

    def inpaint_automatic1111(
        self,
        resize_init_image,
        resize_mask_image,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        num_inference_steps,
        seed,
    ):
        server_address = self.kwargs["server_address"]
        payload = create_img2img_payload(
            resize_init_image,
            prompt,
            negative_prompt,
            mask_image=resize_mask_image,
            strength=strength,
            steps=num_inference_steps,
            cfg_scale=guidance_scale,
        )
        images = make_img2img_request(server_address, payload)
        if len(images) != 1:
            raise ValueError("Expected one image from the server.")
        return Image.open(BytesIO(base64.b64decode(images[0])))

    def inpaint_diffusers(
        self,
        resize_init_image,
        resize_mask_image,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        num_inference_steps,
        seed,
    ):
        if seed == -1:
            seed = np.random.randint(0, 2**32)
        generator = torch.Generator(torch_get_device()).manual_seed(seed)
        pipeline = self.pipeline
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resize_init_image,
            mask_image=resize_mask_image,
            generator=generator,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        return image

    def inpaint_image2image(
        self,
        resize_init_image,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        num_inference_steps,
        seed,
    ):
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resize_init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        return result.images[0]


def create_inpainting_pipeline(model, workflow, state):
    workflow_path = None
    if model == "comfyui":
        if workflow is not None and len(workflow) > 0:
            workflow_path = state.workflow_path()

            need_to_update = False
            if not workflow_path.exists():
                need_to_update = True
            else:
                old_workflow = workflow_path.read_bytes()
                if old_workflow != workflow:
                    need_to_update = True

            if need_to_update:
                # dcc.Upload always has the format 'data:filetype;base64,'
                workflow = workflow.split(",")[1]
                workflow = base64.b64decode(workflow)
                workflow_path.write_bytes(workflow)
                print("ComfyUI workflow updated")

    pipeline = InpaintingModel(
        model,
        server_address=state.server_address,
        workflow_path=workflow_path,
        api_key=state.api_key,
    )
    if state.pipeline_spec is None or state.pipeline_spec != pipeline:
        state.pipeline_spec = pipeline
        pipeline.load_model()
    return state.pipeline_spec


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


def load_image_from_file(image_path, mode="RGB"):
    image = Image.open(image_path)
    image = image.convert(mode)
    return image


@nb.jit(nopython=True, cache=True)
def find_nearest_alpha(alpha, near_a):  # pragma: no cover
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
                    near_a[height - i - 1, width - j - 1, 2] = near_a[
                        height - i, width - j - 1, 2
                    ]
                if j > 0:
                    near_a[height - i - 1, width - j - 1, 3] = near_a[
                        height - i - 1, width - j, 3
                    ]


@nb.jit(nopython=True, parallel=True)
def patch_pixels(image, mask_image, nearest_alpha, patch_indices):  # pragma: no cover
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
                surrounding_pixels[:count] * weights.reshape(-1, 1), axis=0
            ) / np.sum(weights)
            # alpha blend the patched color with the original color
            alpha = image[i, j, 3]
            orig_color = image[i, j, :3]
            image[i, j, :3] = (
                alpha / 255.0 * orig_color + (255.0 - alpha) / 255.0 * average_color
            )
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

    # find the alpha channel of the patched image that is new and not part of the original image
    inverted_alpha = 255 - alpha
    new_alpha = np.minimum(mask_image, inverted_alpha)
    new_alpha = np.expand_dims(new_alpha, axis=2)
    new_alpha = new_alpha / 255.0

    blurred_image = cv2.GaussianBlur(image, (155, 155), 0)

    # composite
    image[:, :, :3] = (
        image[:, :, :3] * (1 - new_alpha) + blurred_image[:, :, :3] * new_alpha
    )

    return image


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Inpain images based on a mask and prompt."
    )
    parser.add_argument("-i", "--image", type=str, help="Path to the input image")
    parser.add_argument("-m", "--mask", type=str, help="Path to the mask image")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="inpainted_image.png",
        help="Path to the output filename",
    )
    parser.add_argument(
        "-a",
        "--patch",
        action="store_true",
        help="Patch the image using the fast inpainting algorithm.",
    )
    parser.add_argument(
        "-s",
        "--strength",
        type=float,
        default=0.7,
        help="The strength of diffusion applied to the image",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k",
        help="The positive prompt to use for inpainting",
    )
    parser.add_argument(
        "-n",
        "--negative-prompt",
        type=str,
        default="bad anatomy, deformed, ugly, disfigured",
        help="The negative prompt to use for inpainting",
    )
    args = parser.parse_args()

    init_image = load_image_from_file(args.image, mode="RGBA")
    mask_image = load_image_from_file(args.mask, mode="L")

    # blur the mask image to feather the edges
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(55))

    init_image = patch_image(np.array(init_image), np.array(mask_image))
    init_image = Image.fromarray(init_image)

    if not args.patch:
        pipeline = InpaintingModel("stabilityai/stable-diffusion-3-medium-diffusers")
        # pipeline = InpaintingModel(
        #    'automatic1111', server_address='localhost:7860')
        # pipeline = InpaintingModel(
        #    'comfyui',
        #    workflow_path='workflow.json',
        #    server_address='localhost:8188')

        pipeline.load_model()

        new_image = pipeline.inpaint(
            args.prompt,
            args.negative_prompt,
            init_image,
            mask_image,
            strength=args.strength,
            crop=True,
        )
    else:
        new_image = init_image

    new_image.save(args.output, format="PNG")

    image = make_image_grid([init_image, mask_image, new_image], rows=1, cols=3)

    image = np.array(image)

    cv2.imshow("Inpainting", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
