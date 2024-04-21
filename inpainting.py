#!/usr/bin/env python
# (c) 2024 Niels Provos
#

import argparse
import io
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import cv2
import numpy as np
from PIL import Image, ImageFilter
from utils import torch_get_device, find_square_bounding_box

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


def inpaint(pipelinespec, prompt, negative_prompt, init_image, mask_image, crop=False, seed=-1):
    original_init_image = init_image.copy()
    
    if crop:
        bounding_box = find_square_bounding_box(mask_image)
        init_image = init_image.crop(bounding_box)
        mask_image = mask_image.crop(bounding_box)

    pipeline = pipelinespec.pipeline
    if hasattr(pipeline, 'mask_processor'):
        blurred_mask = pipeline.mask_processor.blur(mask_image, blur_factor=20)
    else:
        blurred_mask = mask_image.filter(ImageFilter.GaussianBlur(20))

    # resize images to match the model input size
    dimension = pipelinespec.get_dimension()
    resize_init_image = init_image.convert(
        'RGB').resize((dimension, dimension))
    resize_mask_image = blurred_mask.resize((dimension, dimension))

    if seed == -1:
        seed = np.random.randint(0, 2**32)
    generator = torch.Generator(torch_get_device()).manual_seed(seed)
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt,
                     image=resize_init_image, mask_image=resize_mask_image, generator=generator).images[0]

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




def main():
    parser = argparse.ArgumentParser(
        description='Inpain images based on a mask and prompt.')
    parser.add_argument('-i', '--image', type=str,
                        help='Path to the input image')
    parser.add_argument('-m', '--mask', type=str,
                        help='Path to the mask image')
    parser.add_argument('-o', '--output', type=str,
                        default='inpainted_image.png', help='Path to the output filename')
    args = parser.parse_args()

    pipeline = PipelineSpec(
        'kandinsky-community/kandinsky-2-2-decoder-inpaint',
        variant='',
        dimension=512)

    pipeline.create_pipeline()

    init_image = load_image_from_file(args.image, mode='RGBA')
    mask_image = load_image_from_file(args.mask, mode='L')

    prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured"

    new_image = inpaint(pipeline, prompt, negative_prompt, init_image, mask_image, crop=True)

    new_image.save(args.output, format='PNG')

    image = make_image_grid(
        [init_image, mask_image, new_image], rows=1, cols=3)

    image = np.array(image)

    cv2.imshow("Inpainting", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
