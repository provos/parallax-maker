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
from PIL import Image
from utils import torch_get_device

# pretrained_model = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
# pretrained_model = "runwayml/stable-diffusion-v1-5"
# pretrained_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

def create_inpainting_pipeline(pretrained_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
    pipeline = AutoPipelineForInpainting.from_pretrained(
        pretrained_model, torch_dtype=torch.float16, variant="fp16"
    )
    
    device = torch_get_device()
    pipeline.to(device)
    return pipeline

def inpaint(pipeline, prompt, negative_prompt, init_image, mask_image, seed=92):
    generator = torch.Generator(torch_get_device()).manual_seed(seed)
    
    # resize images to 1024x1024 to match sdxl model input size
    resize_init_image = init_image.resize((1024, 1024))
    resize_mask_image = mask_image.resize((1024, 1024))
    
    blurred_mask = pipeline.mask_processor.blur(resize_mask_image, blur_factor=20)

    image = pipeline(prompt=prompt, negative_prompt=negative_prompt,
                     image=resize_init_image, mask_image=blurred_mask, generator=generator).images[0]

    image = image.resize(init_image.size)

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
                        default='.', help='Path to the output directory')
    args = parser.parse_args()
    
    pipeline = create_inpainting_pipeline()

    init_image = load_image_from_file(args.image, mode='RGB')
    mask_image = load_image_from_file(args.mask, mode='L')
        
    prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured"

    image = inpaint(pipeline, prompt, negative_prompt, init_image, mask_image)

    image = make_image_grid([init_image, mask_image, image], rows=1, cols=3)

    image = np.array(image)

    cv2.imshow("Inpainting", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()