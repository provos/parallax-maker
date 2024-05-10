import base64
import requests
from PIL import Image
from io import BytesIO
import json
import argparse

from utils import to_image_url, image_from_data_url

MAX_DIMENSION = 2048


def create_img2img_payload(
        input_image, positive_prompt, negative_prompt,
        mask_image=None,
        strength=0.8,
        steps=20, cfg_scale=8.0):
    """
    Create a payload for the img2img API.

    Args:
        input_image (PIL.Image.Image): The input image.
        positive_prompt (str): The positive prompt for generating the output image.
        negative_prompt (str): The negative prompt for generating the output image.
        mask_image (PIL.Image.Image, optional): The mask for the input image. Defaults to None.
        strength (float, optional): The denoising strength. Defaults to 0.8.
        steps (int, optional): The number of steps for generating the output image. Defaults to 20.
        cfg_scale (float, optional): The scale factor for the configuration. Defaults to 8.0.

    Returns:
        dict: The payload for the img2img API.
    """
    width, height = input_image.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        # resize the image so the largest dimension is MAX_DIMENSION
        if width > height:
            new_width = MAX_DIMENSION
            new_height = int(MAX_DIMENSION * height / width)
        else:
            new_height = MAX_DIMENSION
            new_width = int(MAX_DIMENSION * width / height)
        input_image = input_image.resize((new_width, new_height))
        mask_image = mask_image.resize((new_width, new_height))

    payload = {
        "init_images": [
            to_image_url(input_image)
        ],
        "prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": strength,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale
    }
    
    if mask_image is not None:
        payload["mask"] = to_image_url(mask_image)
    
    return payload


def make_img2img_request(server_address, payload):
    server_url = f"http://{server_address}/sdapi/v1/img2img"
    payload = json.dumps(payload)
    resp = requests.post(url=server_url, data=payload, timeout=500).json()

    return resp.get("images")


if __name__ == '__main__':
    # Parse arguments - input image, positive prompt, negative prompt, server address
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input-image', type=str,
                        help='Path to the input image')
    parser.add_argument('-m', '--mask-image', type=str,
                        default=None,
                        help='Path to the mask image')
    parser.add_argument('-p', '--prompt', type=str, help='Positive prompt')
    parser.add_argument('-n', '--negative-prompt', type=str,
                        default='', help='Negative prompt')
    parser.add_argument('-e', '--steps', type=int,
                        default=30, help='Number of steps')
    parser.add_argument('-s', '--server', type=str, help='Server address')
    args = parser.parse_args()

    image = Image.open(args.input_image)
    mask_image = None
    if args.mask_image is not None:
        mask_image = Image.open(args.mask_image)

    payload = create_img2img_payload(
        image, args.prompt, args.negative_prompt, mask_image=mask_image, steps=args.steps)
    images = make_img2img_request(args.server, payload)
    if images is None:
        print("No response from server")
    else:
        for image in images:
            image = Image.open(BytesIO(base64.b64decode(image)))
            image.show()
            input("Press Enter to continue...")
