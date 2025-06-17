from PIL import Image
from io import BytesIO
import requests
import argparse


class StabilityAI:
    """
    Implements the Stability AI API for generating and inpainting images.

    This API has several limitations:
    - Inpainting masks cannot be feathered or have soft edges. Consequently, the inpainting does not work with low strength.
    """

    MAX_PIXELS = 9437184

    def __init__(self, api_key):
        self.api_key = api_key

    def validate_key(self):
        response = requests.get(
            "https://api.stability.ai/v1/user/balance",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        if response.status_code != 200:
            return False, None

        # Do something with the payload...
        payload = response.json()
        assert "credits" in payload, f"Invalid API response: {payload}."
        return True, payload["credits"]

    def generate_image(
        self, prompt, negative_prompt="", aspect_ratio="16:9", output_format="png"
    ):
        """
        Generates an image using the Stability AI API.

        Args:
            prompt (str): The main text prompt for generating the image.
            negative_prompt (str, optional): The negative text prompt for generating the image. Defaults to ''.
            aspect_ratio (str, optional): The aspect ratio of the generated image. Defaults to "16:9".
            output_format (str, optional): The output format of the generated image. Defaults to "png".

        Returns:
            PIL.Image.Image: The generated image as a PIL Image object.

        Raises:
            Exception: If the API response status code is not 200.
        """
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
            files={"none": ""},
            data={
                "mode": "text-to-image",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
            },
        )

        if response.status_code != 200:
            raise Exception(str(response.json()))

        return Image.open(BytesIO(response.content))

    def image_to_image(
        self, image, prompt, negative_prompt="", strength=0.8, output_format="png"
    ):
        """
        Generates an image-to-image request using the Stability AI API.

        Args:
            image (PIL.Image.Image): The input image to be transformed.
            prompt (str): The prompt to guide the inpainting process.
            negative_prompt (str, optional): The negative prompt to guide the inpainting process. Defaults to ''.
            strength (float, optional): The strength of the transformation effect. Defaults to 0.8.
            output_format (str, optional): The format of the output image. Defaults to "png".

        Returns:
            PIL.Image.Image: The inpainted image.

        Raises:
            ValueError: If the image is smaller than 64x64 pixels.
            Exception: If the API request fails.

        """
        # check that the image it at least 64x64
        if image.width < 64 or image.height < 64:
            raise ValueError("Image must be at least 64x64")

        orig_width, orig_height = image.width, image.height

        image = self._resize_image(image)

        # prepare image and mask data
        image_data = BytesIO()
        image.save(image_data, format="PNG")
        image_data = image_data.getvalue()

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
            files={"image": image_data},
            data={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "mode": "image-to-image",
                "strength": strength,
                "output_format": output_format,
            },
        )

        if response.status_code != 200:
            raise Exception(str(response.json()))

        image = Image.open(BytesIO(response.content))
        if orig_width != image.width or orig_height != image.height:
            image = image.resize((orig_width, orig_height), Image.LANCZOS)
        return image

    def _resize_image(self, image, max_pixels=MAX_PIXELS):
        if image.width * image.height > max_pixels:
            # square root as with multiply the ratio twice below
            ratio = (max_pixels / (image.width * image.height)) ** 0.5
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image

    def inpaint_image(
        self, image, mask, prompt, negative_prompt="", strength=1.0, output_format="png"
    ):
        """
        Inpaints an image using the Stability AI API.

        Args:
            image (PIL.Image.Image): The input image to be inpainted.
            mask (PIL.Image.Image): The mask indicating the areas to be inpainted.
            prompt (str): The prompt to guide the inpainting process.
            negative_prompt (str, optional): The negative prompt to guide the inpainting process. Defaults to ''.
            strength (float, optional): The strength of the inpainting effect. Defaults to 1.0.
            output_format (str, optional): The format of the output image. Defaults to "png".

        Returns:
            PIL.Image.Image: The inpainted image.

        Raises:
            ValueError: If the image is smaller than 64x64 pixels.
            Exception: If the API request fails.

        """
        # check that the image it at least 64x64
        if image.width < 64 or image.height < 64:
            raise ValueError("Image must be at least 64x64")

        orig_width, orig_height = image.width, image.height

        image = self._resize_image(image)
        mask = self._resize_image(mask)

        # scale the mask by strength - the API does not work well with low strength
        if strength < 1.0:
            mask = mask.point(lambda p: p * strength)

        # prepare image and mask data
        image_data = BytesIO()
        image.save(image_data, format="PNG")
        image_data = image_data.getvalue()
        mask_data = BytesIO()
        mask.save(mask_data, format="PNG")
        mask_data = mask_data.getvalue()

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/edit/inpaint",
            headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
            files={"image": image_data, "mask": mask_data},
            data={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "output_format": output_format,
            },
        )

        if response.status_code != 200:
            raise Exception(str(response.json()))

        image = Image.open(BytesIO(response.content))
        if orig_width != image.width or orig_height != image.height:
            image = image.resize((orig_width, orig_height), Image.LANCZOS)
        return image

    def upscale_image(self, image, prompt, negative_prompt="", output_format="png"):
        # check that the image it at least 64x64
        if image.width < 64 or image.height < 64:
            raise ValueError("Image must be at least 64x64")

        if image.width * image.height > 1024 * 1024:
            raise ValueError(
                f"Image must be at most 1024x1024: Got {image.width}x{image.height} = {image.width * image.height} pixels."
            )

        # prepare image and mask data
        image_data = BytesIO()
        image.save(image_data, format="PNG")
        image_data = image_data.getvalue()

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/upscale/conservative",
            headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
            files={"image": image_data},
            data={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "output_format": output_format,
            },
        )

        if response.status_code != 200:
            raise Exception(str(response.json()))

        upscaled_image = Image.open(BytesIO(response.content))
        return upscaled_image


def main():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument(
        "-p",
        "--prompt",
        default="an intelligent AI robot on the surface of the moon; intelligent and cute glowing eyes; "
        + "surrounded by canyons and rocks; dystopian atmosphere; scfi vibe; bladerunner and cyberpunk; "
        + "distant stars visible in the sky behind the mountains; photorealistic; everything in sharp focus",
    )
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--output-format", default="png")
    parser.add_argument("-i", "--image", type=str, default=None)
    parser.add_argument("-m", "--mask", type=str, default=None)
    parser.add_argument("-s", "--strength", type=float, default=0.8)
    parser.add_argument("-u", "--upscale", action="store_true")
    args = parser.parse_args()

    ai = StabilityAI(args.api_key)
    success, _ = ai.validate_key()
    if not success:
        print("Invalid API key.")
        return
    if args.image and args.upscale:
        image = Image.open(args.image)
        image = ai._resize_image(image, 1024 * 1024)
        image = ai.upscale_image(image, args.prompt)
    elif args.image and args.mask:
        image = Image.open(args.image)
        mask = Image.open(args.mask)

        image = ai.inpaint_image(image, mask, args.prompt, strength=1.0)
    elif args.image:
        print("Running image to image")
        image = Image.open(args.image)
        image = ai.image_to_image(image, args.prompt, strength=args.strength)
    else:
        image = ai.generate_image(
            args.prompt,
            aspect_ratio=args.aspect_ratio,
            output_format=args.output_format,
        )
    image.save("output.png")


if __name__ == "__main__":
    main()
