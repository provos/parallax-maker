from PIL import Image
from io import BytesIO
import requests
import argparse

class StabilityAI:
    MAX_PIXELS = 9437184
    
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_image(self, prompt, negative_prompt='', aspect_ratio="16:9", output_format="png"):
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
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={"none": ''},
            data={
                'mode': "text-to-image",
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'aspect_ratio': aspect_ratio,
                'output_format': output_format,
            },
        )

        if response.status_code != 200:
            raise Exception(str(response.json()))

        return Image.open(BytesIO(response.content))
    
    def inpaint_image(self, prompt, image, mask, strength=1.0, negative_prompt='', output_format="png"):
        """
        Inpaints an image using the Stability AI API.

        Args:
            prompt (str): The prompt to guide the inpainting process.
            image (PIL.Image.Image): The input image to be inpainted.
            mask (PIL.Image.Image): The mask indicating the areas to be inpainted.
            strength (float, optional): The strength of the inpainting effect. Defaults to 1.0.
            negative_prompt (str, optional): The negative prompt to guide the inpainting process. Defaults to ''.
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
        
        # ensure that the image is not larger than 9,437,184 pixels
        if image.width * image.height > self.MAX_PIXELS:
            # square root as with multiply the ratio twice below
            ratio = (self.MAX_PIXELS / (image.width * image.height)) ** 0.5
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            mask = mask.resize((new_width, new_height), Image.LANCZOS)
        
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
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={"image": image_data, "mask": mask_data},
            data={
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'output_format': output_format,
            }
        )
        
        if response.status_code != 200:
            raise Exception(str(response.json()))
        
        image = Image.open(BytesIO(response.content))
        if orig_width != image.width or orig_height != image.height:
            image = image.resize((orig_width, orig_height), Image.LANCZOS)
        return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument('-p', "--prompt",
                        default='an intelligent AI robot on the surface of the moon; intelligent and cute glowing eyes; ' + 
                        'surrounded by canyons and rocks; dystopian atmosphere; scfi vibe; bladerunner and cyberpunk; ' + 
                        'distant stars visible in the sky behind the mountains; photorealistic; everything in sharp focus')
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--output-format", default="png")
    parser.add_argument('-i', '--image', type=str, default=None)
    parser.add_argument('-m', '--mask', type=str, default=None)
    args = parser.parse_args()

    ai = StabilityAI(args.api_key)
    
    if args.image and args.mask:
        image = Image.open(args.image)
        mask = Image.open(args.mask)        
        image = ai.inpaint_image(args.prompt, image, mask, strength=1.0)
    else:
        image = ai.generate_image(args.prompt, aspect_ratio=args.aspect_ratio, output_format=args.output_format)
    image.save("output.png")

if __name__ == "__main__":
    main()