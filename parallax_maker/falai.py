#!/usr/bin/env python
# (c) 2024 Niels Provos

import os
from PIL import Image
from io import BytesIO
import fal_client
import requests
import argparse
from typing import Optional, Tuple, Dict, Any


class FalAI:
    """
    Implements the fal.ai API for inpainting images using various models.
    
    Supports:
    - Foocus inpainting
    - Flux general inpainting
    - SD/SDXL inpainting
    """
    
    # Model endpoints on fal.ai
    MODELS = {
        "foocus": "fal-ai/fooocus/inpaint",
        "flux-general": "fal-ai/flux-general/inpainting",
        "sd": "fal-ai/inpaint",  # Supports both SD and SDXL
        "sdxl": "fal-ai/inpaint"
    }
    
    def __init__(self, api_key: str):
        """
        Initialize the FalAI client with an API key.
        
        Args:
            api_key (str): The fal.ai API key
        """
        self.api_key = api_key
        # Set the API key as environment variable for fal_client
        os.environ["FAL_KEY"] = api_key
    
    def validate_key(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the API key by making a simple request.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            # Try a simple model call with minimal parameters
            result = fal_client.run(
                "fal-ai/fast-sdxl",
                arguments={
                    "prompt": "test",
                    "image_size": "square_hd",
                    "num_inference_steps": 1,
                    "num_images": 1
                }
            )
            return True, None
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return False, "Invalid API key"
            return False, f"Connection error: {error_msg}"
    
    def _download_image(self, url: str) -> Image.Image:
        """Download an image from a URL and return as PIL Image."""
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    
    def _prepare_image_for_upload(self, image: Image.Image) -> str:
        """
        Prepare image for upload to fal.media CDN.
        
        Args:
            image: PIL Image to upload
            
        Returns:
            str: URL of uploaded image
        """
        # Save image to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        # Upload to fal.media
        url = fal_client.upload(img_bytes.getvalue(), "image/png")
        return url
    
    def inpaint_foocus(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: int = -1
    ) -> Image.Image:
        """
        Inpaint using Foocus model.
        
        Args:
            image: Input image
            mask: Mask indicating areas to inpaint
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            strength: Inpainting strength (0-1)
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps
            seed: Random seed (-1 for random)
            
        Returns:
            Inpainted image
        """
        # Upload images
        image_url = self._prepare_image_for_upload(image)
        mask_url = self._prepare_image_for_upload(mask)
        
        arguments = {
            "inpaint_image_url": image_url,
            "mask_image_url": mask_url,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "inpaint_strength": strength,
            "inpaint_respective_field": 0.618,  # Default value from API docs
            "inpaint_engine": "v2.6",  # Latest version
            "seed": seed if seed != -1 else None,
            "num_images": 1,
            "enable_safety_checker": False
        }
        
        result = fal_client.run(
            self.MODELS["foocus"],
            arguments=arguments
        )
        
        # Download and return the first generated image
        image_url = result["images"][0]["url"]
        return self._download_image(image_url)
    
    def inpaint_flux_general(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.8,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        seed: int = -1
    ) -> Image.Image:
        """
        Inpaint using Flux general inpainting model.
        
        Args:
            image: Input image
            mask: Mask indicating areas to inpaint
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (not used by Flux)
            strength: Inpainting strength (0-1)
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps
            seed: Random seed (-1 for random)
            
        Returns:
            Inpainted image
        """
        # Upload images
        image_url = self._prepare_image_for_upload(image)
        mask_url = self._prepare_image_for_upload(mask)
        
        arguments = {
            "image_url": image_url,
            "mask_url": mask_url,
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "seed": seed if seed != -1 else None,
            "num_images": 1,
            "enable_safety_checker": False
        }
        
        result = fal_client.run(
            self.MODELS["flux-general"],
            arguments=arguments
        )
        
        # Download and return the first generated image
        image_url = result["images"][0]["url"]
        return self._download_image(image_url)
    
    def inpaint_sd(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        model_type: str = "sdxl",
        seed: int = -1
    ) -> Image.Image:
        """
        Inpaint using Stable Diffusion or SDXL model.
        
        Args:
            image: Input image
            mask: Mask indicating areas to inpaint
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            strength: Inpainting strength (0-1)
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps
            model_type: "sd" or "sdxl"
            seed: Random seed (-1 for random)
            
        Returns:
            Inpainted image
        """
        # Upload images
        image_url = self._prepare_image_for_upload(image)
        mask_url = self._prepare_image_for_upload(mask)
        
        arguments = {
            "image_url": image_url,
            "mask_url": mask_url,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "model": model_type,  # "sd" or "sdxl"
            "seed": seed if seed != -1 else None,
            "num_images": 1,
            "enable_safety_checker": False,
            "sync_mode": True
        }
        
        result = fal_client.run(
            self.MODELS[model_type],
            arguments=arguments
        )
        
        # Download and return the first generated image
        image_url = result["images"][0]["url"]
        return self._download_image(image_url)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Test fal.ai inpainting")
    parser.add_argument("--api-key", required=True, help="fal.ai API key")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-m", "--mask", required=True, help="Mask image path")
    parser.add_argument(
        "-p", "--prompt", 
        default="a beautiful landscape",
        help="Prompt for inpainting"
    )
    parser.add_argument(
        "-n", "--negative-prompt",
        default="",
        help="Negative prompt"
    )
    parser.add_argument(
        "--model",
        choices=["foocus", "flux-general", "sd", "sdxl"],
        default="sdxl",
        help="Model to use"
    )
    parser.add_argument(
        "-s", "--strength",
        type=float,
        default=0.8,
        help="Inpainting strength"
    )
    parser.add_argument(
        "-o", "--output",
        default="output_falai.png",
        help="Output filename"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = FalAI(args.api_key)
    
    # Validate API key
    success, error = client.validate_key()
    if not success:
        print(f"API key validation failed: {error}")
        return
    
    print("API key validated successfully")
    
    # Load images
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    
    # Perform inpainting
    print(f"Inpainting with {args.model} model...")
    
    if args.model == "foocus":
        result = client.inpaint_foocus(
            image, mask, args.prompt, args.negative_prompt, 
            strength=args.strength
        )
    elif args.model == "flux-general":
        result = client.inpaint_flux_general(
            image, mask, args.prompt, args.negative_prompt,
            strength=args.strength
        )
    else:  # sd or sdxl
        result = client.inpaint_sd(
            image, mask, args.prompt, args.negative_prompt,
            strength=args.strength, model_type=args.model
        )
    
    # Save result
    result.save(args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()