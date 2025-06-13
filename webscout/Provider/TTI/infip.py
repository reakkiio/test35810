"""InfipAI TTI-Compatible Provider - Generate images with Infip AI! 🎨

This module provides access to Infip's image generation API through a unified interface.
Supports img3, img4, and uncen models with various aspect ratios and customization options.

Example Usage:
    from webscout.Provider.TTI.infip import InfipAI
    
    # Initialize the provider
    client = InfipAI()
    
    # Generate an image
    response = client.images.create(
        model="img3",
        prompt="A beautiful sunset over mountains",
        n=1,
        aspect_ratio="IMAGE_ASPECT_RATIO_LANDSCAPE",
        seed=42
    )
    
    # Get the image URL
    image_url = response.data[0].url
    print(f"Generated image: {image_url}")

Available Models:
    - img3: High-quality image generation
    - img4: Enhanced image generation model
    - uncen: Uncensored image generation model

Supported Aspect Ratios:
    - IMAGE_ASPECT_RATIO_LANDSCAPE: 16:9 landscape
    - IMAGE_ASPECT_RATIO_PORTRAIT: 9:16 portrait  
    - IMAGE_ASPECT_RATIO_SQUARE: 1:1 square
"""

import requests
from typing import Optional
from webscout.Provider.TTI.utils import (
    ImageData,
    ImageResponse
)
from webscout.Provider.TTI.base import TTICompatibleProvider, BaseImages
from webscout.litagent import LitAgent
import time


class Images(BaseImages):
    def __init__(self, client):
        self._client = client

    def create(
        self,
        *,
        model: str,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        user: Optional[str] = None,
        style: str = "none",
        aspect_ratio: str = "IMAGE_ASPECT_RATIO_LANDSCAPE",
        timeout: int = 60,
        image_format: str = "png",
        seed: Optional[int] = None,
        **kwargs,
    ) -> ImageResponse:
        """
        Create images using Infip AI API.

        Args:
            model: The model to use ("img3", "img4", or "uncen")
            prompt: Text description of the image to generate
            n: Number of images to generate (default: 1)
            size: Image size (ignored, aspect_ratio is used instead)
            response_format: "url" or "b64_json" (default: "url")
            user: Optional user identifier (ignored)
            style: Optional style (ignored)
            aspect_ratio: Image aspect ratio ("IMAGE_ASPECT_RATIO_LANDSCAPE", 
                         "IMAGE_ASPECT_RATIO_PORTRAIT", "IMAGE_ASPECT_RATIO_SQUARE")
            timeout: Request timeout in seconds (default: 60)
            image_format: Image format "png" or "jpeg" (ignored by API)
            seed: Random seed for reproducibility (default: 0 for random)
            **kwargs: Additional parameters

        Returns:
            ImageResponse: The generated images

        Raises:
            ValueError: If model is not supported
            RuntimeError: If image generation fails
        """
        if model not in self._client.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model}' not supported. Available models: {self._client.AVAILABLE_MODELS}")

        # Validate aspect ratio
        valid_ratios = [
            "IMAGE_ASPECT_RATIO_LANDSCAPE",
            "IMAGE_ASPECT_RATIO_PORTRAIT", 
            "IMAGE_ASPECT_RATIO_SQUARE"
        ]
        if aspect_ratio not in valid_ratios:
            aspect_ratio = "IMAGE_ASPECT_RATIO_LANDSCAPE"

        # Prepare request payload
        payload = {
            "prompt": prompt,
            "num_images": n,
            "seed": seed if seed is not None else 0,
            "aspect_ratio": aspect_ratio,
            "models": model
        }

        try:
            # Make API request
            response = self._client.session.post(
                self._client.api_endpoint,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            if "images" not in result or not result["images"]:
                raise RuntimeError("No images returned from Infip API")

            # Process response based on format
            result_data = []
            
            if response_format == "url":
                for image_url in result["images"]:
                    result_data.append(ImageData(url=image_url))
            elif response_format == "b64_json":
                # For b64_json format, we need to download and encode the images
                import base64
                for image_url in result["images"]:
                    try:
                        img_response = self._client.session.get(image_url, timeout=timeout)
                        img_response.raise_for_status()
                        b64_data = base64.b64encode(img_response.content).decode('utf-8')
                        result_data.append(ImageData(b64_json=b64_data))
                    except Exception as e:
                        raise RuntimeError(f"Failed to download image for base64 encoding: {e}")
            else:
                raise ValueError("response_format must be 'url' or 'b64_json'")

            return ImageResponse(created=int(time.time()), data=result_data)

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to generate image with Infip API: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during image generation: {e}")


class InfipAI(TTICompatibleProvider):
    """
    Infip AI provider for text-to-image generation.
    
    This provider interfaces with the Infip API to generate images from text prompts.
    It supports multiple models and aspect ratios for flexible image creation.
    """
    
    AVAILABLE_MODELS = ["img3", "img4", "uncen"]

    def __init__(self, **kwargs):
        """
        Initialize the Infip AI provider.
        
        Args:
            **kwargs: Additional configuration options
        """
        self.api_endpoint = "https://api.infip.pro/generate"
        self.session = requests.Session()
        
        # Set up headers with user agent
        agent = LitAgent()
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": agent.random()
        }
        self.session.headers.update(self.headers)
        
        # Initialize the images interface
        self.images = Images(self)

    @property
    def models(self):
        """
        Get available models for the provider.
        
        Returns:
            Object with list() method that returns available model names
        """
        class ModelList:
            def list(self):
                return InfipAI.AVAILABLE_MODELS
        
        return ModelList()


if __name__ == "__main__":
    client = InfipAI()
    response = client.images.create(
        model="img3",
        prompt="A beautiful sunset over mountains",
        n=1,
        aspect_ratio="IMAGE_ASPECT_RATIO_LANDSCAPE",
        seed=42
    )
    print(response.data[0].url)