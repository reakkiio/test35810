from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Generator
from .utils import ImageResponse
import random

class BaseImages(ABC):
    @abstractmethod
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
        aspect_ratio: str = "1:1",
        timeout: int = None,
        image_format: str = "png",
        seed: Optional[int] = None,
        **kwargs
    ) -> ImageResponse:
        """
        Abstract method to create images from a prompt.

        Args:
            model: The model to use for image generation.
            prompt: The prompt for the image.
            n: Number of images to generate.
            size: Image size.
            response_format: "url" or "b64_json".
            user: Optional user identifier.
            style: Optional style.
            aspect_ratio: Optional aspect ratio.
            timeout: Request timeout in seconds.
            image_format: "png" or "jpeg" for output format.
            seed: Optional random seed for reproducibility.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ImageResponse: The generated images.
        """
        raise NotImplementedError

class TTICompatibleProvider(ABC):
    """
    Abstract Base Class for TTI providers mimicking the OpenAI Python client structure.
    Requires a nested 'images.create' structure.
    """
    images: BaseImages

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass

    @property
    @abstractmethod
    def models(self):
        """
        Property that returns an object with a .list() method returning available models.
        Subclasses must implement this property.
        """
        pass
