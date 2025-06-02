import requests
import base64
from typing import Optional, List, Dict, Any
from webscout.Provider.TTI.utils import (
    ImageData,
    ImageResponse,
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
        aspect_ratio: str = "1:1",
        timeout: int = 60,
        image_format: str = "png",
        seed: Optional[int] = None,
        **kwargs,
    ) -> ImageResponse:
        """
        Create images using the Imagen API.

        Args:
            model: The model to use (e.g., "imagen_3_5")
            prompt: The text prompt for image generation
            n: Number of images to generate
            size: Image size (e.g., "1024x1024")
            response_format: "url" or "b64_json"
            timeout: Request timeout in seconds
            **kwargs: Additional parameters

        Returns:
            ImageResponse: The generated images
        """
        if not prompt:
            raise ValueError("Prompt is required!")

        result_data = []

        for _ in range(n):
            # Prepare the request payload
            payload = {
                "prompt": prompt,
                "model": model,
                "size": size,
                "response_format": "url",  # Always request URL from API
            }

            try:
                # Make the API request
                resp = self._client.session.request(
                    "post",
                    self._client.api_endpoint,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()

                # Parse the response
                result = resp.json()

                if not result or "data" not in result:
                    raise RuntimeError("Invalid response from Imagen API")

                # Process each image in the response
                for item in result["data"]:
                    if response_format == "url":
                        if "url" in item and item["url"]:
                            result_data.append(ImageData(url=item["url"]))
                        else:
                            raise RuntimeError("No URL found in API response")

                    elif response_format == "b64_json":
                        if "url" in item and item["url"]:
                            # Download the image and convert to base64
                            img_resp = self._client.session.request(
                                "get",
                                item["url"],
                                timeout=timeout,
                            )
                            img_resp.raise_for_status()
                            img_bytes = img_resp.content
                            b64_string = base64.b64encode(img_bytes).decode("utf-8")
                            result_data.append(ImageData(b64_json=b64_string))
                        elif "b64_json" in item and item["b64_json"]:
                            result_data.append(ImageData(b64_json=item["b64_json"]))
                        else:
                            raise RuntimeError("No image data found in API response")

                    else:
                        raise ValueError("response_format must be 'url' or 'b64_json'")

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to generate image with Imagen API: {e}")
            except Exception as e:
                raise RuntimeError(f"Error processing Imagen API response: {e}")

        return ImageResponse(created=int(time.time()), data=result_data)


class ImagenAI(TTICompatibleProvider):
    """
    Imagen API provider for text-to-image generation.

    This provider interfaces with the Imagen API at imagen.exomlapi.com
    to generate images from text prompts.
    """

    AVAILABLE_MODELS = ["imagen_3_5", "imagen_3"]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Imagen API client.

        Args:
            api_key: Optional API key for authentication (if required)
        """
        self.api_endpoint = "https://imagen.exomlapi.com/v1/images/generations"
        self.session = requests.Session()
        self.user_agent = LitAgent().random()
        self.api_key = api_key

        # Set up headers based on the provided request details
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://imagen.exomlapi.com",
            "referer": "https://imagen.exomlapi.com/",
            "sec-ch-ua": '"Chromium";v="136", "Microsoft Edge";v="136", "Not.A/Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": self.user_agent,
        }

        # Add API key to headers if provided
        if self.api_key:
            self.headers["authorization"] = f"Bearer {self.api_key}"

        self.session.headers.update(self.headers)
        self.images = Images(self)

    @property
    def models(self):
        """
        Get available models for the Imagen API.

        Returns:
            Object with list() method that returns available models
        """

        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS

        return _ModelList()


if __name__ == "__main__":
    from rich import print

    # Example usage
    client = ImagenAI()

    try:
        response = client.images.create(
            model="imagen_3_5",
            prompt="red car",
            response_format="url",
            n=1,
            size="1024x1024",
            timeout=30,
        )
        print("Generated image successfully:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
