import requests
import random
import string
import json
import time
from typing import Optional, List, Dict, Any
from webscout.Provider.TTI.utils import (
    ImageData,
    ImageResponse
)
from webscout.Provider.TTI.base import TTICompatibleProvider, BaseImages
from io import BytesIO
import os
import tempfile
from webscout.litagent import LitAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class Images(BaseImages):
    def __init__(self, client):
        self._client = client
        self.base_url = "https://api.together.xyz/v1"
        # Create a session - it will automatically get proxies from the global monkey patch!
        self.session = requests.Session()
        self._setup_session_with_retries()

    def _setup_session_with_retries(self):
        """Setup session with retry strategy and timeout configurations"""
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_api_key(self) -> str:
        """Get API key from activation endpoint or cache"""
        if hasattr(self._client, '_api_key_cache') and self._client._api_key_cache:
            return self._client._api_key_cache
            
        try:
            activation_endpoint = "https://www.codegeneration.ai/activate-v2"
            response = requests.get(
                activation_endpoint,
                headers={"Accept": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            activation_data = response.json()
            api_key = activation_data["openAIParams"]["apiKey"]
            self._client._api_key_cache = api_key
            return api_key
        except Exception as e:
            raise Exception(f"Failed to get activation key: {e}")

    def build_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build headers with API authorization"""
        api_key = self.get_api_key()
        
        agent = LitAgent()
        fp = agent.generate_fingerprint("chrome")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
            "accept-language": fp["accept_language"],
            "user-agent": fp["user_agent"],
            "sec-ch-ua": fp["sec_ch_ua"],
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
        }
        if extra:
            headers.update(extra)
        return headers

    def create(
        self,
        model: str = None,
        prompt: str = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        user: Optional[str] = None,
        style: str = None,
        aspect_ratio: str = None,
        timeout: int = 120,
        image_format: str = "png",
        enhance: bool = True,
        steps: int = 20,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ImageResponse:
        """
        Create images using Together.xyz image models
        
        Args:
            model: Image model to use (defaults to first available)
            prompt: Text description of the image to generate
            n: Number of images to generate (1-4)
            size: Image size in format "WIDTHxHEIGHT" 
            response_format: "url" or "b64_json"
            timeout: Request timeout in seconds
            steps: Number of inference steps (1-50)
            seed: Random seed for reproducible results
            **kwargs: Additional model-specific parameters
        """
        if not prompt:
            raise ValueError(
                "Describe the image you want to create (use the 'prompt' property)."
            )

        # Use provided model or default to first available
        if not model:
            model = self._client.AVAILABLE_MODELS[0]
        elif model not in self._client.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model}' not available. Choose from: {self._client.AVAILABLE_MODELS}")

        # Parse size
        if 'x' in size:
            width, height = map(int, size.split('x'))
        else:
            width = height = int(size)

        # Build request body
        body = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            # Clamp steps to 1-4 as required by Together.xyz API
            "steps": min(max(steps, 1), 4),
            "n": min(max(n, 1), 4),  # Clamp between 1-4
        }

        # Add optional parameters
        if seed is not None:
            body["seed"] = seed
            
        # Add any additional kwargs
        body.update(kwargs)

        try:
            resp = self.session.request(
                "post",
                f"{self.base_url}/images/generations",
                json=body,
                headers=self.build_headers(),
                timeout=timeout,
            )
            
            data = resp.json()
            
            # Check for errors
            if "error" in data:
                error_msg = data["error"].get("message", str(data["error"]))
                raise RuntimeError(f"Together.xyz API error: {error_msg}")
                
            if not data.get("data") or len(data["data"]) == 0:
                raise RuntimeError("Failed to process image. No data found.")

            result = data["data"]
            result_data = []
            
            for i, item in enumerate(result):
                if response_format == "url":
                    if "url" in item:
                        result_data.append(ImageData(url=item["url"]))
                else:  # b64_json
                    if "b64_json" in item:
                        result_data.append(ImageData(b64_json=item["b64_json"]))
            
            if not result_data:
                raise RuntimeError("No valid image data found in response")
                
            return ImageResponse(data=result_data)
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Request timed out after {timeout} seconds. Try reducing image size or steps.")
        except requests.exceptions.RequestException as e:
            # Print the response content for debugging if available
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print("[Together.xyz API error details]", e.response.text)
                except Exception:
                    pass
            raise RuntimeError(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            raise RuntimeError("Invalid JSON response from Together.xyz API")
        except Exception as e:
            raise RuntimeError(f"An error occurred: {str(e)}")


class TogetherImage(TTICompatibleProvider):
    """
    Together.xyz Text-to-Image provider
    Updated: 2025-08-01 10:42:41 UTC by OEvortex
    Supports FLUX and other image generation models
    """
    
    # Image models from Together.xyz API (filtered for image type only)
    AVAILABLE_MODELS = [
        "black-forest-labs/FLUX.1-canny",
        "black-forest-labs/FLUX.1-depth",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-dev-lora",
        "black-forest-labs/FLUX.1-kontext-dev",
        "black-forest-labs/FLUX.1-kontext-max",
        "black-forest-labs/FLUX.1-kontext-pro",
        "black-forest-labs/FLUX.1-krea-dev",
        "black-forest-labs/FLUX.1-pro",
        "black-forest-labs/FLUX.1-redux",
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-schnell-Free",
        "black-forest-labs/FLUX.1.1-pro"
    ]

    def __init__(self):
        self.images = Images(self)
        self._api_key_cache = None

    @property 
    def models(self):
        class _ModelList:
            def list(inner_self):
                return TogetherImage.AVAILABLE_MODELS
                
        return _ModelList()

    def convert_model_name(self, model: str) -> str:
        """Convert model alias to full model name"""
        if model in self.AVAILABLE_MODELS:
            return model
        
        # Default to first available model
        return self.AVAILABLE_MODELS[0]

    # def fetch_available_models(self) -> List[str]:
    #     """Fetch current image models from Together.xyz API"""
    #     try:
    #         api_key = self.images.get_api_key()
    #         headers = {
    #             "Authorization": f"Bearer {api_key}",
    #             "Accept": "application/json"
    #         }
            
    #         response = requests.get(
    #             "https://api.together.xyz/v1/models", 
    #             headers=headers, 
    #             timeout=30
    #         )
    #         response.raise_for_status()
    #         models_data = response.json()
            
    #         # Filter image models
    #         image_models = []
    #         for model in models_data:
    #             if isinstance(model, dict) and model.get("type", "").lower() == "image":
    #                 image_models.append(model["id"])
            
    #         return sorted(image_models)
        
    #     except Exception as e:
    #         return self.AVAILABLE_MODELS


if __name__ == "__main__":
    from rich import print
    client = TogetherImage()
    
    # Test with a sample prompt
    response = client.images.create(
        model="black-forest-labs/FLUX.1-schnell-Free",  # Free FLUX model
        prompt="A majestic dragon flying over a mystical forest, fantasy art, highly detailed",
        size="1024x1024",
        n=1,
        steps=25,
        response_format="url",
        timeout=120,
    )
    print(response)