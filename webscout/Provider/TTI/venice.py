"""Venice AI TTI-Compatible Provider - Generate images with Venice AI! 🎨

Examples:
    >>> from webscout.Provider.TTI.venice import VeniceAI
    >>> client = VeniceAI()
    >>> response = client.images.create(
    ...     model="stable-diffusion-3.5-rev2",
    ...     prompt="red car",
    ...     n=1
    ... )
    >>> print(response)
"""

import requests
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
import time
import json
import random
import string

try:
    from PIL import Image
except ImportError:
    Image = None


class Images(BaseImages):
    def __init__(self, client: "VeniceAI"):
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
        cfg_scale: float = 3.5,
        steps: int = 25,
        **kwargs,
    ) -> ImageResponse:
        """
        Create images using Venice AI API.
        
        Args:
            model: The model to use for image generation
            prompt: Text description of the image to generate
            n: Number of images to generate (default: 1)
            size: Image size in format "WIDTHxHEIGHT" (default: "1024x1024")
            response_format: "url" or "b64_json" (default: "url")
            user: Optional user identifier (ignored)
            style: Optional style (ignored)
            aspect_ratio: Image aspect ratio (default: "1:1")
            timeout: Request timeout in seconds (default: 60)
            image_format: Output image format "png" or "jpeg" (default: "png")
            seed: Random seed for reproducibility (optional)
            cfg_scale: CFG scale for generation (default: 3.5)
            steps: Number of inference steps (default: 25)
            **kwargs: Additional parameters
            
        Returns:
            ImageResponse: The generated images
        """
        if Image is None:
            raise ImportError("Pillow (PIL) is required for image format conversion.")

        if model not in self._client.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model}' not supported. Available models: {self._client.AVAILABLE_MODELS}")

        images = []
        urls = []
        agent = LitAgent()

        def upload_file_with_retry(img_bytes, image_format, max_retries=3):
            ext = "jpg" if image_format.lower() == "jpeg" else "png"
            for attempt in range(max_retries):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=f".{ext}", delete=False
                    ) as tmp:
                        tmp.write(img_bytes)
                        tmp.flush()
                        tmp_path = tmp.name
                    with open(tmp_path, "rb") as f:
                        files = {"fileToUpload": (f"image.{ext}", f, f"image/{ext}")}
                        data = {"reqtype": "fileupload", "json": "true"}
                        headers = {"User-Agent": agent.random()}
                        if attempt > 0:
                            headers["Connection"] = "close"
                        resp = requests.post(
                            "https://catbox.moe/user/api.php",
                            files=files,
                            data=data,
                            headers=headers,
                            timeout=timeout,
                        )
                        if resp.status_code == 200 and resp.text.strip():
                            text = resp.text.strip()
                            if text.startswith("http"):
                                return text
                            try:
                                result = resp.json()
                                if "url" in result:
                                    return result["url"]
                            except json.JSONDecodeError:
                                if "http" in text:
                                    return text
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                finally:
                    if tmp_path and os.path.isfile(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
            return None

        def upload_file_alternative(img_bytes, image_format):
            try:
                ext = "jpg" if image_format.lower() == "jpeg" else "png"
                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp.flush()
                    tmp_path = tmp.name
                try:
                    if not os.path.isfile(tmp_path):
                        return None
                    with open(tmp_path, "rb") as img_file:
                        files = {"file": img_file}
                        response = requests.post("https://0x0.st", files=files)
                        response.raise_for_status()
                        image_url = response.text.strip()
                        if not image_url.startswith("http"):
                            return None
                        return image_url
                except Exception:
                    return None
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception:
                return None

        # Parse size to width and height
        if "x" in size:
            width, height = map(int, size.split("x"))
        else:
            width = height = int(size)

        for _ in range(n):
            # Generate random IDs for the request
            request_id = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
            message_id = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
            user_id = f"user_anon_{''.join(random.choices(string.digits, k=10))}"
            
            # Generate seed if not provided
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            # Prepare the request payload based on the provided example
            payload = {
                "aspectRatio": aspect_ratio,
                "cfgScale": cfg_scale,
                "customSeed": "",
                "embedExifMetadata": True,
                "enhanceCreativity": 0.35,
                "favoriteImageStyles": [],
                "format": "webp",
                "height": height,
                "hideWatermark": False,
                "imageToImageCfgScale": 15,
                "imageToImageStrength": 33,
                "isConstrained": True,
                "isCustomSeed": seed is not None,
                "isDefault": True,
                "loraStrength": 75,
                "matureFilter": True,
                "negativePrompt": kwargs.get("negative_prompt", ""),
                "recentImageStyles": [],
                "replication": 0.35,
                "steps": steps,
                "stylePreset": "",
                "stylesTab": 0,
                "upscaleEnhance": True,
                "upscaleScale": 2,
                "variants": 1,
                "width": width,
                "safeVenice": True,
                "requestId": request_id,
                "type": "image",
                "modelId": model,
                "modelName": self._client.MODEL_NAMES.get(model, "Venice AI"),
                "modelType": "image",
                "prompt": prompt,
                "seed": seed,
                "messageId": message_id,
                "userId": user_id,
                "simpleMode": False,
                "parentMessageId": None,
                "clientProcessingTime": random.randint(5, 15)
            }

            try:
                # Make the API request
                resp = self._client.session.post(
                    self._client.api_endpoint,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()

                # Venice API returns binary image content directly
                img_bytes = resp.content
                
                # Convert to png or jpeg in memory
                with BytesIO(img_bytes) as input_io:
                    with Image.open(input_io) as im:
                        out_io = BytesIO()
                        if image_format.lower() == "jpeg":
                            im = im.convert("RGB")
                            im.save(out_io, format="JPEG")
                        else:
                            im.save(out_io, format="PNG")
                        img_bytes = out_io.getvalue()

                images.append(img_bytes)

                if response_format == "url":
                    uploaded_url = upload_file_with_retry(img_bytes, image_format)
                    if not uploaded_url:
                        uploaded_url = upload_file_alternative(img_bytes, image_format)
                    if uploaded_url:
                        urls.append(uploaded_url)
                    else:
                        raise RuntimeError(
                            "Failed to upload image to catbox.moe using all available methods"
                        )

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to generate image with Venice AI: {e}")
            except Exception as e:
                raise RuntimeError(f"Error processing Venice AI response: {e}")

        result_data = []
        if response_format == "url":
            for url in urls:
                result_data.append(ImageData(url=url))
        elif response_format == "b64_json":
            import base64
            for img in images:
                b64 = base64.b64encode(img).decode("utf-8")
                result_data.append(ImageData(b64_json=b64))
        else:
            raise ValueError("response_format must be 'url' or 'b64_json'")

        return ImageResponse(created=int(time.time()), data=result_data)


class VeniceAI(TTICompatibleProvider):
    """
    Venice AI provider for text-to-image generation.
    
    This provider interfaces with the Venice AI API at outerface.venice.ai
    to generate images from text prompts using various Stable Diffusion models.
    """
    
    AVAILABLE_MODELS = [
        "stable-diffusion-3.5-rev2",
        "hidream",
        "flux.1-dev-akash",
        "flux.1-dev-uncensored-akash",
        "pony-realism-akash"
    ]
    
    MODEL_NAMES = {
        "stable-diffusion-3.5-rev2": "Venice SD35",
        "hidream": "HiDream",
        "flux.1-dev-akash": "FLUX Standard",
        "flux.1-dev-uncensored-akash": "FLUX Custom",
        "pony-realism-akash": "Pony Realism"
    }

    def __init__(self):
        """
        Initialize the Venice AI provider.
        """
        self.api_endpoint = "https://outerface.venice.ai/api/inference/image"
        self.session = requests.Session()
        self.user_agent = LitAgent().random()
        
        # Set up headers based on the provided request details
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://venice.ai",
            "referer": "https://venice.ai/",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "sec-gpc": "1",
            "user-agent": self.user_agent,
            "x-venice-timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "x-venice-version": "interface@20250726.112947+c7924af"
        }
        
        self.session.headers.update(self.headers)
        self.images = Images(self)

    @property
    def models(self):
        """
        Get available models for the Venice AI provider.
        
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
    client = VeniceAI()
    
    try:
        response = client.images.create(
            model="stable-diffusion-3.5-rev2",
            prompt="red car",
            response_format="url",
            n=1,
            size="1024x1024",
            timeout=60,
        )
        print("Generated image successfully:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")