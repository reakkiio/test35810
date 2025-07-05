import requests
import base64
from typing import Optional, List, Dict, Any
from webscout.Provider.TTI.utils import (
    ImageData,
    ImageResponse
)
from webscout.Provider.TTI.base import TTICompatibleProvider, BaseImages
from webscout.litagent import LitAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import tempfile
import time
import json
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    Image = None


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
        response_format: str = "b64_json",
        user: Optional[str] = None,
        style: str = None,
        aspect_ratio: str = None,
        timeout: int = 60,
        image_format: str = "png",
        **kwargs,
    ) -> ImageResponse:
        if not prompt:
            raise ValueError(
                "Describe the image you want to create (use the 'prompt' property)."
            )
        # Only one image is supported by MonoChat API, but keep n for compatibility
        body = {
            "prompt": prompt,
            "model": model
        }
        session = self._client.session
        headers = self._client.headers
        images = []
        urls = []

        def upload_file_with_retry(img_bytes, image_format, max_retries=3):
            ext = "jpg" if image_format.lower() == "jpeg" else "png"
            for attempt in range(max_retries):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                        tmp.write(img_bytes)
                        tmp.flush()
                        tmp_path = tmp.name
                    with open(tmp_path, "rb") as f:
                        files = {"fileToUpload": (f"image.{ext}", f, f"image/{ext}")}
                        data = {"reqtype": "fileupload", "json": "true"}
                        headers = {"User-Agent": LitAgent().random()}
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

        try:
            resp = session.post(
                f"{self._client.api_endpoint}/image",
                json=body,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("image"):
                raise RuntimeError("Failed to process image. No image data found.")
            # Always decode the base64 image
            image_bytes = base64.b64decode(data.get("image"))
            if response_format == "b64_json":
                result_data = [ImageData(b64_json=data.get("image"))]
            elif response_format == "url":
                if Image is None:
                    raise ImportError("Pillow (PIL) is required for image format conversion.")
                # Convert to png or jpeg in memory
                with BytesIO(image_bytes) as input_io:
                    with Image.open(input_io) as im:
                        out_io = BytesIO()
                        if image_format.lower() == "jpeg":
                            im = im.convert("RGB")
                            im.save(out_io, format="JPEG")
                        else:
                            im.save(out_io, format="PNG")
                        img_bytes = out_io.getvalue()
                # Try primary upload method with retries
                uploaded_url = upload_file_with_retry(img_bytes, image_format)
                # If primary method fails, try alternative
                if not uploaded_url:
                    uploaded_url = upload_file_alternative(img_bytes, image_format)
                if uploaded_url:
                    result_data = [ImageData(url=uploaded_url)]
                else:
                    raise RuntimeError("Failed to upload image to catbox.moe using all available methods")
            else:
                raise ValueError("response_format must be 'url' or 'b64_json'")
            from time import time as _time
            return ImageResponse(created=int(_time()), data=result_data)
        except Exception as e:
            raise RuntimeError(f"An error occurred: {str(e)}")


class MonoChatAI(TTICompatibleProvider):
    AVAILABLE_MODELS = ["nextlm-image-1", "gpt-image-1", "dall-e-3", "dall-e-2"]

    def __init__(self):
        self.api_endpoint = "https://gg.is-a-furry.dev/api"
        self.session = requests.Session()
        self._setup_session_with_retries()
        self.user_agent = LitAgent().random()
        self.headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": "https://gg.is-a-furry.dev",
            "referer": "https://gg.is-a-furry.dev/",
            "user-agent": self.user_agent,
        }
        self.session.headers.update(self.headers)
        self.images = Images(self)

    def _setup_session_with_retries(self):
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()


if __name__ == "__main__":
    from rich import print
    client = MonoChatAI()
    response = client.images.create(
        model="dall-e-3",
        prompt="A red car on a sunny day",
        response_format="url",
        timeout=60000,
    )
    print(response)