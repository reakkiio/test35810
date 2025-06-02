import requests
from typing import Optional, List, Dict, Any
from webscout.Provider.TTI.utils import (
    ImageData,
    ImageResponse,
)
from webscout.Provider.TTI.base import TTICompatibleProvider, BaseImages
from io import BytesIO
import os
import tempfile
from webscout.litagent import LitAgent
import time
import json

try:
    from PIL import Image
except ImportError:
    Image = None


class Images(BaseImages):
    def __init__(self, client: "PixelMuse"):
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
        **kwargs,
    ) -> ImageResponse:
        """
        image_format: "png" or "jpeg"
        """
        if Image is None:
            raise ImportError("Pillow (PIL) is required for image format conversion.")

        images = []
        urls = []

        def upload_file_with_retry(img_bytes, image_format, max_retries=3):
            """Upload file with retry logic using requests and tempfile"""
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
            """Alternative upload method: save to temp file and upload to 0x0.st"""
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

        for _ in range(n):
            resp = self._client.session.post(
                self._client.api_endpoint,
                json={
                    "prompt": prompt,
                    "model": model,
                    "style": style,
                    "aspect_ratio": aspect_ratio,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "output" in data and len(data["output"]) > 0:
                image_url = data["output"][0]
                img_resp = self._client.session.get(image_url, timeout=timeout)
                img_resp.raise_for_status()
                webp_bytes = img_resp.content

                # Convert webp to png or jpeg in memory
                with BytesIO(webp_bytes) as input_io:
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
                    # Try primary upload method with retries
                    uploaded_url = upload_file_with_retry(img_bytes, image_format)

                    # If primary method fails, try alternative
                    if not uploaded_url:
                        uploaded_url = upload_file_alternative(img_bytes, image_format)

                    if uploaded_url:
                        urls.append(uploaded_url)
                    else:
                        raise RuntimeError(
                            "Failed to upload image to catbox.moe using all available methods"
                        )
            else:
                raise RuntimeError("No image data received from PixelMuse")

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

        from time import time as _time

        return ImageResponse(created=int(_time()), data=result_data)


class PixelMuse(TTICompatibleProvider):
    AVAILABLE_MODELS = ["flux-schnell", "imagen-3-fast", "imagen-3", "recraft-v3"]

    def __init__(self):
        self.api_endpoint = "https://www.pixelmuse.studio/api/predictions"
        self.session = requests.Session()
        self.user_agent = LitAgent().random()
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://www.pixelmuse.studio",
            "referer": "https://www.pixelmuse.studio/",
            "user-agent": self.user_agent,
        }
        self.session.headers.update(self.headers)
        self.images = Images(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS

        return _ModelList()


# Example usage:
if __name__ == "__main__":
    from rich import print

    client = PixelMuse()
    response = client.images.create(
        model="flux-schnell",
        prompt="a white siamese cat",
        response_format="url",
        n=4,
        timeout=30,
    )
    print(response)
