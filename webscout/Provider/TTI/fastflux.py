import requests
import os
import tempfile
import time
import base64
from typing import Optional, List, Dict, Any
from webscout.Provider.TTI.utils import (
    ImageData,
    ImageResponse,
    request_with_proxy_fallback,
)
from webscout.Provider.TTI.base import TTICompatibleProvider, BaseImages
from io import BytesIO
from webscout.litagent import LitAgent

try:
    from PIL import Image
except ImportError:
    Image = None


class Images(BaseImages):
    def __init__(self, client):
        self._client = client

    def create(
        self,
        model: str = "flux_1_schnell",
        prompt: str = None,
        n: int = 1,
        size: str = "1_1",
        response_format: str = "url",
        user: Optional[str] = None,
        style: str = "none",
        aspect_ratio: str = "1:1",
        timeout: int = 60,
        image_format: str = "png",
        is_public: bool = False,
        **kwargs,
    ) -> ImageResponse:
        if not prompt:
            raise ValueError("Prompt is required!")
        agent = LitAgent()
        images = []
        urls = []
        api_url = self._client.api_endpoint
        payload = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "isPublic": is_public,
        }
        for _ in range(n):
            resp = request_with_proxy_fallback(
                self._client.session,
                "post",
                api_url,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            if result and "result" in result:
                image_data = result["result"]
                base64_data = image_data.split(",")[1]
                img_bytes = base64.b64decode(base64_data)
                # Convert to png or jpeg in memory if needed
                if Image is not None:
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
                                    files = {
                                        "fileToUpload": (
                                            f"image.{ext}",
                                            f,
                                            f"image/{ext}",
                                        )
                                    }
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
                                        except Exception:
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
                            with tempfile.NamedTemporaryFile(
                                suffix=f".{ext}", delete=False
                            ) as tmp:
                                tmp.write(img_bytes)
                                tmp.flush()
                                tmp_path = tmp.name
                            try:
                                if not os.path.isfile(tmp_path):
                                    return None
                                with open(tmp_path, "rb") as img_file:
                                    files = {"file": img_file}
                                    response = requests.post(
                                        "https://0x0.st", files=files
                                    )
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

                    uploaded_url = upload_file_with_retry(img_bytes, image_format)
                    if not uploaded_url:
                        uploaded_url = upload_file_alternative(img_bytes, image_format)
                    if uploaded_url:
                        urls.append(uploaded_url)
                    else:
                        raise RuntimeError(
                            "Failed to upload image to catbox.moe using all available methods"
                        )
            else:
                raise RuntimeError("No image data received from FastFlux API")
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


class FastFluxAI(TTICompatibleProvider):
    AVAILABLE_MODELS = [
        "flux_1_schnell",
    ]

    def __init__(self, api_key: str = None):
        self.api_endpoint = "https://api.freeflux.ai/v1/images/generate"
        self.session = requests.Session()
        self.user_agent = LitAgent().random()
        self.api_key = api_key
        self.headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://freeflux.ai",
            "referer": "https://freeflux.ai/",
            "user-agent": self.user_agent,
        }
        if self.api_key:
            self.headers["authorization"] = f"Bearer {self.api_key}"
        self.session.headers.update(self.headers)
        self.images = Images(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS

        return _ModelList()


if __name__ == "__main__":
    from rich import print

    client = FastFluxAI()
    response = client.images.create(
        model="flux_1_schnell",
        prompt="A cool cyberpunk city at night",
        response_format="url",
        n=2,
        timeout=30,
    )
    print(response)
