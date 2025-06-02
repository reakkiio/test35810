"""AIArtaImager TTI-Compatible Provider - Generate stunning AI art with AI Arta! 🎨

Examples:
    >>> from webscout.Provider.TTI.aiarta import AIArta
    >>> client = AIArta()
    >>> response = client.images.create(
    ...     model="flux",
    ...     prompt="A cool cyberpunk city at night",
    ...     n=1
    ... )
    >>> print(response)
"""

import requests
from typing import Optional, List, Dict, Any
from webscout.Provider.TTI.utils import (
    ImageData,
    ImageResponse,
    request_with_proxy_fallback,
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
    def __init__(self, client: "AIArta"):
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

        for _ in range(n):
            # Step 1: Get Authentication Token
            auth_data = self._client.read_and_refresh_token()
            gen_headers = {
                "Authorization": auth_data.get("idToken"),
            }
            # Remove content-type header for form data
            if "content-type" in self._client.session.headers:
                del self._client.session.headers["content-type"]
            # get_model now returns the proper style name from model_aliases
            style_value = self._client.get_model(model)
            image_payload = {
                "prompt": str(prompt),
                "negative_prompt": str(
                    kwargs.get("negative_prompt", "blurry, deformed hands, ugly")
                ),
                "style": str(style_value),
                "images_num": str(1),  # Generate one image at a time in the loop
                "cfg_scale": str(kwargs.get("guidance_scale", 7)),
                "steps": str(kwargs.get("num_inference_steps", 30)),
                "aspect_ratio": str(aspect_ratio),
            }
            # Step 2: Generate Image (send as form data, not JSON)
            image_response = request_with_proxy_fallback(
                self._client.session,
                "post",
                self._client.image_generation_url,
                data=image_payload,  # Use form data instead of JSON
                headers=gen_headers,
                timeout=timeout,
            )
            if image_response.status_code != 200:
                raise RuntimeError(
                    f"AIArta API error {image_response.status_code}: {image_response.text}\nPayload: {image_payload}"
                )
            image_data = image_response.json()
            record_id = image_data.get("record_id")
            if not record_id:
                raise RuntimeError(f"Failed to initiate image generation: {image_data}")
            # Step 3: Check Generation Status
            status_url = self._client.status_check_url.format(record_id=record_id)
            while True:
                status_response = request_with_proxy_fallback(
                    self._client.session,
                    "get",
                    status_url,
                    headers=gen_headers,
                    timeout=timeout,
                )
                status_data = status_response.json()
                status = status_data.get("status")
                if status == "DONE":
                    image_urls = [
                        image["url"] for image in status_data.get("response", [])
                    ]
                    if not image_urls:
                        raise RuntimeError("No image URLs returned from AIArta")
                    img_resp = request_with_proxy_fallback(
                        self._client.session,
                        "get",
                        image_urls[0],
                        timeout=timeout,
                    )
                    img_resp.raise_for_status()
                    img_bytes = img_resp.content
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
                            uploaded_url = upload_file_alternative(
                                img_bytes, image_format
                            )
                        if uploaded_url:
                            urls.append(uploaded_url)
                        else:
                            raise RuntimeError(
                                "Failed to upload image to catbox.moe using all available methods"
                            )
                    break
                elif status in ("IN_QUEUE", "IN_PROGRESS"):
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Image generation failed with status: {status}")

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


class AIArta(TTICompatibleProvider):
    # Model aliases mapping from lowercase keys to proper API style names
    model_aliases = {
        "flux": "Flux",
        "medieval": "Medieval",
        "vincent_van_gogh": "Vincent Van Gogh",
        "f_dev": "F Dev",
        "low_poly": "Low Poly",
        "dreamshaper_xl": "Dreamshaper-xl",
        "anima_pencil_xl": "Anima-pencil-xl",
        "biomech": "Biomech",
        "trash_polka": "Trash Polka",
        "no_style": "No Style",
        "cheyenne_xl": "Cheyenne-xl",
        "chicano": "Chicano",
        "embroidery_tattoo": "Embroidery tattoo",
        "red_and_black": "Red and Black",
        "fantasy_art": "Fantasy Art",
        "watercolor": "Watercolor",
        "dotwork": "Dotwork",
        "old_school_colored": "Old school colored",
        "realistic_tattoo": "Realistic tattoo",
        "japanese_2": "Japanese_2",
        "realistic_stock_xl": "Realistic-stock-xl",
        "f_pro": "F Pro",
        "revanimated": "RevAnimated",
        "katayama_mix_xl": "Katayama-mix-xl",
        "sdxl_l": "SDXL L",
        "cor_epica_xl": "Cor-epica-xl",
        "anime_tattoo": "Anime tattoo",
        "new_school": "New School",
        "death_metal": "Death metal",
        "old_school": "Old School",
        "juggernaut_xl": "Juggernaut-xl",
        "photographic": "Photographic",
        "sdxl_1_0": "SDXL 1.0",
        "graffiti": "Graffiti",
        "mini_tattoo": "Mini tattoo",
        "surrealism": "Surrealism",
        "neo_traditional": "Neo-traditional",
        "on_limbs_black": "On limbs black",
        "yamers_realistic_xl": "Yamers-realistic-xl",
        "pony_xl": "Pony-xl",
        "playground_xl": "Playground-xl",
        "anything_xl": "Anything-xl",
        "flame_design": "Flame design",
        "kawaii": "Kawaii",
        "cinematic_art": "Cinematic Art",
        "professional": "Professional",
        "black_ink": "Black Ink",
    }

    AVAILABLE_MODELS = list(model_aliases.keys())
    default_model = "Flux"
    default_image_model = default_model

    def __init__(self):
        self.image_generation_url = "https://img-gen-prod.ai-arta.com/api/v1/text2image"
        self.status_check_url = (
            "https://img-gen-prod.ai-arta.com/api/v1/text2image/{record_id}/status"
        )
        self.auth_url = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key=AIzaSyB3-71wG0fIt0shj0ee4fvx1shcjJHGrrQ"
        self.token_refresh_url = "https://securetoken.googleapis.com/v1/token?key=AIzaSyB3-71wG0fIt0shj0ee4fvx1shcjJHGrrQ"
        self.session = requests.Session()
        self.user_agent = LitAgent().random()
        self.headers = {
            "accept": "application/json",
            "accept-language": "en-US,en;q=0.9",
            "origin": "https://img-gen-prod.ai-arta.com",
            "referer": "https://img-gen-prod.ai-arta.com/",
            "user-agent": self.user_agent,
        }
        self.session.headers.update(self.headers)
        self.images = Images(self)

    def get_auth_file(self) -> str:
        path = os.path.join(os.path.expanduser("~"), ".ai_arta_cookies")
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"auth_{self.__class__.__name__}.json"
        return os.path.join(path, filename)

    def create_token(self, path: str) -> Dict[str, Any]:
        auth_payload = {"clientType": "CLIENT_TYPE_ANDROID"}
        proxies = self.session.proxies if self.session.proxies else None
        auth_response = request_with_proxy_fallback(
            self.session,
            "post",
            self.auth_url,
            json=auth_payload,
            timeout=60,
            proxies=proxies,
        )
        auth_data = auth_response.json()
        auth_token = auth_data.get("idToken")
        if not auth_token:
            raise Exception("Failed to obtain authentication token.")
        with open(path, "w") as f:
            json.dump(auth_data, f)
        return auth_data

    def refresh_token(self, refresh_token: str) -> tuple[str, str]:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        response = request_with_proxy_fallback(
            self.session,
            "post",
            self.token_refresh_url,
            data=payload,
            timeout=60,
        )
        response_data = response.json()
        return response_data.get("id_token"), response_data.get("refresh_token")

    def read_and_refresh_token(self) -> Dict[str, Any]:
        path = self.get_auth_file()
        if os.path.isfile(path):
            with open(path, "r") as f:
                auth_data = json.load(f)
            diff = time.time() - os.path.getmtime(path)
            expires_in = int(auth_data.get("expiresIn", 3600))
            if diff < expires_in:
                if diff > expires_in / 2:
                    auth_data["idToken"], auth_data["refreshToken"] = (
                        self.refresh_token(auth_data.get("refreshToken"))
                    )
                    with open(path, "w") as f:
                        json.dump(auth_data, f)
                return auth_data
        return self.create_token(path)

    def get_model(self, model_name: str) -> str:
        # Convert to lowercase for lookup
        model_key = model_name.lower()
        # Return the proper style name from model_aliases, or the original if not found
        return self.model_aliases.get(model_key, model_name)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS

        return _ModelList()


# Example usage:
if __name__ == "__main__":
    from rich import print

    client = AIArta()
    response = client.images.create(
        model="flux",
        prompt="a white siamese cat",
        response_format="url",
        n=2,
        timeout=30,
    )
    print(response)
