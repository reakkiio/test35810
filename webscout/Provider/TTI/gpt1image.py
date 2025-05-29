import requests
import random
import string
from typing import Optional, List, Dict, Any
from webscout.Provider.TTI.utils import ImageData, ImageResponse
from webscout.Provider.TTI.base import TTICompatibleProvider, BaseImages
from io import BytesIO
import os
import tempfile
from webscout.litagent import LitAgent
import time

class Images(BaseImages):
    def __init__(self, client):
        self._client = client
        self.base_url = "https://gpt1image.exomlapi.com"


    def build_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        agent = LitAgent()
        fp = agent.generate_fingerprint("chrome")
        headers = {
            "Content-Type": "application/json",
            "accept": fp["accept"],
            "accept-language": fp["accept_language"],
            "origin": self.base_url,
            "referer": f"{self.base_url}/",
            "user-agent": fp["user_agent"],
            "sec-ch-ua": fp["sec_ch_ua"],
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-forwarded-for": fp["x-forwarded-for"],
            "x-real-ip": fp["x-real-ip"],
            "x-request-id": fp["x-request-id"],
        }
        if extra:
            headers.update(extra)
        return headers

    def create(self,
        model: str = None,
        prompt: str = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        user: Optional[str] = None,
        style: str = None,
        aspect_ratio: str = None,
        timeout: int = 60,
        image_format: str = "png",
        enhance: bool = True,
        **kwargs
    ) -> ImageResponse:
        if not prompt:
            raise ValueError("Deskripsikan gambar yang ingin kamu buat (gunakan properti 'prompt').")
        body = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "is_enhance": enhance,
            "response_format": response_format
        }
        try:
            resp = requests.post(f"{self.base_url}/v1/images/generations", json=body, headers=self.build_headers(), timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("data") or len(data["data"]) == 0:
                error_info = f", info server: {data.get('error')}" if data.get('error') else ""
                raise RuntimeError(f"Gagal memproses gambar. Data tidak ditemukan{error_info}.")
            result = data["data"]
            result_data = []
            for item in result:
                if response_format == "url":
                    result_data.append(ImageData(url=item.get("url")))
                else:
                    result_data.append(ImageData(b64_json=item.get("b64_json")))
            return ImageResponse(data=result_data)
        except Exception as e:
            raise RuntimeError(f"Terjadi kesalahan: {str(e)}")

class GPT1Image(TTICompatibleProvider):
    AVAILABLE_MODELS = ["gpt1image"]
    def __init__(self):
        self.images = Images(self)
    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

if __name__ == "__main__":
    from rich import print
    client = GPT1Image()
    response = client.images.create(
        prompt="A futuristic robot in a neon city",
        response_format="url",
        n=1,
        timeout=None,
    )
    print(response)
