from typing import List, Optional
from pydantic import BaseModel, Field
import time

class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None

class ImageResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageData]
