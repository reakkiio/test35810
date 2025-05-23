"""ImgSys Provider Package - Generate images from multiple providers! 🎨

Examples:
    >>> from webscout import ImgSys
    >>> provider = ImgSys()
    >>> images = provider.generate("A cool cyberpunk city")
    >>> provider.save(images, dir="my_images")
"""

from .sync_imgsys import ImgSys

__all__ = ["ImgSys"]
