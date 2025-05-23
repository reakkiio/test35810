"""PollinationsAI Provider Package - Your go-to for text-to-image generation! 🎨

Examples:
    >>> from webscout import PollinationsAI
    >>> provider = PollinationsAI()
    >>> images = provider.generate("A cool cyberpunk city")
    >>> provider.save(images, dir="my_images")
"""

from .sync_pollinations import PollinationsAI

__all__ = ["PollinationsAI"]
