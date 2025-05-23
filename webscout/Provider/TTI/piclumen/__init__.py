"""PiclumenImager Provider Package - Your go-to for high-quality AI art! 🎨

Examples:
    >>> from webscout import PiclumenImager
    >>> provider = PiclumenImager()
    >>> images = provider.generate("A cool underwater creature")
    >>> provider.save(images, dir="my_images")
"""

from .sync_piclumen import PiclumenImager

__all__ = ["PiclumenImager"]
