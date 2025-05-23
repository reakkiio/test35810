"""
Artbit Providers - Your go-to solution for generating fire images! 🔥

Examples:
    >>> from webscout import ArtbitImager
    >>> provider = ArtbitImager(logging=True)
    >>> images = provider.generate("Cool art")
    >>> paths = provider.save(images)
"""

from .sync_artbit import ArtbitImager

__all__ = ["ArtbitImager"]
