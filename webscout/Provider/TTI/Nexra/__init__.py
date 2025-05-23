"""
Nexra - Your go-to provider for generating fire images! 🔥

Examples:
    >>> from webscout import NexraImager
    >>> provider = NexraImager()
    >>> images = provider.generate("Cool art")
    >>> paths = provider.save(images)
"""

from .sync_nexra import NexraImager

__all__ = ["NexraImager"]
