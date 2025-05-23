"""
HuggingFace Providers - Your go-to solution for generating fire images! 🔥

Examples:
    >>> from webscout import HFimager
    >>> provider = HFimager(api_token="your-hf-token")
    >>> images = provider.generate("Cool art")
    >>> paths = provider.save(images)
"""

from .sync_huggingface import HFimager

__all__ = ["HFimager"]
