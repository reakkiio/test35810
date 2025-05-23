"""
AiForce - Your go-to provider for generating fire images! 🔥

Examples:
    >>> from webscout import AiForceimager
    >>> provider = AiForceimager()
    >>> images = provider.generate("Cool art")
    >>> paths = provider.save(images)
"""

from .sync_aiforce import AiForceimager

__all__ = ["AiForceimager"]
