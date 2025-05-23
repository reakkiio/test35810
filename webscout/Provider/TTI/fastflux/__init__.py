"""
FastFlux Image Generator - Your go-to provider for generating fire images! 🔥

Examples:
    >>> from webscout import FastFluxImager
    >>> provider = FastFluxImager()
    >>> images = provider.generate("Cool art")
    >>> paths = provider.save(images)
"""

from .sync_fastflux import FastFluxImager

__all__ = ["FastFluxImager"]
