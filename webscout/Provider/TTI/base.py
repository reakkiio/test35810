from abc import ABC, ABCMeta, abstractmethod
from typing import List, Dict, Optional, Any, Union, Generator
from .utils import ImageResponse
import random
import requests
try:
    import httpx
except ImportError:
    httpx = None

try:
    from curl_cffi.requests import Session as CurlSession, AsyncSession as CurlAsyncSession
except ImportError:
    CurlSession = None
    CurlAsyncSession = None

from webscout.Provider.OPENAI.autoproxy import get_auto_proxy

# Global proxy manager for direct requests.Session monkey patching
class _GlobalProxyManager:
    """
    Global singleton to manage proxy configuration for all requests.Session instances.
    This allows direct monkey patching of requests.Session.__init__ to automatically
    apply proxies without needing extra code in providers.
    """
    _instance = None
    _proxies = {}
    _original_session_init = None
    _patched = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_proxies(self, proxies: Dict[str, str]):
        """Set the global proxy configuration"""
        self._proxies = proxies or {}
        self._ensure_patched()

    def get_proxies(self) -> Dict[str, str]:
        """Get the current global proxy configuration"""
        return self._proxies.copy()

    def _ensure_patched(self):
        """Ensure requests.Session is monkey patched to use global proxies"""
        if not self._patched:
            # Store original __init__ method
            self._original_session_init = requests.Session.__init__

            # Create patched __init__ method
            def patched_session_init(session_self, *args, **kwargs):
                # Call original __init__
                self._original_session_init(session_self, *args, **kwargs)
                # Apply global proxies if available
                if self._proxies:
                    session_self.proxies.update(self._proxies)

            # Apply the monkey patch
            requests.Session.__init__ = patched_session_init
            self._patched = True

    def unpatch(self):
        """Remove the monkey patch (for testing/cleanup)"""
        if self._patched and self._original_session_init:
            requests.Session.__init__ = self._original_session_init
            self._patched = False

# Global instance
_proxy_manager = _GlobalProxyManager()

class BaseImages(ABC):
    @abstractmethod
    def create(
        self,
        *,
        model: str,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        user: Optional[str] = None,
        style: str = "none",
        aspect_ratio: str = "1:1",
        timeout: int = None,
        image_format: str = "png",
        seed: Optional[int] = None,
        **kwargs
    ) -> ImageResponse:
        """
        Abstract method to create images from a prompt.

        Args:
            model: The model to use for image generation.
            prompt: The prompt for the image.
            n: Number of images to generate.
            size: Image size.
            response_format: "url" or "b64_json".
            user: Optional user identifier.
            style: Optional style.
            aspect_ratio: Optional aspect ratio.
            timeout: Request timeout in seconds.
            image_format: "png" or "jpeg" for output format.
            seed: Optional random seed for reproducibility.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ImageResponse: The generated images.
        """
        raise NotImplementedError

class ProxyAutoMeta(ABCMeta):
    """
    Simplified metaclass that uses global proxy manager to automatically configure
    proxies for all requests.Session instances. Much cleaner than the old approach!
    """
    def __call__(cls, *args, **kwargs):
        # Check if auto proxy is disabled
        disable_auto_proxy = kwargs.get('disable_auto_proxy', False) or getattr(cls, 'DISABLE_AUTO_PROXY', False)

        # Get proxies from kwargs
        proxies = kwargs.get('proxies', None)

        # Auto-fetch proxies if not provided and not disabled
        if proxies is None and not disable_auto_proxy:
            try:
                proxies = {"http": get_auto_proxy(), "https": get_auto_proxy()}
            except Exception as e:
                print(f"Failed to fetch auto-proxy: {e}")
                proxies = {}
        elif proxies is None:
            proxies = {}

        # Set global proxies BEFORE creating the instance so that any sessions created
        # during __init__ will automatically get the proxies!
        _proxy_manager.set_proxies(proxies)

        # Now create the instance - any requests.Session() created will have proxies
        instance = super().__call__(*args, **kwargs)

        # Store proxies on instance for reference
        instance.proxies = proxies

        # Add a simple helper method for backward compatibility
        def get_proxied_session():
            # Since we monkey patched requests.Session, this will automatically have proxies!
            return requests.Session()
        instance.get_proxied_session = get_proxied_session

        return instance

class TTICompatibleProvider(ABC, metaclass=ProxyAutoMeta):
    """
    Abstract Base Class for TTI providers mimicking the OpenAI Python client structure.
    Requires a nested 'images.create' structure.
    All subclasses automatically get proxy support via ProxyAutoMeta.

    Available proxy helpers:
    - self.get_proxied_session() - returns a requests.Session with proxies
    - self.get_proxied_curl_session() - returns a curl_cffi.Session with proxies
    - self.get_proxied_curl_async_session() - returns a curl_cffi.AsyncSession with proxies
    """
    images: BaseImages

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass

    @property
    @abstractmethod
    def models(self):
        """
        Property that returns an object with a .list() method returning available models.
        Subclasses must implement this property.
        """
        pass
