from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Optional

import requests

from .utils import ImageResponse

try:
    import httpx
except ImportError:
    httpx = None

try:
    from curl_cffi.requests import AsyncSession as CurlAsyncSession
    from curl_cffi.requests import Session as CurlSession
except ImportError:
    CurlSession = None
    CurlAsyncSession = None

from webscout.Provider.OPENAI.autoproxy import get_auto_proxy


# Global proxy manager for direct requests.Session monkey patching
class _GlobalProxyManager:
    """Singleton to transparently apply proxies to HTTP sessions."""

    _instance = None
    _proxies = {}

    _original_session_init = None
    _original_session_request = None

    _original_curl_session_init = None
    _original_curl_session_request = None

    _original_curl_async_session_init = None
    _original_curl_async_session_request = None

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
        """Patch popular HTTP clients so proxies are applied everywhere."""
        if self._patched:
            return

        # ---- requests.Session patches ----
        self._original_session_init = requests.Session.__init__

        def patched_session_init(session_self, *args, **kwargs):
            self._original_session_init(session_self, *args, **kwargs)
            if self._proxies:
                session_self.proxies.update(self._proxies)

        requests.Session.__init__ = patched_session_init

        self._original_session_request = requests.Session.request

        def patched_session_request(session_self, method, url, *a, **kw):
            if self._proxies and 'proxies' not in kw:
                kw['proxies'] = self._proxies
            return self._original_session_request(session_self, method, url, *a, **kw)

        requests.Session.request = patched_session_request

        # ---- curl_cffi Session patches ----
        if CurlSession:
            self._original_curl_session_init = CurlSession.__init__

            def patched_curl_init(session_self, *args, **kwargs):
                if self._proxies and 'proxies' not in kwargs:
                    kwargs['proxies'] = self._proxies
                self._original_curl_session_init(session_self, *args, **kwargs)

            CurlSession.__init__ = patched_curl_init

            if hasattr(CurlSession, 'request'):
                self._original_curl_session_request = CurlSession.request

                def patched_curl_request(session_self, method, url, *a, **kw):
                    if self._proxies and 'proxies' not in kw:
                        kw['proxies'] = self._proxies
                    return self._original_curl_session_request(session_self, method, url, *a, **kw)

                CurlSession.request = patched_curl_request

        if CurlAsyncSession:
            self._original_curl_async_session_init = CurlAsyncSession.__init__

            def patched_curl_async_init(session_self, *args, **kwargs):
                if self._proxies and 'proxies' not in kwargs:
                    kwargs['proxies'] = self._proxies
                self._original_curl_async_session_init(session_self, *args, **kwargs)

            CurlAsyncSession.__init__ = patched_curl_async_init

            if hasattr(CurlAsyncSession, 'request'):
                self._original_curl_async_session_request = CurlAsyncSession.request

                async def patched_curl_async_request(session_self, method, url, *a, **kw):
                    if self._proxies and 'proxies' not in kw:
                        kw['proxies'] = self._proxies
                    return await self._original_curl_async_session_request(session_self, method, url, *a, **kw)

                CurlAsyncSession.request = patched_curl_async_request

        self._patched = True

    def unpatch(self):
        """Remove all monkey patches (primarily for tests)."""
        if not self._patched:
            return

        if self._original_session_init:
            requests.Session.__init__ = self._original_session_init
        if self._original_session_request:
            requests.Session.request = self._original_session_request

        if CurlSession and self._original_curl_session_init:
            CurlSession.__init__ = self._original_curl_session_init
        if CurlSession and self._original_curl_session_request:
            CurlSession.request = self._original_curl_session_request

        if CurlAsyncSession and self._original_curl_async_session_init:
            CurlAsyncSession.__init__ = self._original_curl_async_session_init
        if CurlAsyncSession and self._original_curl_async_session_request:
            CurlAsyncSession.request = self._original_curl_async_session_request

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
    """Metaclass providing seamless proxy injection for providers."""

    def __call__(cls, *args, **kwargs):
        # Determine if automatic proxying should be disabled
        disable_auto_proxy = kwargs.get('disable_auto_proxy', False) or getattr(cls, 'DISABLE_AUTO_PROXY', False)

        # Proxies may be supplied explicitly
        proxies = kwargs.get('proxies', None)

        # Otherwise try to fetch one automatically
        if proxies is None and not disable_auto_proxy:
            try:
                proxies = {"http": get_auto_proxy(), "https": get_auto_proxy()}
            except Exception as e:
                print(f"Failed to fetch auto-proxy: {e}")
                proxies = {}
        elif proxies is None:
            proxies = {}

        # Patch global sessions before instantiation so any sessions created in __init__ get proxies
        _proxy_manager.set_proxies(proxies)

        instance = super().__call__(*args, **kwargs)

        # Expose proxies on the instance
        instance.proxies = proxies

        # If proxies are set, patch any existing session-like attributes
        if proxies:
            for attr in dir(instance):
                obj = getattr(instance, attr)
                if isinstance(obj, requests.Session):
                    obj.proxies.update(proxies)
                if CurlSession and isinstance(obj, CurlSession):
                    try:
                        obj.proxies.update(proxies)
                    except (ValueError, KeyError, AttributeError):
                        print("Failed to update proxies for CurlSession due to an expected error.")
                if CurlAsyncSession and isinstance(obj, CurlAsyncSession):
                    try:
                        obj.proxies.update(proxies)
                    except (ValueError, KeyError, AttributeError):
                        print("Failed to update proxies for CurlAsyncSession due to an expected error.")

        # Helper for backward compatibility
        def get_proxied_session():
            s = requests.Session()
            s.proxies.update(proxies)
            return s

        instance.get_proxied_session = get_proxied_session

        def get_proxied_curl_session(impersonate="chrome120", **kw):
            if CurlSession:
                return CurlSession(proxies=proxies, impersonate=impersonate, **kw)
            raise ImportError("curl_cffi is not installed")

        instance.get_proxied_curl_session = get_proxied_curl_session

        def get_proxied_curl_async_session(impersonate="chrome120", **kw):
            if CurlAsyncSession:
                return CurlAsyncSession(proxies=proxies, impersonate=impersonate, **kw)
            raise ImportError("curl_cffi is not installed")

        instance.get_proxied_curl_async_session = get_proxied_curl_async_session

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
