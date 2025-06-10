"""
Auto-proxy module for OpenAI-compatible providers.
This module provides automatic proxy injection for HTTP sessions using a remote proxy list.
"""

import random
import time
from abc import ABCMeta
from typing import Dict, List, Optional, Any
import requests

# Optional imports for different HTTP clients
try:
    import httpx
except ImportError:
    httpx = None

try:
    from curl_cffi.requests import Session as CurlSession
    from curl_cffi.requests import AsyncSession as CurlAsyncSession
except ImportError:
    CurlSession = None
    CurlAsyncSession = None

# Global proxy cache
_proxy_cache = {
    'proxies': [],
    'last_updated': 0,
    'cache_duration': 300  # 5 minutes
}

PROXY_SOURCE_URL = "http://207.180.209.185:5000/ips.txt"


def fetch_proxies() -> List[str]:
    """
    Fetch proxy list from the remote source.

    Returns:
        List[str]: List of proxy URLs in format 'http://user:pass@host:port'
    """
    try:
        response = requests.get(PROXY_SOURCE_URL, timeout=10)
        response.raise_for_status()

        proxies = []
        for line in response.text.strip().split('\n'):
            line = line.strip()
            if line and line.startswith('http://'):
                proxies.append(line)

        return proxies

    except Exception:
        return []


def get_cached_proxies() -> List[str]:
    """
    Get proxies from cache or fetch new ones if cache is expired.

    Returns:
        List[str]: List of proxy URLs
    """
    current_time = time.time()

    # Check if cache is expired or empty
    if (current_time - _proxy_cache['last_updated'] > _proxy_cache['cache_duration'] or
        not _proxy_cache['proxies']):

        new_proxies = fetch_proxies()
        if new_proxies:
            _proxy_cache['proxies'] = new_proxies
            _proxy_cache['last_updated'] = current_time
        else:
            pass

    return _proxy_cache['proxies']


def get_auto_proxy() -> Optional[str]:
    """
    Get a random proxy from the cached proxy list.

    Returns:
        Optional[str]: A proxy URL or None if no proxies available
    """
    proxies = get_cached_proxies()
    if not proxies:
        return None

    proxy = random.choice(proxies)
    return proxy


def get_proxy_dict(proxy_url: Optional[str] = None) -> Dict[str, str]:
    """
    Convert a proxy URL to a dictionary format suitable for requests/httpx.

    Args:
        proxy_url: Proxy URL, if None will get one automatically

    Returns:
        Dict[str, str]: Proxy dictionary with 'http' and 'https' keys
    """
    if proxy_url is None:
        proxy_url = get_auto_proxy()

    if proxy_url is None:
        return {}

    return {
        'http': proxy_url,
        'https': proxy_url
    }


def test_proxy(proxy_url: str, timeout: int = 10) -> bool:
    """
    Test if a proxy is working by making a request to a test URL.

    Args:
        proxy_url: The proxy URL to test
        timeout: Request timeout in seconds

    Returns:
        bool: True if proxy is working, False otherwise
    """
    try:
        test_url = "https://httpbin.org/ip"
        proxies = {'http': proxy_url, 'https': proxy_url}

        response = requests.get(test_url, proxies=proxies, timeout=timeout)
        return response.status_code == 200

    except Exception:
        return False


class ProxyAutoMeta(ABCMeta):
    """
    Metaclass to ensure all OpenAICompatibleProvider subclasses automatically get proxy support.
    This will inject proxies into any requests.Session, httpx.Client, or curl_cffi session attributes found on the instance.

    To disable automatic proxy injection, set disable_auto_proxy=True in the constructor or
    set the class attribute DISABLE_AUTO_PROXY = True.
    """

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        # Check if auto proxy is disabled
        disable_auto_proxy = kwargs.get('disable_auto_proxy', False) or getattr(cls, 'DISABLE_AUTO_PROXY', False)

        # Get proxies from various sources
        proxies = getattr(instance, 'proxies', None) or kwargs.get('proxies', None)

        if proxies is None and not disable_auto_proxy:
            try:
                proxy_url = get_auto_proxy()
                if proxy_url:
                    proxies = get_proxy_dict(proxy_url)
                else:
                    proxies = {}
            except Exception:
                proxies = {}
        elif proxies is None:
            proxies = {}

        instance.proxies = proxies

        # Patch existing sessions if we have valid proxies
        if proxies:
            _patch_instance_sessions(instance, proxies)

        # Provide helper methods for creating proxied sessions
        _add_proxy_helpers(instance, proxies)

        return instance


def _patch_instance_sessions(instance: Any, proxies: Dict[str, str]) -> None:
    """
    Patch existing session objects on the instance with proxy configuration.

    Args:
        instance: The class instance to patch
        proxies: Proxy dictionary to apply
    """
    for attr_name in dir(instance):
        if attr_name.startswith('_'):
            continue

        try:
            attr_obj = getattr(instance, attr_name)

            # Patch requests.Session objects
            if isinstance(attr_obj, requests.Session):
                attr_obj.proxies.update(proxies)

            # Patch httpx.Client objects
            elif httpx and isinstance(attr_obj, httpx.Client):
                try:
                    # httpx uses different proxy format
                    attr_obj._proxies = proxies
                except Exception:
                    pass

            # Patch curl_cffi Session objects
            elif CurlSession and isinstance(attr_obj, CurlSession):
                try:
                    attr_obj.proxies.update(proxies)
                except Exception:
                    pass

            # Patch curl_cffi AsyncSession objects
            elif CurlAsyncSession and isinstance(attr_obj, CurlAsyncSession):
                try:
                    attr_obj.proxies.update(proxies)
                except Exception:
                    pass

        except Exception:
            continue


def _add_proxy_helpers(instance: Any, proxies: Dict[str, str]) -> None:
    """
    Add helper methods to the instance for creating proxied sessions.

    Args:
        instance: The class instance to add methods to
        proxies: Proxy dictionary to use in helper methods
    """

    def get_proxied_session():
        """Get a requests.Session with proxies configured"""
        session = requests.Session()
        session.proxies.update(proxies)
        return session

    def get_proxied_httpx_client(**kwargs):
        """Get an httpx.Client with proxies configured"""
        if httpx:
            return httpx.Client(proxies=proxies, **kwargs)
        else:
            raise ImportError("httpx is not installed")

    def get_proxied_curl_session(impersonate="chrome120", **kwargs):
        """Get a curl_cffi Session with proxies configured"""
        if CurlSession:
            return CurlSession(proxies=proxies, impersonate=impersonate, **kwargs)
        else:
            raise ImportError("curl_cffi is not installed")

    def get_proxied_curl_async_session(impersonate="chrome120", **kwargs):
        """Get a curl_cffi AsyncSession with proxies configured"""
        if CurlAsyncSession:
            return CurlAsyncSession(proxies=proxies, impersonate=impersonate, **kwargs)
        else:
            raise ImportError("curl_cffi is not installed")

    # Add methods to instance
    instance.get_proxied_session = get_proxied_session
    instance.get_proxied_httpx_client = get_proxied_httpx_client
    instance.get_proxied_curl_session = get_proxied_curl_session
    instance.get_proxied_curl_async_session = get_proxied_curl_async_session


def get_working_proxy(max_attempts: int = 5, timeout: int = 10) -> Optional[str]:
    """
    Get a working proxy by testing multiple proxies from the list.

    Args:
        max_attempts: Maximum number of proxies to test
        timeout: Timeout for each proxy test

    Returns:
        Optional[str]: A working proxy URL or None if none found
    """
    proxies = get_cached_proxies()
    if not proxies:
        return None

    # Shuffle to avoid always testing the same proxies first
    test_proxies = random.sample(proxies, min(max_attempts, len(proxies)))

    for proxy in test_proxies:
        if test_proxy(proxy, timeout):
            return proxy

    return None


def refresh_proxy_cache() -> int:
    """
    Force refresh the proxy cache.

    Returns:
        int: Number of proxies loaded
    """
    global _proxy_cache
    _proxy_cache['last_updated'] = 0  # Force refresh
    proxies = get_cached_proxies()
    return len(proxies)


def get_proxy_stats() -> Dict[str, Any]:
    """
    Get statistics about the proxy cache.

    Returns:
        Dict[str, Any]: Statistics including count, last update time, etc.
    """
    return {
        'proxy_count': len(_proxy_cache['proxies']),
        'last_updated': _proxy_cache['last_updated'],
        'cache_duration': _proxy_cache['cache_duration'],
        'cache_age_seconds': time.time() - _proxy_cache['last_updated'],
        'source_url': PROXY_SOURCE_URL
    }


def set_proxy_cache_duration(duration: int) -> None:
    """
    Set the proxy cache duration.

    Args:
        duration: Cache duration in seconds
    """
    global _proxy_cache
    _proxy_cache['cache_duration'] = duration