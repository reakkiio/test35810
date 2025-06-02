# ProxyFox integration for OpenAI-compatible providers
# This module provides a singleton proxy pool for all providers

import proxyfox
import requests


def get_auto_proxy(
    protocol: str = "https",
    country: str | None = None,
    max_speed_ms: int = 1000,
    *,
    test_url: str = "https://www.google.com",
    timeout: int = 5,
    attempts: int = 5,
) -> str:
    """Return a verified working proxy string using ProxyFox.

    The function fetches a proxy from ProxyFox and optionally verifies it
    by making a request to ``test_url``.  If verification fails, a new proxy
    is fetched until ``attempts`` is exhausted.
    """

    kwargs = {"protocol": protocol, "max_speed_ms": max_speed_ms}
    if country:
        kwargs["country"] = country

    last_error: Exception | None = None
    for _ in range(max(attempts, 1)):
        proxy = proxyfox.get_one(**kwargs)
        if not test_url:
            return proxy

        try:
            requests.get(
                test_url,
                proxies={"http": proxy, "https": proxy},
                timeout=timeout,
            )
            return proxy
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc

    raise RuntimeError(f"Unable to obtain working proxy: {last_error}")


# Optionally: pool support for advanced usage
_pool = None


def get_proxy_pool(size=10, refresh_interval=300, protocol="https", max_speed_ms=1000):
    global _pool
    if _pool is None:
        _pool = proxyfox.create_pool(
            size=size,
            refresh_interval=refresh_interval,
            protocol=protocol,
            max_speed_ms=max_speed_ms,
        )
    return _pool


def get_pool_proxy():
    pool = get_proxy_pool()
    return pool.get()


def get_all_pool_proxies():
    pool = get_proxy_pool()
    return pool.all()


if __name__ == "__main__":
    print(get_auto_proxy())
