import time
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from webscout.Provider.OPENAI.autoproxy import get_auto_proxy


def request_with_proxy_fallback(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: Optional[int] = None,
    retries: int = 3,
    **kwargs,
) -> requests.Response:
    """Perform a request using rotating proxies.

    The request first uses the session's current proxy configuration. If the
    request fails due to connectivity issues, a new proxy is fetched using
    :func:`get_auto_proxy` and the request is retried. This continues up to
    ``retries`` times before raising an exception.
    """

    if retries < 1:
        retries = 1

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ProxyError,
            requests.exceptions.Timeout,
        ) as exc:
            last_error = exc
            try:
                proxy = get_auto_proxy()
                session.proxies.update({"http": proxy, "https": proxy})
            except Exception as fetch_err:  # pragma: no cover - network dependent
                last_error = fetch_err
        except Exception:
            # Other errors should not trigger proxy retry
            raise

    raise RuntimeError(f"All proxy attempts failed: {last_error}")


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageData]
