# ProxyFox integration for OpenAI-compatible providers
# This module provides a singleton proxy pool for all providers

import proxyfox
import threading

class ProxyFoxPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, size=20, refresh_interval=180, protocol='https', max_speed_ms=300):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance.pool = proxyfox.create_pool(
                        size=size,
                        refresh_interval=refresh_interval,
                        protocol=protocol,
                        max_speed_ms=max_speed_ms  # Aggressively filter for fast proxies
                    )
        return cls._instance

    def get_proxy(self):
        return self.pool.get()

    def all_proxies(self):
        return self.pool.all()

# Global singleton for use in all providers
proxy_pool = ProxyFoxPool()

def get_auto_proxy():
    return proxy_pool.get_proxy()
