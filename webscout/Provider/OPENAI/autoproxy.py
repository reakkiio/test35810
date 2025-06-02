# # ProxyFox integration for OpenAI-compatible providers
# # This module provides a singleton proxy pool for all providers

# import proxyfox

# def get_auto_proxy(protocol='https', country=None, max_speed_ms=1000):
#     """
#     Returns a single proxy string (e.g. '11.22.33.44:8080') using proxyfox.
#     You can specify protocol, country, and max_speed_ms for filtering.
#     """
#     kwargs = {'protocol': protocol, 'max_speed_ms': max_speed_ms}
#     if country:
#         kwargs['country'] = country
#     return proxyfox.get_one(**kwargs)

# # Optionally: pool support for advanced usage
# _pool = None

# def get_proxy_pool(size=10, refresh_interval=300, protocol='https', max_speed_ms=1000):
#     global _pool
#     if _pool is None:
#         _pool = proxyfox.create_pool(
#             size=size,
#             refresh_interval=refresh_interval,
#             protocol=protocol,
#             max_speed_ms=max_speed_ms
#         )
#     return _pool

# def get_pool_proxy():
#     pool = get_proxy_pool()
#     return pool.get()

# def get_all_pool_proxies():
#     pool = get_proxy_pool()
#     return pool.all()

# if __name__ == "__main__":
#     print(get_auto_proxy())