# WebScout Auto-Proxy System

The WebScout Auto-Proxy system provides automatic proxy injection for all OpenAI-compatible providers. This system fetches proxies from a remote source and automatically configures them for HTTP sessions.

## Features

- **Automatic Proxy Injection**: All OpenAI-compatible providers automatically get proxy support
- **Multiple HTTP Client Support**: Works with `requests`, `httpx`, and `curl_cffi`
- **Proxy Pool Management**: Automatically fetches and caches proxies from remote source
- **Working Proxy Detection**: Tests proxies to find working ones
- **Easy Disable Option**: Can be disabled per provider instance or globally

## How It Works

The system uses a metaclass (`ProxyAutoMeta`) that automatically:

1. Fetches proxies from `http://207.180.209.185:5000/ips.txt`
2. Caches proxies for 5 minutes to avoid excessive requests
3. Randomly selects a proxy for each provider instance
4. Patches existing HTTP session objects with proxy configuration
5. Provides helper methods for creating proxied sessions

## Usage

### Automatic Usage (Default)

All OpenAI-compatible providers automatically get proxy support:

```python
from webscout.Provider.OPENAI.yep import YEPCHAT

# Proxy is automatically configured
client = YEPCHAT()

# All requests will use the configured proxy
response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Disabling Auto-Proxy

You can disable automatic proxy injection:

```python
# Disable for a specific instance
client = YEPCHAT(disable_auto_proxy=True)

# Or set a class attribute to disable for all instances
class MyProvider(OpenAICompatibleProvider):
    DISABLE_AUTO_PROXY = True
```

### Manual Proxy Configuration

You can also provide your own proxies:

```python
custom_proxies = {
    'http': 'http://user:pass@proxy.example.com:8080',
    'https': 'http://user:pass@proxy.example.com:8080'
}

client = YEPCHAT(proxies=custom_proxies)
```

### Using Helper Methods

Each provider instance gets helper methods for creating proxied sessions:

```python
client = YEPCHAT()

# Get a requests.Session with proxies configured
session = client.get_proxied_session()

# Get a curl_cffi Session with proxies configured
curl_session = client.get_proxied_curl_session(impersonate="chrome120")

# Get an httpx.Client with proxies configured (if httpx is installed)
httpx_client = client.get_proxied_httpx_client()
```

## Direct API Usage

You can also use the proxy functions directly:

```python
from webscout.Provider.OPENAI.autoproxy import (
    get_auto_proxy,
    get_proxy_dict,
    get_working_proxy,
    test_proxy,
    get_proxy_stats
)

# Get a random proxy
proxy = get_auto_proxy()

# Get proxy in dictionary format
proxy_dict = get_proxy_dict()

# Find a working proxy (tests multiple proxies)
working_proxy = get_working_proxy(max_attempts=5)

# Test if a proxy is working
is_working = test_proxy(proxy)

# Get proxy cache statistics
stats = get_proxy_stats()
```

## Proxy Format

The system expects proxies in the format:
```
http://username:password@host:port
```

Example:
```
http://fnXlN8NP6StpxZkxmNLyOt2MaVLQunpGC7K96j7R0KbnE5sU_2RdYRxaoy7P2yfqrD7Y8UFexv8kpTyK0LwkDQ==:fnXlN8NP6StpxZkxmNLyOt2MaVLQunpGC7K96j7R0KbnE5sU_2RdYRxaoy7P2yfqrD7Y8UFexv8kpTyK0LwkDQ==@190.103.177.163:80
```

## Configuration

### Cache Duration

You can adjust the proxy cache duration:

```python
from webscout.Provider.OPENAI.autoproxy import set_proxy_cache_duration

# Set cache to 10 minutes
set_proxy_cache_duration(600)
```

### Force Refresh

You can force refresh the proxy cache:

```python
from webscout.Provider.OPENAI.autoproxy import refresh_proxy_cache

# Force refresh and get number of proxies loaded
count = refresh_proxy_cache()
print(f"Loaded {count} proxies")
```

## Error Handling

The system gracefully handles errors:

- If proxy fetching fails, providers work without proxies
- If a proxy test fails, the system tries other proxies
- If no working proxy is found, providers fall back to direct connections

## Logging

The system uses Python's logging module. To see proxy-related logs:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Or specifically for the autoproxy module
logger = logging.getLogger('webscout.Provider.OPENAI.autoproxy')
logger.setLevel(logging.DEBUG)
```

## Testing

Run the test suite to verify functionality:

```bash
python webscout/Provider/OPENAI/test_autoproxy.py
```

## Implementation Details

### ProxyAutoMeta Metaclass

The `ProxyAutoMeta` metaclass is applied to `OpenAICompatibleProvider` and:

1. Intercepts class instantiation
2. Checks for `disable_auto_proxy` parameter or class attribute
3. Fetches and configures proxies if not disabled
4. Patches existing session objects
5. Adds helper methods to the instance

### Session Patching

The system automatically patches these session types:
- `requests.Session` - Updates the `proxies` attribute
- `httpx.Client` - Sets the `_proxies` attribute
- `curl_cffi.Session` - Updates the `proxies` attribute
- `curl_cffi.AsyncSession` - Updates the `proxies` attribute

### Proxy Source

Proxies are fetched from: `http://207.180.209.185:5000/ips.txt`

The system expects one proxy per line in the format shown above.

## Troubleshooting

### No Proxies Available

If you see "No proxies available" messages:
1. Check if the proxy source URL is accessible
2. Verify your internet connection
3. Check if the proxy format is correct

### Proxy Test Failures

If proxy tests fail:
1. Some proxies may be temporarily unavailable (normal)
2. The test URL (`https://httpbin.org/ip`) may be blocked
3. Network connectivity issues

### Provider Not Getting Proxies

If a provider doesn't get automatic proxies:
1. Ensure it inherits from `OpenAICompatibleProvider`
2. Check if `disable_auto_proxy` is set
3. Verify the metaclass is properly imported

## Contributing

To add proxy support to a new provider:

1. Inherit from `OpenAICompatibleProvider`
2. Accept `disable_auto_proxy` parameter in `__init__`
3. Use `self.proxies` for HTTP requests
4. Optionally use helper methods like `self.get_proxied_session()`

The metaclass will handle the rest automatically!
