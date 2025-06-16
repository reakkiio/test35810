# 🖼️ Webscout Text-to-Image (TTI) Providers

Webscout includes a collection of Text-to-Image providers that follow a common interface inspired by the OpenAI Python client. Each provider exposes an `images.create()` method which returns an `ImageResponse` object containing either image URLs or base64 data.

These providers allow you to easily generate AI‑created art from text prompts while handling image conversion and temporary hosting automatically.

## ✨ Features

- **Unified API** – Consistent `images.create()` method for all providers
- **Multiple Providers** – Generate images using different third‑party services
- **URL or Base64 Output** – Receive image URLs (uploaded to catbox.moe/0x0.st) or base64 encoded bytes
- **PNG/JPEG Conversion** – Images are converted in memory to your chosen format
- **Model Listing** – Query available models with `provider.models.list()`

## 📦 Supported Providers

| Provider         | Available Models (examples)               |
| ---------------- | ----------------------------------------- |
| `AIArta`         | `flux`, `medieval`, `dreamshaper_xl`, ... |
| `InfipAI`        | `img3`, `img4`, `uncen`                   |
| `MagicStudioAI`  | `magicstudio`                             |
| `PixelMuse`      | `flux-schnell`, `imagen-3`, `recraft-v3`  |
| `PiclumenAI`     | `piclumen-v1`                             |
| `PollinationsAI` | `flux`, `turbo`, `gptimage`               |

> **Note**: Some providers require the `Pillow` package for image processing.

## 🚀 Quick Start

```python
from webscout.Provider.TTI import PixelMuse

# Initialize the provider
client = PixelMuse()

# Generate two images and get URLs
response = client.images.create(
    model="flux-schnell",
    prompt="A futuristic city skyline at sunset",
    n=2,
    response_format="url"
)

print(response)
```

### Accessing Models

Each provider exposes the models it supports:

```python
from webscout.Provider.TTI import AIArta

ai = AIArta()
print(ai.models.list())  # List model identifiers
```

### Base64 Output

If you prefer the raw image data:

```python
response = client.images.create(
    model="flux-schnell",
    prompt="Crystal mountain landscape",
    response_format="b64_json"
)
# `response.data` will contain base64 strings
```

## 🔧 Provider Specifics

- **AIArta** – Uses Firebase authentication tokens and supports many tattoo‑style models.
- **InfipAI** – Offers various models for different image styles.
- **MagicStudioAI** – Generates images through MagicStudio's public endpoint.
- **PixelMuse** – Supports several models and converts images from WebP.
- **PiclumenAI** – Returns JPEG images directly from the API.
- **PollinationsAI** – Allows setting a custom seed for reproducible results.

## 🤝 Contributing

Contributions and additional providers are welcome! Feel free to submit a pull request.
