# PixelMuse Image Provider 🎨

A powerful image generation provider using PixelMuse's API to create stunning AI art.

## Features ✨

- Generate high-quality AI images from text prompts
- Support for multiple models
- Customizable aspect ratio and style
- Automatic image saving with customizable filenames
- Built-in retry mechanism for reliability

## Installation 📦

No additional installation required! This provider is included in the webscout package.

## Usage 🚀

```python
from webscout.Provider.TTI.pixelmuse import PixelMuseImager

# Create an instance of the provider
imager = PixelMuseImager(model="flux-schnell")

# Generate images
images = imager.generate(
    prompt="A magical forest with glowing mushrooms",
    amount=1,
    style="none",
    aspect_ratio="1:1"
)

# Save the generated images
paths = imager.save(images)
print(f"Saved images to: {paths}")
```

## Parameters ⚙️

### Initialization
- `model` (str): Which model to use (default: "flux-schnell")
- `timeout` (int): Request timeout in seconds (default: 60)
- `proxies` (dict): Proxy settings for requests (default: {})
- `logging` (bool): Enable logging (default: True)

### Generate Method
- `prompt` (str): Your creative prompt
- `amount` (int): Number of images to generate (default: 1)
- `max_retries` (int): Maximum retry attempts (default: 3)
- `retry_delay` (int): Seconds to wait between retries (default: 5)
- `style` (str): Style to apply (default: "none")
- `aspect_ratio` (str): Aspect ratio (default: "1:1")

### Save Method
- `response` (List[bytes]): List of image data
- `name` (str): Base name for saved files (default: prompt)
- `dir` (str): Directory to save images (default: current directory)
- `filenames_prefix` (str): Prefix for filenames (default: "")

## Available Models 🎯

- `flux-schnell`: Fast and efficient model
- `imagen-3-fast`: Quick Imagen 3 model
- `imagen-3`: Full Imagen 3 model
- `recraft-v3`: Recraft v3 model

## Notes 📝

- Currently only supports 1:1 aspect ratio
- Images are saved in WebP format
- Requires internet connection
- Rate limits may apply based on PixelMuse's API policies

## License 📄

This provider is part of the webscout package. See the main package license for details. 