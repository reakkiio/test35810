# 🎨 MagicStudio Image Generation

Generate amazing images with MagicStudio's AI art generator! 🚀

## 🌟 Features

- Fast and reliable image generation
- Both sync and async implementations
- Smart retry mechanism
- Proxy support
- Custom timeouts
- Easy-to-use interface

## 📦 Installation

```bash
pip install webscout
```

## 🚀 Quick Start

### Sync Usage

```python
from webscout import MagicStudioImager

# Initialize the provider
provider = MagicStudioImager()

# Generate a single image
images = provider.generate("A beautiful sunset over mountains")
paths = provider.save(images)

# Generate multiple images
images = provider.generate(
    prompt="Epic dragon breathing fire",
    amount=3
)
paths = provider.save(images, dir="dragon_pics")
```


## ⚙️ Configuration

```python
# Custom settings
provider = MagicStudioImager(
    timeout=120,  # Longer timeout
    proxies={
        'http': 'http://proxy:8080',
        'https': 'http://proxy:8080'
    }
)

# Advanced usage
images = provider.generate(
    prompt="A shiny red sports car",
    amount=3,
    max_retries=5,
    retry_delay=3
)
```

## 🛡️ Error Handling

```python
try:
    images = provider.generate("Cool art")
    paths = provider.save(images)
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Generation failed: {e}")
```

## 💡 Tips

1. Use clear, descriptive prompts
2. Set longer timeouts for multiple images
3. Enable proxies for better reliability
4. Use retry mechanism for stability
5. Save images with meaningful names
