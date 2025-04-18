<div align="center">
  <h1>🔍 Webscout AI Search Providers</h1>
  <p><strong>Powerful AI-powered search capabilities with multiple provider support</strong></p>
</div>

> [!NOTE]
> AI Search Providers leverage advanced language models and search algorithms to deliver high-quality, context-aware responses with web search integration.

## ✨ Features

- **Multiple Search Providers**: Support for 7+ specialized AI search services
- **Streaming Responses**: Real-time streaming of AI-generated responses
- **Raw Response Format**: Access to raw response data when needed
- **Automatic Text Handling**: Smart response formatting and cleaning
- **Robust Error Handling**: Comprehensive error management
- **Cross-Platform Compatibility**: Works seamlessly across different environments

## 📦 Supported Search Providers

| Provider | Description | Key Features |
|----------|-------------|-------------|
| **DeepFind** | General purpose AI search | Web-based, reference removal, clean formatting |
| **Felo** | Fast streaming search | Advanced capabilities, real-time streaming |
| **Isou** | Scientific search | Multiple model selection, citation handling |
| **Genspark** | Efficient search | Fast response, markdown link removal |
| **Monica** | Comprehensive search | Related question suggestions, source references |
| **WebPilotAI** | Web-integrated search | Web page analysis, content extraction |
| **Scira** | Research-focused search | Multiple models (Grok3, Claude), vision support |
| **IAsk** | Multi-mode search | Question, Academic, Fast modes, detail levels |
| **Hika** | General AI search | Simple interface, clean text output |
| **Perplexity** | Advanced AI search & chat | Multiple modes (Pro, Reasoning), model selection, source control |

## 🚀 Installation

```bash
pip install -U webscout
```

## 💻 Quick Start Guide

### Basic Usage Pattern

All AI Search providers follow a consistent usage pattern:

```python
from webscout import ProviderName

# Initialize the provider
ai = ProviderName()

# Basic search
response = ai.search("Your query here")
print(response)  # Automatically formats the response

# Streaming search
for chunk in ai.search("Your query here", stream=True):
    print(chunk, end="", flush=True)  # Print response as it arrives
```

### Provider Examples

<details>
<summary><strong>DeepFind Example</strong></summary>

```python
from webscout import DeepFind

# Initialize the search provider
ai = DeepFind()

# Basic search
response = ai.search("What is Python?")
print(response)

# Streaming search
for chunk in ai.search("Tell me about AI", stream=True):
    print(chunk, end="")
```
</details>

<details>
<summary><strong>Scira Example</strong></summary>

```python
from webscout import Scira

# Initialize with default model (Grok3)
ai = Scira()

# Basic search
response = ai.search("What is the impact of climate change?")
print(response)

# Streaming search with Claude model
ai = Scira(model="scira-claude")
for chunk in ai.search("Explain quantum computing", stream=True):
    print(chunk, end="", flush=True)

# Available models:
# - scira-default (Grok3)
# - scira-grok-3-mini (Grok3-mini)
# - scira-vision (Grok2-Vision)
# - scira-claude (Sonnet-3.7)
# - scira-optimus (optimus)
```
</details>

<details>
<summary><strong>Isou Example</strong></summary>

```python
from webscout import Isou

# Initialize with specific model
ai = Isou(model="siliconflow:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

# Get a response with scientific information
response = ai.search("Explain the double-slit experiment")
print(response)
```
</details>

<details>
<summary><strong>Perplexity Example</strong></summary>

```python
from webscout import Perplexity

# Initialize (optionally pass cookies for authenticated features)
# cookies = {"perplexity-user": "your_cookie_value"}
# ai = Perplexity(cookies=cookies)
ai = Perplexity() # Anonymous access

# Basic search (auto mode)
response = ai.search("What is the weather in London?")
print(response)

# Streaming search
for chunk in ai.search("Explain black holes", stream=True):
    print(chunk, end="", flush=True)

# Pro search with specific model (requires authentication via cookies)
# try:
#     ai_pro = Perplexity(cookies=your_cookies)
#     response_pro = ai_pro.search("Latest AI research papers", mode='pro', model='gpt-4o', sources=['scholar'])
#     print(response_pro)
# except Exception as e:
#     print(f"Pro search failed: {e}")

# Available modes: 'auto', 'pro', 'reasoning', 'deep research'
# Available sources: 'web', 'scholar', 'social'
# Models depend on the mode selected.
```
</details>

## 🎛️ Advanced Configuration

<details>
<summary><strong>Timeout and Proxy Settings</strong></summary>

```python
# Configure timeout
ai = DeepFind(timeout=60)  # 60 seconds timeout

# Use with proxy
proxies = {'http': 'http://proxy.com:8080'}
ai = Felo(proxies=proxies)

# Configure max tokens (for providers that support it)
ai = Genspark(max_tokens=800)

# Configure model and group for Scira
ai = Scira(model="scira-claude", group="web")
```
</details>

<details>
<summary><strong>Response Formats</strong></summary>

```python
# Get raw response format
response = ai.search("Hello", stream=True, raw=True)
# Output: {'text': 'Hello'}, {'text': ' there!'}, etc.

# Get formatted text response
response = ai.search("Hello", stream=True)
# Output: Hello there!
```
</details>

## 🔧 Provider Capabilities

| Provider | Key Capabilities | Technical Details |
|----------|-----------------|-------------------|
| **DeepFind** | • Web-based AI search<br>• Automatic reference removal<br>• Clean response formatting | • Streaming support with progress tracking<br>• JSON response parsing<br>• Error handling |
| **Felo** | • Advanced search capabilities<br>• Real-time response streaming<br>• JSON-based response parsing | • Automatic text cleaning<br>• Session management<br>• Rate limiting support |
| **Isou** | • Multiple model selection<br>• Scientific and general category support<br>• Citation handling | • Deep and simple search modes<br>• Specialized model options<br>• Markdown formatting |
| **Genspark** | • Fast response generation<br>• Markdown link removal<br>• JSON structure normalization | • Session-based API interactions<br>• Efficient content parsing<br>• Streaming optimization |
| **Monica** | • Comprehensive search responses<br>• Related question suggestions<br>• Source references | • Answer snippets<br>• Clean formatted responses<br>• Web content integration |
| **WebPilotAI** | • Web page analysis<br>• Content extraction<br>• Structured data retrieval | • URL processing<br>• HTML parsing<br>• Metadata extraction |
| **Scira** | • Research-focused search<br>• Multiple model options<br>• Vision support | • Grok3, Claude, Vision models<br>• Customizable group parameters<br>• Efficient content parsing |
| **IAsk** | • Multi-mode search (Question, Academic, etc.)<br>• Adjustable detail level<br>• Source citation | • Asynchronous backend (sync wrapper)<br>• WebSocket communication<br>• HTML parsing & formatting |
| **Hika** | • General AI search<br>• Simple streaming<br>• Basic text cleaning | • SSE streaming<br>• Custom headers for auth<br>• JSON response parsing |
| **Perplexity** | • Multiple search modes (Pro, Reasoning)<br>• Model selection per mode<br>• Source filtering (web, scholar, social)<br>• Follow-up questions | • `curl_cffi` for Cloudflare bypass<br>• Socket.IO communication<br>• SSE streaming<br>• Requires cookies for Pro features |

## 🛡️ Error Handling

<details>
<summary><strong>Exception Handling Example</strong></summary>

```python
from webscout import exceptions

try:
    response = ai.search("Your query")
except exceptions.APIConnectionError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
```
</details>

## 📝 Response Handling

<details>
<summary><strong>Working with Response Objects</strong></summary>

```python
# Response objects automatically convert to text
response = ai.search("What is AI?")
print(response)  # Prints formatted text

# Access raw text if needed
print(response.text)
```
</details>

## 🔒 Best Practices

<details>
<summary><strong>Streaming for Long Responses</strong></summary>

```python
for chunk in ai.search("Long query", stream=True):
    print(chunk, end="", flush=True)
```
</details>

<details>
<summary><strong>Error Handling</strong></summary>

```python
try:
    response = ai.search("Query")
except exceptions.APIConnectionError:
    # Handle connection errors
    pass
```
</details>

<details>
<summary><strong>Provider Selection Guide</strong></summary>

| Use Case | Recommended Provider |
|----------|----------------------|
| General purpose search | **DeepFind**, **Hika** |
| Fast streaming responses | **Felo** |
| Scientific or specialized queries | **Isou**, **Scira** |
| Clean and efficient responses | **Genspark** |
| Comprehensive answers with sources | **Monica**, **IAsk** |
| Web page interaction/analysis | **WebPilotAI** |
| Advanced control (modes, models) | **Perplexity**, **Scira**, **Isou** |
| Research-focused | **Scira**, **Isou**, **Perplexity** (with scholar source) |

</details>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
