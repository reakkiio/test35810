<div align="center">
  <a href="https://github.com/OEvortex/Webscout">
    <img src="https://img.shields.io/badge/WebScout-OpenAI%20Compatible%20Providers-blue?style=for-the-badge&logo=python&logoColor=white" alt="WebScout OpenAI Compatible Providers">
  </a>
  <br/>
  <h1>WebScout OpenAI-Compatible Providers</h1>
  <p><strong>Seamlessly integrate with various AI providers using OpenAI-compatible interfaces</strong></p>
  <p>
    Access multiple AI providers through a standardized OpenAI-compatible interface, making it easy to switch between providers without changing your code.
  </p>
</div>

## 🚀 Overview

The WebScout OpenAI-Compatible Providers module offers a standardized way to interact with various AI providers using the familiar OpenAI API structure. This makes it easy to:

- Use the same code structure across different AI providers
- Switch between providers without major code changes
- Leverage the OpenAI ecosystem of tools and libraries with alternative AI providers

## ⚙️ Available Providers

Currently, the following providers are implemented with OpenAI-compatible interfaces:

### DeepInfra

Access DeepInfra's powerful models through an OpenAI-compatible interface.

**Available Models:**
- `deepseek-ai/DeepSeek-V3`
- `google/gemma-2-27b-it`, `google/gemma-2-9b-it`
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
- `microsoft/phi-4`, `microsoft/Phi-4-multimodal-instruct`
- `microsoft/WizardLM-2-8x22B`
- `mistralai/Mistral-Small-24B-Instruct-2501`
- `nvidia/Llama-3.1-Nemotron-70B-Instruct`
- `Qwen/QwQ-32B`, `Qwen/Qwen2.5-72B-Instruct`, `Qwen/Qwen2.5-Coder-32B-Instruct`

### Glider

Access Glider.so's models through an OpenAI-compatible interface.

**Available Models:**
- `chat-llama-3-1-70b`
- `chat-llama-3-1-8b`
- `chat-llama-3-2-3b`
- `deepseek-ai/DeepSeek-R1`

### ChatGPTClone

Access ChatGPT Clone API through an OpenAI-compatible interface.

**Available Models:**
- `gpt-4`
- `gpt-3.5-turbo`

### X0GPT

Access X0GPT API through an OpenAI-compatible interface.

**Available Models:**
- `gpt-4`
- `gpt-3.5-turbo`

### WiseCat

Access WiseCat API through an OpenAI-compatible interface.

**Available Models:**
- `chat-model-small`
- `chat-model-large`
- `chat-model-reasoning`

### Venice

Access Venice AI API through an OpenAI-compatible interface.

**Available Models:**
- `mistral-31-24b`
- `llama-3.2-3b-akash`
- `qwen2dot5-coder-32b`
- `deepseek-coder-v2-lite`

### ExaAI

Access ExaAI's O3-Mini model through an OpenAI-compatible interface.

**Available Models:**
- `O3-Mini`

**Note:** ExaAI does not support system messages. Any system messages will be automatically removed from the conversation.

### TypeGPT

Access TypeGPT.net's models through an OpenAI-compatible interface.

**Available Models:**
TypeGPT supports the following models:
- `gpt-4o-mini-2024-07-18`: OpenAI's GPT-4o mini model
- `chatgpt-4o-latest`: Latest version of ChatGPT with GPT-4o
- `deepseek-r1`: DeepSeek's R1 model
- `deepseek-v3`: DeepSeek's V3 model
- `uncensored-r1`: Uncensored version of DeepSeek R1
- `Image-Generator`: For generating images

## 💻 Usage

### Basic Usage with DeepInfra

```python
from webscout.Provider.OPENAI import DeepInfra

# Initialize the client
client = DeepInfra()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ],
    temperature=0.7,
    max_tokens=500
)

# Print the response
print(response.choices[0].message.content)
```

### Basic Usage with Glider

```python
from webscout.Provider.OPENAI import Glider

# Initialize the client
client = Glider()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="chat-llama-3-1-70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ],
    max_tokens=500
)

# Print the response
print(response.choices[0].message.content)
```

### Streaming Responses

```python
from webscout.Provider.OPENAI import DeepInfra

# Initialize the client
client = DeepInfra()

# Create a streaming completion
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True,
    temperature=0.7
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

### Streaming with Glider

```python
from webscout.Provider.OPENAI import Glider

# Initialize the client
client = Glider()

# Create a streaming completion
stream = client.chat.completions.create(
    model="chat-llama-3-1-70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

### Basic Usage with ChatGPTClone

```python
from webscout.Provider.OPENAI import ChatGPTClone

# Initialize the client
client = ChatGPTClone()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ],
    temperature=0.7
)

# Print the response
print(response.choices[0].message.content)
```

### Streaming with ChatGPTClone

```python
from webscout.Provider.OPENAI import ChatGPTClone

# Initialize the client
client = ChatGPTClone()

# Create a streaming completion
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

### Basic Usage with X0GPT

```python
from webscout.Provider.OPENAI import X0GPT

# Initialize the client
client = X0GPT()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="gpt-4",  # Model name doesn't matter for X0GPT
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ]
)

# Print the response
print(response.choices[0].message.content)
```

### Streaming with X0GPT

```python
from webscout.Provider.OPENAI import X0GPT

# Initialize the client
client = X0GPT()

# Create a streaming completion
stream = client.chat.completions.create(
    model="gpt-4",  # Model name doesn't matter for X0GPT
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

### Basic Usage with WiseCat

```python
from webscout.Provider.OPENAI import WiseCat

# Initialize the client
client = WiseCat()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="chat-model-small",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ]
)

# Print the response
print(response.choices[0].message.content)
```

### Streaming with WiseCat

```python
from webscout.Provider.OPENAI import WiseCat

# Initialize the client
client = WiseCat()

# Create a streaming completion
stream = client.chat.completions.create(
    model="chat-model-small",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

### Basic Usage with Venice

```python
from webscout.Provider.OPENAI import Venice

# Initialize the client
client = Venice(temperature=0.7, top_p=0.9)

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="mistral-31-24b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ]
)

# Print the response
print(response.choices[0].message.content)
```

### Streaming with Venice

```python
from webscout.Provider.OPENAI import Venice

# Initialize the client
client = Venice()

# Create a streaming completion
stream = client.chat.completions.create(
    model="mistral-31-24b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

### Basic Usage with ExaAI

```python
from webscout.Provider.OPENAI import ExaAI

# Initialize the client
client = ExaAI()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="O3-Mini",
    messages=[
        # Note: ExaAI does not support system messages (they will be removed)
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "Tell me about Python programming."}
    ]
)

# Print the response
print(response.choices[0].message.content)
```

### Streaming with ExaAI

```python
from webscout.Provider.OPENAI import ExaAI

# Initialize the client
client = ExaAI()

# Create a streaming completion
stream = client.chat.completions.create(
    model="O3-Mini",
    messages=[
        # Note: ExaAI does not support system messages (they will be removed)
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

### Basic Usage with TypeGPT

```python
from webscout.Provider.OPENAI import TypeGPT

# Initialize the client
client = TypeGPT()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="chatgpt-4o-latest",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ]
)

# Print the response
print(response.choices[0].message.content)
```

### Streaming with TypeGPT

```python
from webscout.Provider.OPENAI import TypeGPT

# Initialize the client
client = TypeGPT()

# Create a streaming completion
stream = client.chat.completions.create(
    model="chatgpt-4o-latest",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
```

## 🔄 Response Format

The response format mimics the OpenAI API structure:

### Non-streaming Response

```json
{
  "id": "chatcmpl-123abc",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
  },
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "This is a response from the model."
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

### Streaming Response Chunks

Each chunk in a streaming response follows this format:

```json
{
  "id": "chatcmpl-123abc",
  "object": "chat.completion.chunk",
  "created": 1677858242,
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "choices": [
    {
      "delta": {
        "content": "This "
      },
      "finish_reason": null,
      "index": 0
    }
  ]
}
```

## 🧩 Architecture

The OpenAI-compatible providers are built on a modular architecture:

- `base.py`: Contains abstract base classes that define the OpenAI-compatible interface
- `utils.py`: Provides data structures that mimic OpenAI's response format
- Provider-specific implementations (e.g., `deepinfra.py`): Implement the abstract interfaces for specific providers

This architecture makes it easy to add new providers while maintaining a consistent interface.

## 📝 Notes

- Some providers may require API keys for full functionality
- Not all OpenAI features are supported by all providers
- Response formats are standardized to match OpenAI's format, but the underlying content depends on the specific provider and model

## 🤝 Contributing

Want to add a new OpenAI-compatible provider? Follow these steps:

1. Create a new file in the `webscout/Provider/OPENAI` directory
2. Implement the `OpenAICompatibleProvider` interface
3. Add appropriate tests
4. Update this README with information about the new provider

## 📚 Related Documentation

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [DeepInfra Documentation](https://deepinfra.com/docs)
- [Glider.so Website](https://glider.so/)
- [ChatGPT Clone Website](https://chatgpt-clone-ten-nu.vercel.app/)
- [X0GPT Website](https://x0-gpt.devwtf.in/)
- [WiseCat Website](https://wise-cat-groq.vercel.app/)
- [Venice AI Website](https://venice.ai/)
- [ExaAI Website](https://o3minichat.exa.ai/)
- [TypeGPT Website](https://chat.typegpt.net/)

<div align="center">
  <a href="https://t.me/PyscoutAI"><img alt="Telegram Group" src="https://img.shields.io/badge/Telegram%20Group-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
  <a href="https://buymeacoffee.com/oevortex"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black"></a>
</div>
