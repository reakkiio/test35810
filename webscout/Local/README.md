<div align="center">
  <a href="https://github.com/OE-LUCIFER/Webscout">
    <img src="https://img.shields.io/badge/Inferno-Local%20LLM%20Server-orange?style=for-the-badge&logo=python&logoColor=white" alt="Inferno Logo">
  </a>

  <h1>Inferno</h1>

  <p><strong>A powerful llama-cpp-python based LLM serving tool similar to Ollama</strong></p>

  <p>
    Run local LLMs with an OpenAI-compatible API, interactive CLI, and seamless Hugging Face integration.
  </p>

  <!-- Badges -->
  <p>
    <a href="https://pypi.org/project/webscout/"><img src="https://img.shields.io/pypi/v/webscout.svg?style=flat-square&logo=pypi&label=PyPI" alt="PyPI Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/pypi/pyversions/webscout?style=flat-square&logo=python" alt="Python Version"></a>
  </p>
</div>

> [!IMPORTANT]
> Inferno is part of the Webscout package and can be installed with:
> ```bash
> pip install webscout[Local]
> ```
> This will install all required dependencies for running local LLMs.

> [!NOTE]
> Inferno supports both command-line usage with `inferno` or `webscout-local` commands, and can also be run as a Python module with `python -m inferno` or `python -m webscout.Local`.

<div align="center">
  <!-- Social/Support Links -->
  <p>
    <a href="https://t.me/PyscoutAI"><img alt="Telegram Group" src="https://img.shields.io/badge/Telegram%20Group-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
    <a href="https://youtube.com/@OEvortex"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white"></a>
    <a href="https://buymeacoffee.com/oevortex"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black"></a>
  </p>
</div>

## 🚀 Features

- **Hugging Face Integration:** Download models directly with interactive file selection
- **Flexible Model Specification:** Support for `repo_id:filename` format for direct file targeting
- **OpenAI-Compatible API:** Use with any client that supports the OpenAI API
- **Interactive CLI:** Powerful command-line interface for model management
- **Chat Interface:** Built-in interactive chat with system prompt customization
- **Streaming Support:** Real-time streaming responses for chat and completions
- **GPU Acceleration:** Utilize GPU for faster inference when available
- **Context Window Control:** Adjust context size for different models and use cases

## ⚙️ Installation

Install Inferno using pip with the Local extra:

```bash
pip install -U webscout[Local]
```

## 🖥️ Command Line Interface

Inferno provides a powerful command-line interface for managing and using LLMs:

```bash
# Show available commands
inferno --help

# Alternative command name
webscout-local --help

# Using as a Python module
python -m inferno --help
python -m webscout.Local --help
```

| Command | Description |
|---------|-------------|
| `inferno pull <model>` | Download a model from Hugging Face |
| `inferno list` | List downloaded models |
| `inferno serve <model>` | Start a model server |
| `inferno run <model>` | Chat with a model interactively |
| `inferno remove <model>` | Remove a downloaded model |
| `inferno version` | Show version information |

## 📋 Usage Guide

### Download a Model

```bash
# Download a model from Hugging Face (interactive file selection)
inferno pull Abhaykoul/HAI3-raw-Q4_K_M-GGUF

# Download a specific file using repo_id:filename format
inferno pull Abhaykoul/HAI3-raw-Q4_K_M-GGUF:hai3-raw-q4_k_m.gguf
```

### List Downloaded Models

```bash
inferno list
```

### Start the Server

```bash
# Start the server with a downloaded model
inferno serve HAI3-raw-Q4_K_M-GGUF

# Start the server with a model from Hugging Face (downloads if needed)
inferno serve Abhaykoul/HAI3-raw-Q4_K_M-GGUF

# Start the server with a specific file using repo_id:filename format
inferno serve Abhaykoul/HAI3-raw-Q4_K_M-GGUF:hai3-raw-q4_k_m.gguf

# Specify host and port
inferno serve HAI3-raw-Q4_K_M-GGUF --host 0.0.0.0 --port 8080
```

### Chat with a Model

```bash
inferno run HAI3-raw-Q4_K_M-GGUF
```

#### Available Chat Commands

| Command | Description |
|---------|-------------|
| `/help` or `/?` | Show available commands |
| `/bye` | Exit the chat |
| `/set system <prompt>` | Set the system prompt (use quotes for multi-word prompts) |
| `/set context <size>` | Set context window size (default: 4096) |
| `/clear` or `/cls` | Clear the terminal screen |
| `/reset` | Reset all settings |

### Remove a Model

```bash
inferno remove HAI3-raw-Q4_K_M-GGUF
```

## 🔌 API Usage

Inferno provides an OpenAI-compatible API. You can use it with any client that supports the OpenAI API.

### Python Example

```python
import openai

# Configure the client
openai.api_key = "dummy"  # Not used but required
openai.api_base = "http://localhost:8000/v1"  # Default Inferno API URL

# Chat completion
response = openai.ChatCompletion.create(
    model="HAI3-raw-Q4_K_M-GGUF",  # Use the model name
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)

# Streaming chat completion
for chunk in openai.ChatCompletion.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    stream=True
):
    if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 🧩 Integration with Applications

Inferno can be easily integrated with various applications that support the OpenAI API format:

```python
# Example with LangChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configure to use local Inferno server
chat = ChatOpenAI(
    model_name="HAI3-raw-Q4_K_M-GGUF",
    openai_api_key="dummy",
    openai_api_base="http://localhost:8000/v1",
    streaming=True
)

# Use the model
response = chat([HumanMessage(content="Explain quantum computing in simple terms")])
print(response.content)
```

## 📦 Requirements

- Python 3.9+
- llama-cpp-python
- FastAPI
- Uvicorn
- Rich
- Typer
- Hugging Face Hub
- Pydantic
- Requests

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to Inferno, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive messages
4. Push your branch to your forked repository
5. Submit a pull request to the main repository

## 📄 License

MIT

---

<div align="center">
  <p>Made with ❤️ by the Webscout team</p>
</div>
