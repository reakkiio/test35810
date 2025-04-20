"""
Webscout.Local - A llama-cpp-python based LLM serving tool with Ollama-compatible API
"""
from webscout.version import __version__

# Import main components for easier access
from .llm import LLMInterface
from .model_manager import ModelManager
from .server import start_server

# Define what's available when using `from webscout.Local import *`
__all__ = ["LLMInterface", "ModelManager", "start_server"]
