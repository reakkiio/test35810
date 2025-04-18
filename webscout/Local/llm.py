"""
LLM interface for webscout.Local using llama-cpp-python
"""

from typing import Dict, Any, List, Optional, Union, Generator, Callable

from llama_cpp import Llama
from rich.console import Console

from .config import config
from .model_manager import ModelManager

console = Console()

class LLMInterface:
    """
    Interface for LLM models using llama-cpp-python.
    Provides methods for loading models and generating completions or chat responses.
    """
    model_name: str
    model_manager: ModelManager
    model_path: Optional[str]
    llm: Optional[Llama]

    def __init__(self, model_name: str) -> None:
        """
        Initialize the LLM interface.
        Args:
            model_name (str): Name of the model to load.
        Raises:
            ValueError: If the model is not found locally.
        """
        self.model_name = model_name
        self.model_manager = ModelManager()
        self.model_path = self.model_manager.get_model_path(model_name)
        if not self.model_path:
            raise ValueError(f"Model {model_name} not found. Please download it first.")
        self.llm = None

    def load_model(self, n_gpu_layers: Optional[int] = None, n_ctx: Optional[int] = None, verbose: bool = False) -> None:
        """
        Load the model into memory.
        Args:
            n_gpu_layers (Optional[int]): Number of layers to offload to GPU (-1 for all).
            n_ctx (Optional[int]): Context size.
            verbose (bool): Whether to show verbose output.
        Raises:
            ValueError: If model loading fails.
        """
        if n_gpu_layers is None:
            n_gpu_layers = config.get("default_gpu_layers", -1)
        if n_ctx is None:
            n_ctx = config.get("default_context_length", 4096)
        console.print(f"[bold blue]Loading model {self.model_name}...[/bold blue]")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=verbose
            )
            console.print(f"[bold green]Model {self.model_name} loaded successfully[/bold green]")
        except Exception as e:
            raise ValueError(f"Failed to load model from file: {self.model_path}\n{str(e)}")

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a completion for the given prompt.
        Args:
            prompt (str): The prompt to complete.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            stream (bool): Whether to stream the response.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: Completion result or generator for streaming.
        """
        if self.llm is None:
            self.load_model()
        if stream:
            return self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=stop or [],
            )
        else:
            return self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stop=stop or [],
            )

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a chat completion for the given messages.
        Args:
            messages (List[Dict[str, str]]): List of chat messages.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            stream (bool): Whether to stream the response.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: Chat completion result or generator for streaming.
        """
        if self.llm is None:
            self.load_model()
        processed_messages: List[Dict[str, str]] = messages.copy()
        system_messages = [m for m in processed_messages if m.get("role") == "system"]
        non_system_messages = [m for m in processed_messages if m.get("role") != "system"]
        if system_messages:
            processed_messages = [system_messages[0]] + non_system_messages
        else:
            processed_messages = non_system_messages
        if stream:
            return self.llm.create_chat_completion(
                messages=processed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=stop or [],
            )
        else:
            return self.llm.create_chat_completion(
                messages=processed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stop=stop or [],
            )

    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        callback: Callable[[str], None],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> None:
        """
        Stream a chat completion with a callback for each token.
        Args:
            messages (List[Dict[str, str]]): List of chat messages.
            callback (Callable[[str], None]): Function to call with each token.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
        """
        stream = self.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
        )
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    callback(content)
