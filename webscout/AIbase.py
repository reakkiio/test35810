from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Union, Generator, Optional, Any
from typing_extensions import TypeAlias

# Type aliases for better readability
Response: TypeAlias = dict[str, Union[str, bool, None]]
AsyncImageData: TypeAlias = Union[bytes, str, AsyncGenerator[bytes, None]]

class SearchResponse:
    """A wrapper class for search API responses.
    
    This class automatically converts response objects to their text representation
    when printed or converted to string.
    
    Attributes:
        text (str): The text content of the response
        
    Example:
        >>> response = SearchResponse("Hello, world!")
        >>> print(response)
        Hello, world!
        >>> str(response)
        'Hello, world!'
    """
    def __init__(self, text: str):
        self.text = text
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.text

class AIProviderError(Exception):
    pass

class Provider(ABC):

    @abstractmethod
    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Response:
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> str:
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    def get_message(self, response: Response) -> str:
        raise NotImplementedError("Method needs to be implemented in subclass")

class AsyncProvider(ABC):

    @abstractmethod
    async def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Response:
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    async def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> str:
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    async def get_message(self, response: Response) -> str:
        raise NotImplementedError("Method needs to be implemented in subclass")

class TTSProvider(ABC):

    @abstractmethod
    def tts(self, text: str, voice: str = None, verbose: bool = False) -> str:
        """Convert text to speech and save to a temporary file.

        Args:
            text (str): The text to convert to speech
            voice (str, optional): The voice to use. Defaults to provider's default voice.
            verbose (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            str: Path to the generated audio file
        """
        raise NotImplementedError("Method needs to be implemented in subclass")

    def save_audio(self, audio_file: str, destination: str = None, verbose: bool = False) -> str:
        """Save audio to a specific destination.

        Args:
            audio_file (str): Path to the source audio file
            destination (str, optional): Destination path. Defaults to current directory with timestamp.
            verbose (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            str: Path to the saved audio file
        """
        import shutil
        import os
        from pathlib import Path
        import time

        source_path = Path(audio_file)

        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if destination is None:
            # Create a default destination with timestamp in current directory
            timestamp = int(time.time())
            destination = os.path.join(os.getcwd(), f"tts_audio_{timestamp}{source_path.suffix}")

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

        # Copy the file
        shutil.copy2(source_path, destination)

        if verbose:
            print(f"[debug] Audio saved to {destination}")

        return destination

    def stream_audio(self, text: str, voice: str = None, chunk_size: int = 1024, verbose: bool = False) -> Generator[bytes, None, None]:
        """Stream audio in chunks.

        Args:
            text (str): The text to convert to speech
            voice (str, optional): The voice to use. Defaults to provider's default voice.
            chunk_size (int, optional): Size of audio chunks to yield. Defaults to 1024.
            verbose (bool, optional): Whether to print debug information. Defaults to False.

        Yields:
            Generator[bytes, None, None]: Audio data chunks
        """
        # Generate the audio file
        audio_file = self.tts(text, voice=voice, verbose=verbose)

        # Stream the file in chunks
        with open(audio_file, 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk

class AsyncTTSProvider(ABC):

    @abstractmethod
    async def tts(self, text: str, voice: str = None, verbose: bool = False) -> str:
        """Convert text to speech and save to a temporary file asynchronously.

        Args:
            text (str): The text to convert to speech
            voice (str, optional): The voice to use. Defaults to provider's default voice.
            verbose (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            str: Path to the generated audio file
        """
        raise NotImplementedError("Method needs to be implemented in subclass")


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers."""

    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Transcribe audio file to text.

        Args:
            audio_path (Union[str, Path]): Path to the audio file
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Transcription result in OpenAI Whisper format
        """
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    def transcribe_from_url(self, audio_url: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio from URL to text.

        Args:
            audio_url (str): URL of the audio file
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Transcription result in OpenAI Whisper format
        """
        raise NotImplementedError("Method needs to be implemented in subclass")


class AsyncSTTProvider(ABC):
    """Abstract base class for asynchronous Speech-to-Text providers."""

    @abstractmethod
    async def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Transcribe audio file to text asynchronously.

        Args:
            audio_path (Union[str, Path]): Path to the audio file
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Transcription result in OpenAI Whisper format
        """
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    async def transcribe_from_url(self, audio_url: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio from URL to text asynchronously.

        Args:
            audio_url (str): URL of the audio file
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Transcription result in OpenAI Whisper format
        """
        raise NotImplementedError("Method needs to be implemented in subclass")

    async def save_audio(self, audio_file: str, destination: str = None, verbose: bool = False) -> str:
        """Save audio to a specific destination asynchronously.

        Args:
            audio_file (str): Path to the source audio file
            destination (str, optional): Destination path. Defaults to current directory with timestamp.
            verbose (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            str: Path to the saved audio file
        """
        import shutil
        import os
        from pathlib import Path
        import time
        import asyncio

        source_path = Path(audio_file)

        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if destination is None:
            # Create a default destination with timestamp in current directory
            timestamp = int(time.time())
            destination = os.path.join(os.getcwd(), f"tts_audio_{timestamp}{source_path.suffix}")

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

        # Copy the file using asyncio to avoid blocking
        await asyncio.to_thread(shutil.copy2, source_path, destination)

        if verbose:
            print(f"[debug] Audio saved to {destination}")

        return destination

    async def stream_audio(self, text: str, voice: str = None, chunk_size: int = 1024, verbose: bool = False) -> AsyncGenerator[bytes, None]:
        """Stream audio in chunks asynchronously.

        Args:
            text (str): The text to convert to speech
            voice (str, optional): The voice to use. Defaults to provider's default voice.
            chunk_size (int, optional): Size of audio chunks to yield. Defaults to 1024.
            verbose (bool, optional): Whether to print debug information. Defaults to False.

        Yields:
            AsyncGenerator[bytes, None]: Audio data chunks
        """
        import aiofiles

        # Generate the audio file
        audio_file = await self.tts(text, voice=voice, verbose=verbose)

        # Stream the file in chunks
        async with aiofiles.open(audio_file, 'rb') as f:
            while chunk := await f.read(chunk_size):
                yield chunk
                
class AISearch(ABC):
    """Abstract base class for AI-powered search providers.
    
    This class defines the interface for AI search providers that can perform
    web searches and return AI-generated responses based on search results.
    
    All search providers should inherit from this class and implement the
    required methods.
    """
    
    @abstractmethod
    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]:
        """Search using the provider's API and get AI-generated responses.
        
        This method sends a search query to the provider and returns the AI-generated response.
        It supports both streaming and non-streaming modes, as well as raw response format.
        
        Args:
            prompt (str): The search query or prompt to send to the API.
            stream (bool, optional): If True, yields response chunks as they arrive.
                                   If False, returns complete response. Defaults to False.
            raw (bool, optional): If True, returns raw response dictionaries.
                                If False, returns SearchResponse objects that convert to text automatically.
                                Defaults to False.
        
        Returns:
            Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]: 
                - If stream=False: Returns complete response as SearchResponse object
                - If stream=True: Yields response chunks as either Dict or SearchResponse objects
        
        Raises:
            APIConnectionError: If the API request fails
        """
        raise NotImplementedError("Method needs to be implemented in subclass")

class AsyncAISearch(ABC):
    """Abstract base class for asynchronous AI-powered search providers.
    
    This class defines the interface for asynchronous AI search providers that can perform
    web searches and return AI-generated responses based on search results.
    
    All asynchronous search providers should inherit from this class and implement the
    required methods.
    """
    
    @abstractmethod
    async def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[SearchResponse, AsyncGenerator[Union[Dict[str, str], SearchResponse], None]]:
        """Search using the provider's API and get AI-generated responses asynchronously.
        
        This method sends a search query to the provider and returns the AI-generated response.
        It supports both streaming and non-streaming modes, as well as raw response format.
        
        Args:
            prompt (str): The search query or prompt to send to the API.
            stream (bool, optional): If True, yields response chunks as they arrive.
                                   If False, returns complete response. Defaults to False.
            raw (bool, optional): If True, returns raw response dictionaries.
                                If False, returns SearchResponse objects that convert to text automatically.
                                Defaults to False.
        
        Returns:
            Union[SearchResponse, AsyncGenerator[Union[Dict[str, str], SearchResponse], None]]: 
                - If stream=False: Returns complete response as SearchResponse object
                - If stream=True: Yields response chunks as either Dict or SearchResponse objects
        
        Raises:
            APIConnectionError: If the API request fails
        """
        raise NotImplementedError("Method needs to be implemented in subclass")
