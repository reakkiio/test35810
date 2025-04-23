from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, List, Union, Generator, Optional
from typing_extensions import TypeAlias

# Type aliases for better readability
Response: TypeAlias = dict[str, Union[str, bool, None]]
ImageData: TypeAlias = Union[bytes, str, Generator[bytes, None, None]]
AsyncImageData: TypeAlias = Union[bytes, str, AsyncGenerator[bytes, None]]

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

class ImageProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str, amount: int = 1) -> List[bytes]:
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    def save(
        self,
        response: List[bytes],
        name: Optional[str] = None,
        dir: Optional[Union[str, Path]] = None
    ) -> List[str]:
        raise NotImplementedError("Method needs to be implemented in subclass")

class AsyncImageProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        amount: int = 1
    ) -> Union[AsyncGenerator[bytes, None], List[bytes]]:
        raise NotImplementedError("Method needs to be implemented in subclass")

    @abstractmethod
    async def save(
        self,
        response: Union[AsyncGenerator[bytes, None], List[bytes]],
        name: Optional[str] = None,
        dir: Optional[Union[str, Path]] = None
    ) -> List[str]:
        raise NotImplementedError("Method needs to be implemented in subclass")

class AISearch(ABC):

    @abstractmethod
    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Response:
        raise NotImplementedError("Method needs to be implemented in subclass")