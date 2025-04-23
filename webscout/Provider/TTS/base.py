"""
Base class for TTS providers with common functionality.
"""
import os
import tempfile
from pathlib import Path
from typing import Generator, Optional
from webscout.AIbase import TTSProvider

class BaseTTSProvider(TTSProvider):
    """
    Base class for TTS providers with common functionality.
    
    This class implements common methods like save_audio and stream_audio
    that can be used by all TTS providers.
    """
    
    def __init__(self):
        """Initialize the base TTS provider."""
        self.temp_dir = tempfile.mkdtemp(prefix="webscout_tts_")
    
    def save_audio(self, audio_file: str, destination: str = None, verbose: bool = False) -> str:
        """
        Save audio to a specific destination.
        
        Args:
            audio_file (str): Path to the source audio file
            destination (str, optional): Destination path. Defaults to current directory with timestamp.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
            
        Returns:
            str: Path to the saved audio file
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
        """
        import shutil
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
        """
        Stream audio in chunks.
        
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


class AsyncBaseTTSProvider:
    """
    Base class for async TTS providers with common functionality.
    
    This class implements common async methods like save_audio and stream_audio
    that can be used by all async TTS providers.
    """
    
    def __init__(self):
        """Initialize the async base TTS provider."""
        self.temp_dir = tempfile.mkdtemp(prefix="webscout_tts_")
    
    async def save_audio(self, audio_file: str, destination: str = None, verbose: bool = False) -> str:
        """
        Save audio to a specific destination asynchronously.
        
        Args:
            audio_file (str): Path to the source audio file
            destination (str, optional): Destination path. Defaults to current directory with timestamp.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
            
        Returns:
            str: Path to the saved audio file
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
        """
        import shutil
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
    
    async def stream_audio(self, text: str, voice: str = None, chunk_size: int = 1024, verbose: bool = False):
        """
        Stream audio in chunks asynchronously.
        
        Args:
            text (str): The text to convert to speech
            voice (str, optional): The voice to use. Defaults to provider's default voice.
            chunk_size (int, optional): Size of audio chunks to yield. Defaults to 1024.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
            
        Yields:
            AsyncGenerator[bytes, None]: Audio data chunks
        """
        try:
            import aiofiles
        except ImportError:
            raise ImportError("The 'aiofiles' package is required for async streaming. Install it with 'pip install aiofiles'.")
        
        # Generate the audio file
        audio_file = await self.tts(text, voice=voice, verbose=verbose)
        
        # Stream the file in chunks
        async with aiofiles.open(audio_file, 'rb') as f:
            while chunk := await f.read(chunk_size):
                yield chunk
