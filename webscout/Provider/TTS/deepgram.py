##################################################################################
##  Deepgram TTS Provider                                                      ##
##################################################################################
import time
import requests
import pathlib
import base64
import tempfile
from io import BytesIO
from webscout import exceptions
from concurrent.futures import ThreadPoolExecutor, as_completed
from webscout.litagent import LitAgent
try:
    from . import utils
    from .base import BaseTTSProvider
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from webscout.Provider.TTS import utils
    from webscout.Provider.TTS.base import BaseTTSProvider

class DeepgramTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the Deepgram API with OpenAI-compatible interface.
    
    This provider follows the OpenAI TTS API structure with support for:
    - Multiple TTS models (gpt-4o-mini-tts, tts-1, tts-1-hd)
    - Deepgram's Aura voices optimized for English
    - Voice instructions for controlling speech aspects
    - Multiple output formats
    - Streaming support
    """
    
    # Request headers
    headers: dict[str, str] = {
        "User-Agent": LitAgent().random()
    }
    
    # Override supported models for Deepgram
    SUPPORTED_MODELS = None
    
    # Deepgram Aura voices (mapped to OpenAI-compatible names)
    SUPPORTED_VOICES = [
        "asteria",   # aura-asteria-en - Clear and articulate female voice
        "arcas",     # aura-arcas-en - Warm male voice
        "luna",      # aura-luna-en - Gentle female voice
        "zeus",      # aura-zeus-en - Authoritative male voice
        "orpheus",   # aura-orpheus-en - Melodic male voice
        "angus",     # aura-angus-en - Scottish-accented male voice
        "athena",    # aura-athena-en - Professional female voice
        "helios",    # aura-helios-en - Bright male voice
        "hera",      # aura-hera-en - Mature female voice
        "orion",     # aura-orion-en - Deep male voice
        "perseus",   # aura-perseus-en - Young male voice
        "stella"     # aura-stella-en - Friendly female voice
    ]
    
    # Voice mapping for Deepgram API compatibility
    voice_mapping = {
        "asteria": "aura-asteria-en",
        "arcas": "aura-arcas-en", 
        "luna": "aura-luna-en",
        "zeus": "aura-zeus-en",
        "orpheus": "aura-orpheus-en",
        "angus": "aura-angus-en",
        "athena": "aura-athena-en",
        "helios": "aura-helios-en",
        "hera": "aura-hera-en",
        "orion": "aura-orion-en",
        "perseus": "aura-perseus-en",
        "stella": "aura-stella-en"
    }
    
    # Legacy voice mapping for backward compatibility
    all_voices: dict[str, str] = {
        "Asteria": "aura-asteria-en", "Arcas": "aura-arcas-en", "Luna": "aura-luna-en",
        "Zeus": "aura-zeus-en", "Orpheus": "aura-orpheus-en", "Angus": "aura-angus-en",
        "Athena": "aura-athena-en", "Helios": "aura-helios-en", "Hera": "aura-hera-en",
        "Orion": "aura-orion-en", "Perseus": "aura-perseus-en", "Stella": "aura-stella-en"
    }

    def __init__(self, timeout: int = 20, proxies: dict = None):
        """
        Initialize the Deepgram TTS client.
        
        Args:
            timeout (int): Request timeout in seconds
            proxies (dict): Proxy configuration
        """
        super().__init__()
        self.api_url = "https://deepgram.com/api/ttsAudioGeneration"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout
        # Override defaults for Deepgram
        self.default_voice = "asteria"

    def tts(
        self, 
        text: str, 
        model: str = "gpt-4o-mini-tts",
        voice: str = "asteria", 
        response_format: str = "mp3",
        instructions: str = None, 
        verbose: bool = True
    ) -> str:
        """
        Convert text to speech using Deepgram API with OpenAI-compatible parameters.

        Args:
            text (str): The text to convert to speech (max 10,000 characters)
            model (str): The TTS model to use (gpt-4o-mini-tts, tts-1, tts-1-hd)
            voice (str): The voice to use for TTS (asteria, arcas, luna, zeus, etc.)
            response_format (str): Audio format (mp3, opus, aac, flac, wav, pcm)
            instructions (str): Voice instructions (not used by Deepgram but kept for compatibility)
            verbose (bool): Whether to print debug information

        Returns:
            str: Path to the generated audio file

        Raises:
            ValueError: If input parameters are invalid
            exceptions.FailedToGenerateResponseError: If there is an error generating or saving the audio
        """
        # Validate input parameters
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        if len(text) > 10000:
            raise ValueError("Input text exceeds maximum allowed length of 10,000 characters")
            
        # Validate model, voice, and format using base class methods
        model = self.validate_model(model)
        voice = self.validate_voice(voice)
        response_format = self.validate_format(response_format)
        
        # Map voice to Deepgram API format
        deepgram_voice = self.voice_mapping.get(voice, voice)
        
        # Create temporary file with appropriate extension
        file_extension = f".{response_format}" if response_format != "pcm" else ".wav"
        filename = pathlib.Path(tempfile.mktemp(suffix=file_extension, dir=self.temp_dir))

        # Split text into sentences using the utils module for better processing
        sentences = utils.split_sentences(text)
        if verbose:
            print(f"[debug] Processing {len(sentences)} sentences")
            print(f"[debug] Model: {model}")
            print(f"[debug] Voice: {voice} -> {deepgram_voice}")
            print(f"[debug] Format: {response_format}")

        def generate_audio_for_chunk(part_text: str, part_number: int):
            """
            Generate audio for a single chunk of text.

            Args:
                part_text (str): The text chunk to convert
                part_number (int): The chunk number for ordering

            Returns:
                tuple: (part_number, audio_data)

            Raises:
                requests.RequestException: If there's an API error
            """
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    payload = {
                        "text": part_text, 
                        "model": deepgram_voice,
                        # Add model parameter for future Deepgram API compatibility
                        "tts_model": model
                    }
                    response = self.session.post(
                        url=self.api_url,
                        headers=self.headers,
                        json=payload,
                        stream=True,
                        timeout=self.timeout
                    )
                    response.raise_for_status()

                    response_data = response.json().get('data')
                    if response_data:
                        audio_data = base64.b64decode(response_data)
                        if verbose:
                            print(f"[debug] Chunk {part_number} processed successfully")
                        return part_number, audio_data

                    if verbose:
                        print(f"[debug] No data received for chunk {part_number}. Attempt {retry_count + 1}/{max_retries}")

                except requests.RequestException as e:
                    if verbose:
                        print(f"[debug] Error processing chunk {part_number}: {str(e)}. Attempt {retry_count + 1}/{max_retries}")
                    if retry_count == max_retries - 1:
                        raise exceptions.FailedToGenerateResponseError(f"Failed to generate audio for chunk {part_number}: {str(e)}")

                retry_count += 1
                time.sleep(1)

            raise exceptions.FailedToGenerateResponseError(f"Failed to generate audio for chunk {part_number} after {max_retries} attempts")

        try:
            # Using ThreadPoolExecutor to handle requests concurrently
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(generate_audio_for_chunk, sentence.strip(), chunk_num): chunk_num
                    for chunk_num, sentence in enumerate(sentences, start=1)
                }

                # Dictionary to store results with order preserved
                audio_chunks = {}

                for future in as_completed(futures):
                    chunk_num = futures[future]
                    try:
                        part_number, audio_data = future.result()
                        audio_chunks[part_number] = audio_data
                    except Exception as e:
                        raise exceptions.FailedToGenerateResponseError(f"Failed to generate audio for chunk {chunk_num}: {str(e)}")

                # Combine all audio chunks in order
                with open(filename, 'wb') as f:
                    for chunk_num in sorted(audio_chunks.keys()):
                        f.write(audio_chunks[chunk_num])

                if verbose:
                    print(f"[debug] Speech generated successfully")
                    print(f"[debug] Audio saved to {filename}")
                    
                return str(filename)

        except Exception as e:
            if verbose:
                print(f"[debug] Failed to generate audio: {str(e)}")
            raise exceptions.FailedToGenerateResponseError(f"Failed to generate audio: {str(e)}")

    def create_speech(
        self,
        input: str,
        model: str = "gpt-4o-mini-tts", 
        voice: str = "asteria",
        response_format: str = "mp3",
        instructions: str = None,
        verbose: bool = False
    ) -> str:
        """
        OpenAI-compatible speech creation interface.
        
        Args:
            input (str): The text to convert to speech
            model (str): The TTS model to use
            voice (str): The voice to use
            response_format (str): Audio format
            instructions (str): Voice instructions (not used by Deepgram)
            verbose (bool): Whether to print debug information
            
        Returns:
            str: Path to the generated audio file
        """
        return self.tts(
            text=input,
            model=model,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            verbose=verbose
        )

    def with_streaming_response(self):
        """
        Return a streaming response context manager (OpenAI-compatible).
        
        Returns:
            StreamingResponseContextManager: Context manager for streaming responses
        """
        return StreamingResponseContextManager(self)


class StreamingResponseContextManager:
    """
    Context manager for streaming TTS responses (OpenAI-compatible).
    """
    
    def __init__(self, tts_provider: DeepgramTTS):
        self.tts_provider = tts_provider
        self.audio_file = None
    
    def create(
        self,
        input: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "asteria", 
        response_format: str = "mp3",
        instructions: str = None
    ):
        """
        Create speech with streaming capability.
        
        Args:
            input (str): The text to convert to speech
            model (str): The TTS model to use
            voice (str): The voice to use
            response_format (str): Audio format
            instructions (str): Voice instructions
            
        Returns:
            StreamingResponse: Streaming response object
        """
        self.audio_file = self.tts_provider.create_speech(
            input=input,
            model=model,
            voice=voice,
            response_format=response_format,
            instructions=instructions
        )
        return StreamingResponse(self.audio_file)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class StreamingResponse:
    """
    Streaming response object for TTS audio (OpenAI-compatible).
    """
    
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        pass
    
    def stream_to_file(self, file_path: str, chunk_size: int = 1024):
        """
        Stream audio content to a file.
        
        Args:
            file_path (str): Destination file path
            chunk_size (int): Size of chunks to read/write
        """
        import shutil
        shutil.copy2(self.audio_file, file_path)
    
    def iter_bytes(self, chunk_size: int = 1024):
        """
        Iterate over audio bytes in chunks.
        
        Args:
            chunk_size (int): Size of chunks to yield
            
        Yields:
            bytes: Audio data chunks
        """
        with open(self.audio_file, 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk


# Example usage
if __name__ == "__main__":
    # Example usage demonstrating OpenAI-compatible interface
    deepgram = DeepgramTTS()
    
    try:
        # Basic usage
        print("Testing basic speech generation...")
        audio_file = deepgram.create_speech(
            input="Today is a wonderful day to build something people love!",
            model="gpt-4o-mini-tts",
            voice="asteria",
            instructions="Speak in a cheerful and positive tone."
        )
        print(f"Audio file generated: {audio_file}")
        
        # Streaming usage
        print("\nTesting streaming response...")
        with deepgram.with_streaming_response().create(
            input="This is a streaming test with Deepgram Aura voices.",
            voice="luna",
            response_format="wav"
        ) as response:
            response.stream_to_file("deepgram_streaming_test.wav")
            print("Streaming audio saved to deepgram_streaming_test.wav")
            
        # Legacy compatibility test
        print("\nTesting legacy voice compatibility...")
        legacy_audio = deepgram.tts("Testing legacy voice format.", voice="Asteria")
        print(f"Legacy audio file generated: {legacy_audio}")
        
    except exceptions.FailedToGenerateResponseError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")