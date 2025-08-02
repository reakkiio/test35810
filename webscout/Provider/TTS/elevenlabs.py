##################################################################################
##  ElevenLabs TTS Provider                                                      ##
##################################################################################
import time
import requests
import pathlib
import tempfile
from io import BytesIO
from webscout import exceptions
from webscout.litagent import LitAgent
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class ElevenlabsTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the ElevenLabs API with OpenAI-compatible interface.
    
    This provider follows the OpenAI TTS API structure with support for:
    - Multiple TTS models (gpt-4o-mini-tts, tts-1, tts-1-hd)
    - ElevenLabs' multilingual voices
    - Voice instructions for controlling speech aspects
    - Multiple output formats
    - Streaming support
    """
    
    # Request headers
    headers: dict[str, str] = {
        "User-Agent": LitAgent().random()
    }
    
    # Override supported models for ElevenLabs
    SUPPORTED_MODELS = None
    
    # ElevenLabs voices (mapped to OpenAI-compatible names)
    SUPPORTED_VOICES = [
        "brian",    # nPczCjzI2devNBz1zQrb - Deep male voice
        "alice",    # Xb7hH8MSUJpSbSDYk0k2 - Natural female voice
        "bill",     # pqHfZKP75CvOlQylNhV4 - Warm male voice
        "callum",   # N2lVS1w4EtoT3dr4eOWO - Scottish male voice
        "charlie",  # IKne3meq5aSn9XLyUdCD - American male voice
        "charlotte", # XB0fDUnXU5powFXDhCwa - British female voice
        "chris",    # iP95p4xoKVk53GoZ742B - American male voice
        "daniel",   # onwK4e9ZLuTAKqWW03F9 - British male voice
        "eric",     # cjVigY5qzO86Huf0OWal - American male voice
        "george",   # JBFqnCBsd6RMkjVDRZzb - Raspy male voice
        "jessica",  # cgSgspJ2msm6clMCkdW9 - Warm female voice
        "laura",    # FGY2WhTYpPnrIDTdsKH5 - Soft female voice
        "liam",     # TX3LPaxmHKxFdv7VOQHJ - Irish male voice
        "lily",     # pFZP5JQG7iQjIQuC4Bku - British female voice
        "matilda",  # XrExE9yKIg1WjnnlVkGX - Warm female voice
        "sarah",    # EXAVITQu4vr4xnSDxMaL - Soft female voice
        "will"      # bIHbv24MWmeRgasZH58o - American male voice
    ]
    
    # Voice mapping for ElevenLabs API compatibility
    voice_mapping = {
        "brian": "nPczCjzI2devNBz1zQrb",
        "alice": "Xb7hH8MSUJpSbSDYk0k2",
        "bill": "pqHfZKP75CvOlQylNhV4",
        "callum": "N2lVS1w4EtoT3dr4eOWO",
        "charlie": "IKne3meq5aSn9XLyUdCD",
        "charlotte": "XB0fDUnXU5powFXDhCwa",
        "chris": "iP95p4xoKVk53GoZ742B",
        "daniel": "onwK4e9ZLuTAKqWW03F9",
        "eric": "cjVigY5qzO86Huf0OWal",
        "george": "JBFqnCBsd6RMkjVDRZzb",
        "jessica": "cgSgspJ2msm6clMCkdW9",
        "laura": "FGY2WhTYpPnrIDTdsKH5",
        "liam": "TX3LPaxmHKxFdv7VOQHJ",
        "lily": "pFZP5JQG7iQjIQuC4Bku",
        "matilda": "XrExE9yKIg1WjnnlVkGX",
        "sarah": "EXAVITQu4vr4xnSDxMaL",
        "will": "bIHbv24MWmeRgasZH58o"
    }
    
    # Legacy voice mapping for backward compatibility
    all_voices: dict[str, str] = {
        "Brian": "nPczCjzI2devNBz1zQrb", "Alice": "Xb7hH8MSUJpSbSDYk0k2", "Bill": "pqHfZKP75CvOlQylNhV4",
        "Callum": "N2lVS1w4EtoT3dr4eOWO", "Charlie": "IKne3meq5aSn9XLyUdCD", "Charlotte": "XB0fDUnXU5powFXDhCwa",
        "Chris": "iP95p4xoKVk53GoZ742B", "Daniel": "onwK4e9ZLuTAKqWW03F9", "Eric": "cjVigY5qzO86Huf0OWal",
        "George": "JBFqnCBsd6RMkjVDRZzb", "Jessica": "cgSgspJ2msm6clMCkdW9", "Laura": "FGY2WhTYpPnrIDTdsKH5",
        "Liam": "TX3LPaxmHKxFdv7VOQHJ", "Lily": "pFZP5JQG7iQjIQuC4Bku", "Matilda": "XrExE9yKIg1WjnnlVkGX",
        "Sarah": "EXAVITQu4vr4xnSDxMaL", "Will": "bIHbv24MWmeRgasZH58o"
    }

    def __init__(self, timeout: int = 20, proxies: dict = None):
        """
        Initialize the ElevenLabs TTS client.
        
        Args:
            timeout (int): Request timeout in seconds
            proxies (dict): Proxy configuration
        """
        super().__init__()
        self.api_url = "https://api.elevenlabs.io/v1/text-to-speech"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout
        self.params = {'allow_unauthenticated': '1'}
        self.default_voice = "brian"

    def tts(
        self, 
        text: str, 
        model: str = "gpt-4o-mini-tts",
        voice: str = "brian", 
        response_format: str = "mp3",
        instructions: str = None, 
        verbose: bool = True
    ) -> str:
        """
        Convert text to speech using ElevenLabs API with OpenAI-compatible parameters.

        Args:
            text (str): The text to convert to speech (max 10,000 characters)
            model (str): The TTS model to use (gpt-4o-mini-tts, tts-1, tts-1-hd)
            voice (str): The voice to use for TTS (brian, alice, bill, etc.)
            response_format (str): Audio format (mp3, opus, aac, flac, wav, pcm)
            instructions (str): Voice instructions (not used by ElevenLabs but kept for compatibility)
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
        
        # Map voice to ElevenLabs API format
        elevenlabs_voice = self.voice_mapping.get(voice, voice)
        
        # Create temporary file with appropriate extension
        file_extension = f".{response_format}" if response_format != "pcm" else ".wav"
        filename = pathlib.Path(tempfile.mktemp(suffix=file_extension, dir=self.temp_dir))

        # Split text into sentences using the utils module for better processing
        sentences = utils.split_sentences(text)
        if verbose:
            print(f"[debug] Processing {len(sentences)} sentences")
            print(f"[debug] Model: {model}")
            print(f"[debug] Voice: {voice} -> {elevenlabs_voice}")
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
                    json_data = {
                        'text': part_text, 
                        'model_id': 'eleven_multilingual_v2',
                        # Add model parameter for future ElevenLabs API compatibility
                        'tts_model': model
                    }
                    response = self.session.post(
                        url=f'{self.api_url}/{elevenlabs_voice}',
                        params=self.params,
                        headers=self.headers,
                        json=json_data,
                        timeout=self.timeout
                    )
                    response.raise_for_status()

                    # Check if the request was successful
                    if response.ok and response.status_code == 200:
                        if verbose:
                            print(f"[debug] Chunk {part_number} processed successfully")
                        return part_number, response.content
                    else:
                        if verbose:
                            print(f"[debug] No data received for chunk {part_number}. Retrying...")

                except requests.RequestException as e:
                    if verbose:
                        print(f"[debug] Error for chunk {part_number}: {e}. Retrying...")
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
                        audio_chunks[part_number] = audio_data  # Store the audio data in correct sequence
                    except Exception as e:
                        raise exceptions.FailedToGenerateResponseError(f"Failed to generate audio for chunk {chunk_num}: {str(e)}")

                # Combine all audio chunks in order
                combined_audio = BytesIO()
                for part_number in sorted(audio_chunks.keys()):
                    combined_audio.write(audio_chunks[part_number])
                    if verbose:
                        print(f"[debug] Added chunk {part_number} to the combined file.")

                # Save the combined audio data to a single file
                with open(filename, 'wb') as f:
                    f.write(combined_audio.getvalue())
                    
                if verbose:
                    print(f"[debug] Speech generated successfully")
                    print(f"[debug] Audio saved to {filename}")
                    
                return filename.as_posix()

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"[debug] Failed to perform the operation: {e}")
            raise exceptions.FailedToGenerateResponseError(
                f"Failed to perform the operation: {e}"
            )

    def create_speech(
        self,
        input: str,
        model: str = "gpt-4o-mini-tts", 
        voice: str = "brian",
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
            instructions (str): Voice instructions (not used by ElevenLabs)
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
    
    def __init__(self, tts_provider: ElevenlabsTTS):
        self.tts_provider = tts_provider
        self.audio_file = None
    
    def create(
        self,
        input: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "brian", 
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
    elevenlabs = ElevenlabsTTS()
    
    try:
        # Basic usage
        print("Testing basic speech generation...")
        audio_file = elevenlabs.create_speech(
            input="Today is a wonderful day to build something people love!",
            model="gpt-4o-mini-tts",
            voice="brian",
            instructions="Speak in a cheerful and positive tone."
        )
        print(f"Audio file generated: {audio_file}")
        
        # Streaming usage
        print("\nTesting streaming response...")
        with elevenlabs.with_streaming_response().create(
            input="This is a streaming test with ElevenLabs voices.",
            voice="alice",
            response_format="wav"
        ) as response:
            response.stream_to_file("elevenlabs_streaming_test.wav")
            print("Streaming audio saved to elevenlabs_streaming_test.wav")
            
        # Legacy compatibility test
        print("\nTesting legacy voice compatibility...")
        legacy_audio = elevenlabs.tts("Testing legacy voice format.", voice="brian")
        print(f"Legacy audio file generated: {legacy_audio}")
        
    except exceptions.FailedToGenerateResponseError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")