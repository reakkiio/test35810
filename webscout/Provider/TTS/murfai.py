import time
import requests
import pathlib
import tempfile
from io import BytesIO
from urllib.parse import urlencode
from webscout import exceptions
from webscout.litagent import LitAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import utils
from .base import BaseTTSProvider


class MurfAITTS(BaseTTSProvider):
    """
    Text-to-speech provider using the MurfAITTS API with OpenAI-compatible interface.
    
    This provider follows the OpenAI TTS API structure with support for:
    - Multiple TTS models (gpt-4o-mini-tts, tts-1, tts-1-hd)
    - Multiple voices with OpenAI-style naming
    - Voice instructions for controlling speech aspects
    - Multiple output formats (mp3, wav, aac, flac, opus, pcm)
    - Streaming support
    """
    
    # Override supported models for MurfAI (set to None as requested)
    SUPPORTED_MODELS = None
    
    # Override supported voices with real MurfAI voice names
    SUPPORTED_VOICES = [
        "Hazel"     # English (UK) female voice
    ]
    
    # Override supported formats
    SUPPORTED_FORMATS = [
        "mp3",    # Default format for MurfAI
        "wav"     # Alternative format
    ]
    
    # Request headers
    headers: dict[str, str] = {
        "User-Agent": LitAgent().random()
    }
    
    # Voice mapping from real names to MurfAI voice IDs
    voice_mapping: dict[str, str] = {
        "Hazel": "en-UK-hazel"
    }

    def __init__(self, timeout: int = 20, proxies: dict = None):
        """Initializes the MurfAITTS TTS client."""
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout

    def tts(self, text: str, voice: str = "Hazel", verbose: bool = False, **kwargs) -> str:
        """
        Converts text to speech using the MurfAITTS API and saves it to a file.
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use (default: "Hazel")
            verbose (bool): Whether to print debug information
            **kwargs: Additional parameters (model, response_format, instructions)
            
        Returns:
            str: Path to the generated audio file
        """
        # Validate input parameters
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        if len(text) > 10000:
            raise ValueError("Input text exceeds maximum allowed length of 10,000 characters")
            
        # Use default voice if not provided
        if voice is None:
            voice = "Hazel"
            
        # Validate voice using base class method
        self.validate_voice(voice)
        
        # Map real voice name to MurfAI voice ID
        voice_id = self.voice_mapping.get(voice, "en-UK-hazel")  # Default to Hazel
        
        # Get response format from kwargs or use default
        response_format = kwargs.get('response_format', 'mp3')
        response_format = self.validate_format(response_format)
        
        # Create temporary file with appropriate extension
        file_extension = f".{response_format}" if response_format != "pcm" else ".wav"
        filename = pathlib.Path(tempfile.mktemp(suffix=file_extension, dir=self.temp_dir))

        # Split text into sentences
        sentences = utils.split_sentences(text)

        # Function to request audio for each chunk
        def generate_audio_for_chunk(part_text: str, part_number: int):
            while True:
                try:
                    params: dict[str, str] = {
                    "name": voice_id,
                    "text": part_text
                    }
                    encode_param: str = urlencode(params)
                    response = self.session.get(f"https://murf.ai/Prod/anonymous-tts/audio?{encode_param}", headers=self.headers, timeout=self.timeout)
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
                    time.sleep(1)
        try:
            # Using ThreadPoolExecutor to handle requests concurrently
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(generate_audio_for_chunk, sentence.strip(), chunk_num): chunk_num
                        for chunk_num, sentence in enumerate(sentences, start=1)}

                # Dictionary to store results with order preserved
                audio_chunks = {}

                for future in as_completed(futures):
                    chunk_num = futures[future]
                    try:
                        part_number, audio_data = future.result()
                        audio_chunks[part_number] = audio_data  # Store the audio data in correct sequence
                    except Exception as e:
                        if verbose:
                            print(f"[debug] Failed to generate audio for chunk {chunk_num}: {e}")

            # Combine audio chunks in the correct sequence
            combined_audio = BytesIO()
            for part_number in sorted(audio_chunks.keys()):
                combined_audio.write(audio_chunks[part_number])
                if verbose:
                    print(f"[debug] Added chunk {part_number} to the combined file.")

            # Save the combined audio data to a single file
            with open(filename, 'wb') as f:
                f.write(combined_audio.getvalue())
            if verbose:
                print(f"[debug] Final Audio Saved as {filename}")
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
        voice: str = "Hazel",
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
            instructions (str): Voice instructions (not used by MurfAI)
            verbose (bool): Whether to print debug information
            
        Returns:
            str: Path to the generated audio file
        """
        return self.tts(
            text=input,
            voice=voice,
            response_format=response_format,
            verbose=verbose
        )

    def stream_audio(
        self,
        input: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "Hazel",
        response_format: str = "mp3",
        instructions: str = None,
        chunk_size: int = 1024,
        verbose: bool = False
    ):
        """
        Stream audio response in chunks.
        
        Args:
            input (str): The text to convert to speech
            model (str): The TTS model to use
            voice (str): The voice to use
            response_format (str): Audio format
            instructions (str): Voice instructions
            chunk_size (int): Size of audio chunks to yield
            verbose (bool): Whether to print debug information
            
        Yields:
            bytes: Audio data chunks
        """
        # Generate the audio file using create_speech
        audio_file = self.create_speech(
            input=input,
            model=model,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            verbose=verbose
        )
        
        # Stream the file in chunks
        with open(audio_file, 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk

# Example usage
if __name__ == "__main__":
    murfai = MurfAITTS()
    text = "This is a test of the MurfAITTS text-to-speech API. It supports multiple sentences and advanced logging."

    print("[debug] Generating audio...")
    try:
        audio_file = murfai.create_speech(
            input=text,
            model="gpt-4o-mini-tts",
            voice="Hazel",
            response_format="mp3",
            verbose=True
        )
        print(f"Audio saved to: {audio_file}")
    except Exception as e:
        print(f"Error: {e}")