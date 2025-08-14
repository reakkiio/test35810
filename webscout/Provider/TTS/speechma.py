##################################################################################
##  Modified version of code written by t.me/infip1217                          ##
##################################################################################
import time
import requests
import pathlib
import tempfile
from io import BytesIO
from webscout import exceptions
from webscout.litagent import LitAgent
from webscout.Litlogger import Logger, LogLevel
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

class SpeechMaTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the SpeechMa API with OpenAI-compatible interface.
    
    This provider follows the OpenAI TTS API structure with support for:
    - Multiple TTS models (gpt-4o-mini-tts, tts-1, tts-1-hd)
    - Multilingual voices with pitch and rate control
    - Voice instructions for controlling speech aspects
    - Multiple output formats
    - Streaming support
    """
    
    # Request headers
    headers = {
        "authority": "speechma.com",
        "origin": "https://speechma.com",
        "referer": "https://speechma.com/",
        "content-type": "application/json",
        **LitAgent().generate_fingerprint()
    }
    
    # SpeechMa doesn't support different models - set to None
    SUPPORTED_MODELS = None
    
    # All supported voices from SpeechMa API
    SUPPORTED_VOICES = [
        "aditi", "amy", "astrid", "bianca", "carla", "carmen", "celine", "chant",
        "conchita", "cristiano", "dora", "enrique", "ewa", "filiz", "geraint",
        "giorgio", "gwyneth", "hans", "ines", "ivy", "jacek", "jan", "joanna",
        "joey", "justin", "karl", "kendra", "kimberly", "lea", "liv", "lotte",
        "lucia", "lupe", "mads", "maja", "marlene", "mathieu", "matthew", "maxim",
        "mia", "miguel", "mizuki", "naja", "nicole", "penelope", "raveena",
        "ricardo", "ruben", "russell", "salli", "seoyeon", "takumi", "tatyana",
        "vicki", "vitoria", "zeina", "zhiyu", "aditi-neural", "amy-neural",
        "aria-neural", "ayanda-neural", "brian-neural", "emma-neural",
        "jenny-neural", "joey-neural", "justin-neural", "kendra-neural",
        "kimberly-neural", "matthew-neural", "olivia-neural", "ruth-neural",
        "salli-neural", "stephen-neural", "suvi-neural", "camila-neural",
        "lupe-neural", "pedro-neural", "natasha-neural", "william-neural",
        "clara-neural", "liam-neural", "libby-neural", "maisie-neural",
        "ryan-neural", "sonia-neural", "thomas-neural", "aria-multilingual",
        "andrew-multilingual", "brian-multilingual", "emma-multilingual",
        "jenny-multilingual", "ryan-multilingual", "adam-multilingual",
        "liam-multilingual", "aria-turbo", "andrew-turbo", "brian-turbo",
        "emma-turbo", "jenny-turbo", "ryan-turbo", "adam-turbo", "liam-turbo",
        "aria-hd", "andrew-hd", "brian-hd", "emma-hd", "jenny-hd", "andrew-hd-2",
        "aria-hd-2", "adam-hd", "ava-hd", "davis-hd", "brian-hd-2",
        "christopher-hd", "coral-hd", "emma-hd-2", "eric-hd", "fable-hd",
        "jenny-hd-2", "michelle-hd", "roger-hd", "sage-hd", "vale-hd", "verse-hd",
        # Legacy voice names for backward compatibility
        "emma", "ava", "brian", "andrew", "aria", "christopher", "eric", "jenny",
        "michelle", "roger", "libby", "ryan", "sonia", "thomas", "natasha",
        "william", "clara", "liam"
    ]
    
    # Voice mapping for SpeechMa API compatibility (lowercase keys for all voices)
    voice_mapping = {
        # Standard voices
        "aditi": "voice-1",
        "amy": "voice-2",
        "astrid": "voice-3",
        "bianca": "voice-4",
        "carla": "voice-5",
        "carmen": "voice-6",
        "celine": "voice-7",
        "chant": "voice-8",
        "conchita": "voice-9",
        "cristiano": "voice-10",
        "dora": "voice-11",
        "enrique": "voice-12",
        "ewa": "voice-13",
        "filiz": "voice-14",
        "geraint": "voice-15",
        "giorgio": "voice-16",
        "gwyneth": "voice-17",
        "hans": "voice-18",
        "ines": "voice-19",
        "ivy": "voice-20",
        "jacek": "voice-21",
        "jan": "voice-22",
        "joanna": "voice-23",
        "joey": "voice-24",
        "justin": "voice-25",
        "karl": "voice-26",
        "kendra": "voice-27",
        "kimberly": "voice-28",
        "lea": "voice-29",
        "liv": "voice-30",
        "lotte": "voice-31",
        "lucia": "voice-32",
        "lupe": "voice-33",
        "mads": "voice-34",
        "maja": "voice-35",
        "marlene": "voice-36",
        "mathieu": "voice-37",
        "matthew": "voice-38",
        "maxim": "voice-39",
        "mia": "voice-40",
        "miguel": "voice-41",
        "mizuki": "voice-42",
        "naja": "voice-43",
        "nicole": "voice-44",
        "penelope": "voice-45",
        "raveena": "voice-46",
        "ricardo": "voice-47",
        "ruben": "voice-48",
        "russell": "voice-49",
        "salli": "voice-50",
        "seoyeon": "voice-51",
        "takumi": "voice-52",
        "tatyana": "voice-53",
        "vicki": "voice-54",
        "vitoria": "voice-55",
        "zeina": "voice-56",
        "zhiyu": "voice-57",
        # Neural voices
        "aditi-neural": "voice-58",
        "amy-neural": "voice-59",
        "aria-neural": "voice-60",
        "ayanda-neural": "voice-61",
        "brian-neural": "voice-62",
        "emma-neural": "voice-63",
        "jenny-neural": "voice-64",
        "joey-neural": "voice-65",
        "justin-neural": "voice-66",
        "kendra-neural": "voice-67",
        "kimberly-neural": "voice-68",
        "matthew-neural": "voice-69",
        "olivia-neural": "voice-70",
        "ruth-neural": "voice-71",
        "salli-neural": "voice-72",
        "stephen-neural": "voice-73",
        "suvi-neural": "voice-74",
        "camila-neural": "voice-75",
        "lupe-neural": "voice-76",
        "pedro-neural": "voice-77",
        "natasha-neural": "voice-78",
        "william-neural": "voice-79",
        "clara-neural": "voice-80",
        "liam-neural": "voice-81",
        "libby-neural": "voice-82",
        "maisie-neural": "voice-83",
        "ryan-neural": "voice-84",
        "sonia-neural": "voice-85",
        "thomas-neural": "voice-86",
        # Multilingual voices
        "aria-multilingual": "voice-87",
        "andrew-multilingual": "voice-88",
        "brian-multilingual": "voice-89",
        "emma-multilingual": "voice-90",
        "jenny-multilingual": "voice-91",
        "ryan-multilingual": "voice-92",
        "adam-multilingual": "voice-93",
        "liam-multilingual": "voice-94",
        # Turbo voices
        "aria-turbo": "voice-95",
        "andrew-turbo": "voice-96",
        "brian-turbo": "voice-97",
        "emma-turbo": "voice-98",
        "jenny-turbo": "voice-99",
        "ryan-turbo": "voice-100",
        "adam-turbo": "voice-101",
        "liam-turbo": "voice-102",
        # HD voices
        "aria-hd": "voice-103",
        "andrew-hd": "voice-104",
        "brian-hd": "voice-105",
        "emma-hd": "voice-106",
        "jenny-hd": "voice-107",
        "andrew-hd-2": "voice-108",
        "aria-hd-2": "voice-109",
        "adam-hd": "voice-110",
        "ava-hd": "voice-111",
        "davis-hd": "voice-112",
        "brian-hd-2": "voice-113",
        "christopher-hd": "voice-114",
        "coral-hd": "voice-115",
        "emma-hd-2": "voice-116",
        "eric-hd": "voice-117",
        "fable-hd": "voice-118",
        "jenny-hd-2": "voice-119",
        "michelle-hd": "voice-120",
        "roger-hd": "voice-121",
        "sage-hd": "voice-122",
        "vale-hd": "voice-123",
        "verse-hd": "voice-124",
        # Legacy compatibility mappings (lowercase)
        "emma": "voice-116",
        "ava": "voice-111",
        "brian": "voice-113",
        "andrew": "voice-108",
        "aria": "voice-109",
        "christopher": "voice-114",
        "eric": "voice-117",
        "jenny": "voice-119",
        "michelle": "voice-120",
        "roger": "voice-121",
        "libby": "voice-82",
        "ryan": "voice-84",
        "sonia": "voice-85",
        "thomas": "voice-86",
        "natasha": "voice-78",
        "william": "voice-79",
        "clara": "voice-80",
        "liam": "voice-81"
    }

    # Legacy voice mapping for backward compatibility
    all_voices = voice_mapping

    def __init__(self, timeout: int = 20, proxies: dict = None):
        """
        Initialize the SpeechMa TTS client.
        
        Args:
            timeout (int): Request timeout in seconds
            proxies (dict): Proxy configuration
        """
        super().__init__()
        self.api_url = "https://speechma.com/com.api/tts-api.php"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout
        self.logger = Logger(name="SpeechMaTTS", level=LogLevel.INFO)
        # Override defaults for SpeechMa
        self.default_voice = "emma"
        self.default_model = "gpt-4o-mini-tts"

    def create_speech(
        self,
        input: str,
        voice: str = "emma",
        model: str = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str = None,
        **kwargs
    ) -> bytes:
        """
        Create speech from text using OpenAI-compatible interface.
        
        Args:
            input (str): The text to convert to speech
            voice (str): Voice to use for generation
            model (str): TTS model to use
            response_format (str): Audio format (mp3, opus, aac, flac, wav, pcm)
            speed (float): Speed of speech (0.25 to 4.0)
            instructions (str): Voice instructions (not used by SpeechMa)
            **kwargs: Additional parameters (pitch, rate for SpeechMa compatibility)
            
        Returns:
            bytes: Audio data
            
        Raises:
            ValueError: If input parameters are invalid
            exceptions.FailedToGenerateResponseError: If generation fails
        """
        # Validate parameters
        if not input or not isinstance(input, str):
            raise ValueError("Input text must be a non-empty string")
        if len(input) > 10000:
            raise ValueError("Input text exceeds maximum allowed length of 10,000 characters")
            
        model = self.validate_model(model or self.default_model)
        voice = self.validate_voice(voice)
        response_format = self.validate_format(response_format)
        
        # Convert speed to SpeechMa rate parameter
        rate = int((speed - 1.0) * 10)  # Convert 0.25-4.0 to -7.5 to 30, clamp to -10 to 10
        rate = max(-10, min(10, rate))
        
        # Extract SpeechMa-specific parameters
        pitch = kwargs.get('pitch', 0)
        
        # Map voice to SpeechMa format
        speechma_voice = self.voice_mapping.get(voice, self.all_voices.get(voice.title(), "voice-116"))
        
        # Prepare payload
        payload = {
            "text": input,
            "voice": speechma_voice,
            "pitch": pitch,
            "rate": rate,
            "volume": 100
        }
        
        try:
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Validate audio response
            content_type = response.headers.get('content-type', '').lower()
            if ('audio' in content_type or 
                response.content.startswith(b'\xff\xfb') or 
                response.content.startswith(b'ID3') or 
                b'LAME' in response.content[:100]):
                return response.content
            else:
                raise exceptions.FailedToGenerateResponseError(
                    f"Unexpected response format. Content-Type: {content_type}"
                )
                
        except requests.exceptions.RequestException as e:
            raise exceptions.FailedToGenerateResponseError(f"API request failed: {e}")

    def with_streaming_response(self):
        """
        Return a context manager for streaming responses.
        
        Returns:
            SpeechMaStreamingResponse: Context manager for streaming
        """
        return SpeechMaStreamingResponse(self)

    def tts(
        self, 
        text: str, 
        model: str = None,
        voice: str = "emma", 
        response_format: str = "mp3",
        instructions: str = None,
        pitch: int = 0, 
        rate: int = 0, 
        verbose: bool = True
    ) -> str:
        """
        Convert text to speech using SpeechMa API with OpenAI-compatible parameters.

        Args:
            text (str): The text to convert to speech (max 10,000 characters)
            model (str): The TTS model to use (gpt-4o-mini-tts, tts-1, tts-1-hd)
            voice (str): The voice to use for TTS (emma, ava, brian, etc.)
            response_format (str): Audio format (mp3, opus, aac, flac, wav, pcm)
            instructions (str): Voice instructions (not used by SpeechMa but kept for compatibility)
            pitch (int): Voice pitch adjustment (-10 to 10, default: 0)
            rate (int): Voice rate/speed adjustment (-10 to 10, default: 0)
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
        model = self.validate_model(model or self.default_model)
        voice = self.validate_voice(voice)
        response_format = self.validate_format(response_format)
        
        # Map voice to SpeechMa API format
        speechma_voice = self.voice_mapping.get(voice, voice)
        if speechma_voice not in self.all_voices.values():
            # Fallback to legacy voice mapping
            speechma_voice = self.all_voices.get(voice.title(), self.all_voices.get("Emma", "voice-116"))
        
        # Create temporary file with appropriate extension
        file_extension = f".{response_format}" if response_format != "pcm" else ".wav"
        filename = pathlib.Path(tempfile.mktemp(suffix=file_extension, dir=self.temp_dir))

        # Split text into sentences using the utils module for better processing
        sentences = utils.split_sentences(text)
        if verbose:
            print(f"[debug] Processing {len(sentences)} sentences")
            print(f"[debug] Model: {model}")
            print(f"[debug] Voice: {voice} -> {speechma_voice}")
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
                        "voice": speechma_voice,
                        "pitch": pitch,
                        "rate": rate,
                        "volume": 100,
                        # Add model parameter for future SpeechMa API compatibility
                        "tts_model": model
                    }
                    response = self.session.post(
                        url=self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()

                    # Check if response is audio data
                    content_type = response.headers.get('content-type', '').lower()
                    if ('audio' in content_type or 
                        response.content.startswith(b'\xff\xfb') or 
                        response.content.startswith(b'ID3') or 
                        b'LAME' in response.content[:100]):
                        if verbose:
                            print(f"[debug] Chunk {part_number} processed successfully")
                        return part_number, response.content
                    else:
                        raise exceptions.FailedToGenerateResponseError(
                            f"Unexpected response format. Content-Type: {content_type}"
                        )

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise exceptions.FailedToGenerateResponseError(
                            f"Failed to generate audio for chunk {part_number} after {max_retries} retries: {e}"
                        )
                    if verbose:
                        print(f"[debug] Retrying chunk {part_number} (attempt {retry_count + 1})")
                    time.sleep(1)  # Brief delay before retry

        # Process chunks concurrently for better performance
        audio_chunks = []
        if len(sentences) > 1:
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_chunk = {
                    executor.submit(generate_audio_for_chunk, sentence, i): i
                    for i, sentence in enumerate(sentences)
                }
                
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_number, audio_data = future.result()
                        audio_chunks.append((chunk_number, audio_data))
                    except Exception as e:
                        if verbose:
                            print(f"[debug] Error processing chunk: {e}")
                        raise
        else:
            # Single sentence, process directly
            chunk_number, audio_data = generate_audio_for_chunk(sentences[0], 0)
            audio_chunks.append((chunk_number, audio_data))

        # Sort chunks by their original order and combine
        audio_chunks.sort(key=lambda x: x[0])
        combined_audio = b''.join([chunk[1] for chunk in audio_chunks])

        # Save combined audio to file
        try:
            with open(filename, 'wb') as f:
                f.write(combined_audio)
            if verbose:
                print(f"[debug] Audio saved to: {filename}")
            return filename.as_posix()
        except IOError as e:
            raise exceptions.FailedToGenerateResponseError(f"Failed to save audio file: {e}")


class SpeechMaStreamingResponse:
    """Context manager for streaming SpeechMa TTS responses."""
    
    def __init__(self, client: SpeechMaTTS):
        self.client = client
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def create_speech(
        self,
        input: str,
        voice: str = "emma",
        model: str = "gpt-4o-mini-tts",
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str = None,
        **kwargs
    ):
        """
        Create speech with streaming response simulation.
        
        Note: SpeechMa doesn't support true streaming, so this returns
        the complete audio data wrapped in a BytesIO object.
        
        Args:
            input (str): Text to convert to speech
            voice (str): Voice to use
            model (str): TTS model
            response_format (str): Audio format
            speed (float): Speech speed
            instructions (str): Voice instructions
            **kwargs: Additional parameters
            
        Returns:
            BytesIO: Audio data stream
        """
        audio_data = self.client.create_speech(
            input=input,
            voice=voice,
            model=model,
            response_format=response_format,
            speed=speed,
            instructions=instructions,
            **kwargs
        )
        return BytesIO(audio_data)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the SpeechMa TTS client
    speechma = SpeechMaTTS()
    
    # Example 1: Basic usage with legacy method
    print("=== Example 1: Basic TTS ===")
    text = "Hello, this is a test of the SpeechMa text-to-speech API."
    try:
        audio_file = speechma.tts(text, voice="emma", verbose=True)
        print(f"Audio saved to: {audio_file}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: OpenAI-compatible interface
    print("\n=== Example 2: OpenAI-compatible interface ===")
    try:
        audio_data = speechma.create_speech(
            input="This demonstrates the OpenAI-compatible interface.",
            voice="brian",
            model="tts-1-hd",
            response_format="mp3",
            speed=1.2
        )
        print(f"Generated {len(audio_data)} bytes of audio data")
        
        # Save to file
        with open("openai_compatible_test.mp3", "wb") as f:
            f.write(audio_data)
        print("Audio saved to: openai_compatible_test.mp3")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Streaming response context manager
    print("\n=== Example 3: Streaming response ===")
    try:
        with speechma.with_streaming_response() as streaming:
            audio_stream = streaming.create_speech(
                input="This demonstrates streaming response handling.",
                voice="aria",
                model="gpt-4o-mini-tts"
            )
            audio_data = audio_stream.read()
            print(f"Streamed {len(audio_data)} bytes of audio data")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Voice and model validation
    print("\n=== Example 4: Parameter validation ===")
    try:
        # Test supported voices
        print("Supported voices:", speechma.SUPPORTED_VOICES[:5], "...")
        print("Supported models:", speechma.SUPPORTED_MODELS)
        
        # Test with different parameters
        audio_file = speechma.tts(
            text="Testing different voice parameters.",
            voice="christopher",
            model="tts-1",
            pitch=2,
            rate=-1,
            verbose=True
        )
        print(f"Audio with custom parameters saved to: {audio_file}")
    except Exception as e:
        print(f"Error: {e}")