##################################################################################
##  FreeTTS Provider                                                             ##
##################################################################################
import os
import requests
from datetime import datetime
from webscout.Provider.TTS.base import BaseTTSProvider
from webscout.litagent import LitAgent


class FreeTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the FreeTTS API with OpenAI-compatible interface.
    
    This provider follows the OpenAI TTS API structure with support for:
    - Multiple TTS models (gpt-4o-mini-tts, tts-1, tts-1-hd)
    - Dynamic voice loading based on language
    - Voice instructions for controlling speech aspects
    - Multiple output formats
    - Streaming support
    """
    
    headers = {
        "accept": "*/*",
        "accept-language": "ru-RU,ru;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "sec-ch-ua": '"Brave";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "User-Agent": LitAgent().random()
    }
    
    # Override supported models for FreeTTS
    SUPPORTED_MODELS = None
    
    # Override supported voices (will be loaded dynamically)
    SUPPORTED_VOICES = []
    
    # Override supported formats
    SUPPORTED_FORMATS = [
        "mp3",    # Default format for FreeTTS
        "wav",    # Alternative format
        "aac"     # Additional format support
    ]

    def __init__(self, lang="ru-RU", timeout: int = 30, proxies: dict = None):
        """
        Initialize the FreeTTS TTS client.
        
        Args:
            lang (str): Language code for voice selection
            timeout (int): Request timeout in seconds
            proxies (dict): Proxy configuration
        """
        super().__init__()
        self.lang = lang
        self.url = "https://freetts.ru/api/v1/tts" 
        self.select_url = "https://freetts.ru/api/v1/select" 
        self.audio_base_url = "https://freetts.ru" 
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout
        self.voices = {}
        self.load_voices()
        # Set default voice to first available
        self.default_voice = next(iter(self.voices.keys())) if self.voices else "ru-RU001"

    def load_voices(self):
        """Load voice data and format it appropriately"""
        try:
            response = self.session.get(self.select_url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                voices_data = data["data"]["voice"]

                if isinstance(voices_data, list):
                    for voice_info in voices_data:
                        if isinstance(voice_info, dict):
                            voice_id = voice_info.get("code")
                            voice_name = voice_info.get("name", voice_id)
                            if voice_id and voice_id.startswith(self.lang):
                                self.voices[voice_id] = voice_name
                                # Add to supported voices list
                                if voice_id not in self.SUPPORTED_VOICES:
                                    self.SUPPORTED_VOICES.append(voice_id)
                print("Voices loaded successfully")
            else:
                print(f"HTTP Error: {response.status_code}")
        except Exception as e:
            print(f"Error loading voices: {e}")

    def get_available_voices(self):
        """Return all available voices in string format"""
        if not self.voices:
            return "No voices available"
        voices_list = [f"{voice_id}: {name}" for voice_id, name in self.voices.items()]
        return "\n".join(voices_list)
    
    def validate_voice(self, voice: str) -> str:
        """
        Validate and return the voice ID.
        
        Args:
            voice (str): Voice ID to validate
            
        Returns:
            str: Validated voice ID
            
        Raises:
            ValueError: If voice is not supported
        """
        if voice not in self.voices:
            raise ValueError(f"Voice '{voice}' not supported. Available voices: {', '.join(self.voices.keys())}")
        return voice

    def tts(
        self, 
        text: str, 
        model: str = "gpt-4o-mini-tts",
        voice: str = None, 
        response_format: str = "mp3",
        instructions: str = None, 
        verbose: bool = True
    ) -> str:
        """
        Convert text to speech using FreeTTS API with OpenAI-compatible parameters.

        Args:
            text (str): The text to convert to speech
            model (str): The TTS model to use (gpt-4o-mini-tts, tts-1, tts-1-hd)
            voice (str): Voice ID to use for TTS (default: first available)
            response_format (str): Audio format (mp3, wav, aac)
            instructions (str): Voice instructions (not used by FreeTTS but kept for compatibility)
            verbose (bool): Whether to print debug information

        Returns:
            str: Path to the generated audio file

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If there is an error generating or saving the audio
        """
        # Validate input parameters
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        if len(text) > 10000:
            raise ValueError("Input text exceeds maximum allowed length of 10,000 characters")
            
        # Validate model and format using base class methods
        model = self.validate_model(model)
        response_format = self.validate_format(response_format)
        
        # Use default voice if not provided
        if voice is None:
            voice = self.default_voice
            
        # Validate voice
        voice = self.validate_voice(voice)
        
        try:
            if not self.voices:
                raise RuntimeError(f"No voices available for language '{self.lang}'")

            available_voices = self.get_available_voices()
            if not available_voices:
                if verbose:
                    print(f"No available voices for language '{self.lang}'")
                return ""

            payload = {
                "text": text,
                "voiceid": voice,
                "model": model,
                "format": response_format
            }

            response = self.session.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                mp3_path = data.get("data", {}).get("src", "")

                if not mp3_path:
                    raise RuntimeError("Audio file path not found in response")

                mp3_url = self.audio_base_url + mp3_path

                # Create filename with appropriate extension
                file_extension = f".{response_format}" if response_format != "pcm" else ".wav"
                filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + file_extension
                full_path = os.path.abspath(filename)

                with requests.get(mp3_url, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    with open(filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            f.write(chunk)

                if verbose:
                    print(f"[debug] Speech generated successfully")
                    print(f"[debug] Model: {model}")
                    print(f"[debug] Voice: {voice}")
                    print(f"[debug] Format: {response_format}")
                    print(f"[debug] Audio saved to: {filename}")
                    
                return full_path
            else:
                raise RuntimeError(f"API request failed with status code: {response.status_code}")

        except Exception as e:
            if verbose:
                print(f"[debug] Error generating speech: {e}")
            raise RuntimeError(f"Failed to generate speech: {str(e)}")

    def create_speech(
        self,
        input: str,
        model: str = "gpt-4o-mini-tts", 
        voice: str = None,
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
            instructions (str): Voice instructions (not used by FreeTTS)
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

    def stream_audio(
        self,
        input: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = None,
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
    # Initialize the FreeTTS client
    tts_client = FreeTTS(lang="ru-RU")
    
    # Print available voices
    print("Available voices:")
    print(tts_client.get_available_voices())
    
    # Convert text to speech
    try:
        audio_file = tts_client.create_speech(
            input="Привет, как дела?",
            model="gpt-4o-mini-tts",
            voice="ru-RU001",
            response_format="mp3",
            verbose=True
        )
        print(f"Audio saved to: {audio_file}")
    except Exception as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")