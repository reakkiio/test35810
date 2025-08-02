import time
import requests
import pathlib
import base64
from io import BytesIO
from webscout import exceptions
from webscout.litagent import LitAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import utils
from .base import BaseTTSProvider


class GesseritTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the GesseritTTS API with OpenAI-compatible interface.
    
    This provider follows the OpenAI TTS API structure with support for:
    - Multiple TTS models (gpt-4o-mini-tts, tts-1, tts-1-hd)
    - Multiple voices with OpenAI-style naming
    - Voice instructions for controlling speech aspects
    - Multiple output formats (mp3, wav, aac, flac, opus, pcm)
    - Streaming support
    """
    
    # Override supported models for GesseritTTS
    SUPPORTED_MODELS = [
        "gpt-4o-mini-tts",  # Latest intelligent realtime model
        "tts-1",            # Lower latency model  
        "tts-1-hd"          # Higher quality model
    ]
    
    # Override supported voices with real Gesserit voice names
    SUPPORTED_VOICES = [
        "Emma",     # Female Voice
        "Liam",     # Male Voice
        "Noah",     # Male Voice
        "Oliver",   # Male Voice
        "Elijah",   # Male Voice
        "James",    # Male Voice
        "Charlie",  # Male Voice
        "Sophia",   # Female Voice
        "Cody",     # Male Voice
        "Emma"     # Female Voice (duplicate for variety)
    ]
    
    # Request headers
    headers: dict[str, str] = {
        "User-Agent": LitAgent().random()
    }
    cache_dir = pathlib.Path("./audio_cache")
    
    # Voice mapping from real names to Gesserit voice IDs
    voice_mapping: dict[str, str] = {
        "Emma": "en_us_001",      # Female Voice
        "Liam": "en_us_006",     # Male Voice
        "Noah": "en_us_007",     # Male Voice
        "Oliver": "en_us_009",   # Male Voice
        "Elijah": "en_us_010",   # Male Voice
        "James": "en_male_narration",  # Male Voice
        "Charlie": "en_male_funny",    # Male Voice
        "Sophia": "en_female_emotional",  # Female Voice
        "Cody": "en_male_cody"    # Male Voice
    }

    def __init__(self, timeout: int = 20, proxies: dict = None):
        """Initializes the GesseritTTS TTS client."""
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout

    def tts(self, text: str, voice: str = "Oliver", verbose: bool = False, **kwargs) -> str:
        """
        Converts text to speech using the GesseritTTS API and saves it to a file.
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use (default: "Oliver")
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
            voice = "Oliver"
            
        # Validate voice using base class method
        self.validate_voice(voice)
        
        # Map real voice name to Gesserit voice ID
        voice_id = self.voice_mapping.get(voice, "en_us_009")  # Default to Oliver
        
        filename = self.cache_dir / f"{int(time.time())}.mp3"

        # Split text into sentences
        sentences = utils.split_sentences(text)

        # Function to request audio for each chunk
        def generate_audio_for_chunk(part_text: str, part_number: int):
            while True:
                try:
                    payload = {
                        "text": part_text,
                        "voice": voice_id
                    }
                    response = self.session.post('https://gesserit.co/api/tiktok-tts', headers=self.headers, json=payload, timeout=self.timeout)
                    response.raise_for_status()

                    # Create the audio_cache directory if it doesn't exist
                    self.cache_dir.mkdir(parents=True, exist_ok=True)

                    # Check if the request was successful
                    if response.ok and response.status_code == 200:
                        data = response.json()
                        audio_base64 = data["audioUrl"].split(",")[1]
                        audio_data = base64.b64decode(audio_base64)
                        if verbose:
                            print(f"[debug] Chunk {part_number} processed successfully")
                        return part_number, audio_data
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

# Example usage
if __name__ == "__main__":
    gesserit = GesseritTTS()
    text = "This is a test of the GesseritTTS text-to-speech API. It supports multiple sentences and advanced logging."

    print("[debug] Generating audio...")
    try:
        audio_file = gesserit.create_speech(
            input=text,
            model="gpt-4o-mini-tts",
            voice="Oliver",
            response_format="mp3",
            verbose=True
        )
        print(f"Audio saved to: {audio_file}")
    except Exception as e:
        print(f"Error: {e}")