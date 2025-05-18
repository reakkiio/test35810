import time
import requests
import pathlib
import tempfile
from io import BytesIO
from webscout import exceptions
from webscout.litagent import LitAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
from webscout.Provider.TTS import utils
from webscout.Provider.TTS.base import BaseTTSProvider

class SthirTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the Sthir.org TTS API.
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": LitAgent().random(),
    }

    all_voices = {
        "aura-luna-en": "Sophie (American, Feminine)",
        "aura-stella-en": "Isabella (American, Feminine)",
        "aura-athena-en": "Emma (British, Feminine)",
        "aura-hera-en": "Victoria (American, Feminine)",
        "aura-asteria-en": "Maria (American, Feminine)",
        "aura-arcas-en": "Alex (American, Masculine)",
        "aura-zeus-en": "Thomas (American, Masculine)",
        "aura-perseus-en": "Michael (American, Masculine)",
        "aura-angus-en": "Connor (Irish, Masculine)",
        "aura-orpheus-en": "James (American, Masculine)",
        "aura-helios-en": "William (British, Masculine)",
        "aura-orion-en": "Daniel (American, Masculine)",
    }

    def __init__(self, timeout: int = 20, proxies: dict = None):
        """Initializes the SthirTTS client."""
        super().__init__()
        self.api_url = "https://sthir.org/com.api/tts-api.php"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout

    def tts(self, text: str, voice: str = "aura-luna-en") -> str:
        """
        Converts text to speech using the Sthir.org API and saves it to a file.

        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use for TTS (default: "aura-luna-en")

        Returns:
            str: Path to the generated audio file

        Raises:
            exceptions.FailedToGenerateResponseError: If there is an error generating or saving the audio.
        """
        assert (
            voice in self.all_voices
        ), f"Voice '{voice}' not one of [{', '.join(self.all_voices.keys())}]"

        filename = pathlib.Path(tempfile.mktemp(suffix=".mp3", dir=self.temp_dir))
        payload = {"text": text, "voice": voice}

        try:
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200 and len(response.content) > 0:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return filename.as_posix()
            else:
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        raise exceptions.FailedToGenerateResponseError(f"API error: {error_data['error']}")
                except Exception:
                    pass
                raise exceptions.FailedToGenerateResponseError(f"Sthir API error: {response.text}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Failed to perform the operation: {e}")

# Example usage
if __name__ == "__main__":
    sthir = SthirTTS()
    text = "This is a test of the Sthir.org text-to-speech API. It supports multiple voices."
    audio_file = sthir.tts(text, voice="aura-luna-en")
    print(f"Audio saved to: {audio_file}")
