import os
import requests
from datetime import datetime
from datetime import datetime
from webscout.Provider.TTS import BaseTTSProvider
from webscout.litagent import LitAgent


class FreeTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the FreeTTS API.
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

    def __init__(self, lang="ru-RU", timeout: int = 30, proxies: dict = None):
        """Initializes the FreeTTS TTS client."""
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

    def load_voices(self):
        """Загружает данные о голосах и приводит их к нужному виду"""
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
                else:
                    print("Error")
                print("Done")
            else:
                print(f"Error HTTP: {response.status_code}")
        except Exception as e:
            print(f"Error downloading voice: {e}")

    def get_available_voices(self):
        """Возвращает все доступные голоса в формате строки"""
        if not self.voices:
            return "Error"
        voices_list = [f"{voice_id}: {name}" for voice_id, name in self.voices.items()]
        return "\n".join(voices_list)

    def tts(self, text: str, voiceid: str = None) -> str:
        """
        Converts text to speech using the FreeTTS API and saves it to a file.
        Args:
            text (str): The text to convert to speech
            voiceid (str): Voice ID to use for TTS (default: first available)
        Returns:
            str: Path to the generated audio file (MP3)
        Raises:
            AssertionError: If no voices are available
            requests.RequestException: If there's an error communicating with the API
            RuntimeError: If there's an error processing the audio
        """
        try:
            if not self.voices:
                raise RuntimeError(f"No voices available for language '{self.lang}'")

            available_voices = self.get_available_voices()
            if not available_voices:
                print(f"There are no available voices for the language '{self.lang}'")
                return ""

            if voiceid is None:
                voiceid = next(iter(available_voices.keys()))

            payload = {
                "text": text,
                "voiceid": voiceid
            }

            response = requests.post(self.url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                mp3_path = data.get("data", {}).get("src", "")

                if not mp3_path:
                    print("The path to the audio file in the response was not found.")
                    return ""

                mp3_url = self.audio_base_url + mp3_path

                mp3_filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".mp3"
                full_path = os.path.abspath(mp3_filename)

                with requests.get(mp3_url, stream=True) as r:
                    r.raise_for_status()
                    with open(mp3_filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            f.write(chunk)

                print(f"File '{mp3_filename}'saved successfully!")
                return full_path

        except Exception as e:
            print(e)

if __name__ == "__main__":
    tts = FreeTTS(lang="ru")
    available_voices = tts.get_available_voices()
    print("Available voices:", available_voices)

    text_to_speak = input("\nEnter text: ")
    voice_id = "ru-RU001" 
    print("[debug] Generating audio...")
    audio_file = tts.tts(text=text_to_speak, voiceid=voice_id)
    print(f"Audio saved to: {audio_file}")