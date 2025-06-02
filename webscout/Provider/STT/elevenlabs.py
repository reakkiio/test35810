"""
ElevenLabs Speech-to-Text provider with OpenAI Whisper API compatibility.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union
import requests
from rich import print as rprint

from webscout.Provider.STT.base import BaseSTTProvider
from webscout import exceptions
from webscout.litagent import LitAgent


class ElevenLabsSTT(BaseSTTProvider):
    """
    ElevenLabs Speech-to-Text provider with OpenAI Whisper API compatibility.
    
    This provider uses the ElevenLabs API for speech-to-text transcription
    and formats the response to match OpenAI's Whisper API format.
    """
    
    def __init__(
        self,
        api_url: str = "https://api.elevenlabs.io/v1/speech-to-text",
        model_id: str = "scribe_v1",
        allow_unauthenticated: bool = True,
        tag_audio_events: bool = True,
        diarize: bool = True,
        timeout: int = 60,
        proxies: Optional[Dict] = None
    ):
        """Initialize ElevenLabs STT provider.
        
        Args:
            api_url (str): ElevenLabs API endpoint
            model_id (str): Model to use for transcription
            allow_unauthenticated (bool): Allow unauthenticated requests
            tag_audio_events (bool): Tag audio events in transcription
            diarize (bool): Enable speaker diarization
            timeout (int): Request timeout in seconds
            proxies (Optional[Dict]): Proxy configuration
        """
        super().__init__(timeout=timeout)
        self.api_url = api_url
        self.model_id = model_id
        self.allow_unauthenticated = allow_unauthenticated
        self.tag_audio_events = tag_audio_events
        self.diarize = diarize
        self.proxies = proxies or {}
        
        # Initialize user agent and session
        self.session = requests.Session()
        if self.proxies:
            self.session.proxies.update(self.proxies)
        
        self.headers = self._generate_headers()
    
    def _generate_headers(self) -> Dict[str, str]:
        """Generate headers with a random user agent."""
        return {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': LitAgent().random()
        }
    
    def _refresh_user_agent(self) -> None:
        """Refresh the user agent string."""
        self.headers['User-Agent'] = LitAgent().random()
    
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Transcribe a local audio file using ElevenLabs API.
        
        Args:
            audio_path (Union[str, Path]): Path to the local audio file
            **kwargs: Additional parameters (language, response_format, etc.)
            
        Returns:
            Dict[str, Any]: Transcription result in OpenAI Whisper format
            
        Raises:
            exceptions.FailedToGenerateResponseError: If transcription fails
            exceptions.TimeoutE: If request times out
        """
        try:
            audio_path = self._validate_audio_file(audio_path)
            
            # Refresh user agent before request
            self._refresh_user_agent()
            
            # Construct the API URL with parameters
            api_url = self.api_url
            if self.allow_unauthenticated:
                api_url += "?allow_unauthenticated=1"
            
            rprint(f"[bold green]Transcribing audio using ElevenLabs API: {audio_path}[/]")
            
            # Read the file content and prepare form data
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': audio_file,
                    'model_id': (None, self.model_id),
                    'tag_audio_events': (None, 'true' if self.tag_audio_events else 'false'),
                    'diarize': (None, 'true' if self.diarize else 'false')
                }
                
                response = self.session.post(
                    api_url,
                    files=files,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                rprint(f"[bold blue]API Response Status: {response.status_code}[/]")
                
                if response.status_code != 200:
                    raise exceptions.FailedToGenerateResponseError(
                        f"ElevenLabs API returned error: {response.status_code} - {response.text}"
                    )
                
                result = response.json()
                
                # Convert ElevenLabs response to OpenAI Whisper format
                return self._convert_to_openai_format(result, audio_path)
                
        except requests.exceptions.Timeout:
            raise exceptions.TimeoutE("Transcription request timed out")
        except Exception as e:
            if isinstance(e, (exceptions.TimeoutE, exceptions.FailedToGenerateResponseError)):
                raise
            raise exceptions.FailedToGenerateResponseError(f"Transcription failed: {str(e)}")
    
    def transcribe_from_url(self, audio_url: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio from URL.
        
        Note: ElevenLabs API requires file upload, so this method downloads
        the audio first and then transcribes it.
        
        Args:
            audio_url (str): URL of the audio file
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Transcription result in OpenAI Whisper format
        """
        try:
            # Download the audio file to a temporary location
            temp_file = self._download_audio(audio_url)
            
            # Transcribe the downloaded file
            result = self.transcribe(temp_file, **kwargs)
            
            # Clean up temporary file
            temp_file.unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"URL transcription failed: {str(e)}")
    
    def _download_audio(self, audio_url: str) -> Path:
        """Download audio file from URL to temporary location.
        
        Args:
            audio_url (str): URL of the audio file
            
        Returns:
            Path: Path to the downloaded temporary file
        """
        import tempfile
        import shutil
        from urllib.parse import urlparse
        
        # Get file extension from URL
        parsed_url = urlparse(audio_url)
        file_extension = Path(parsed_url.path).suffix or '.mp3'
        
        # Create temporary file
        temp_file = Path(tempfile.mktemp(suffix=file_extension, dir=self.temp_dir))
        
        # Download the file
        response = self.session.get(audio_url, timeout=self.timeout, stream=True)
        response.raise_for_status()
        
        with open(temp_file, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        return temp_file
    
    def _convert_to_openai_format(self, elevenlabs_result: Dict, audio_path: Path) -> Dict[str, Any]:
        """Convert ElevenLabs response to OpenAI Whisper format.
        
        Args:
            elevenlabs_result (Dict): Raw ElevenLabs API response
            audio_path (Path): Path to the audio file
            
        Returns:
            Dict[str, Any]: OpenAI Whisper-compatible response
        """
        text = elevenlabs_result.get("text", "")
        language = elevenlabs_result.get("language", "en")
        duration = elevenlabs_result.get("audio_duration")
        
        # Create segments from text_segments if available
        segments = []
        if "text_segments" in elevenlabs_result and elevenlabs_result["text_segments"]:
            for i, segment_data in enumerate(elevenlabs_result["text_segments"]):
                segment = self._create_segment(
                    id=i,
                    seek=segment_data.get("start", 0.0),
                    start=segment_data.get("start", 0.0),
                    end=segment_data.get("end", 0.0),
                    text=segment_data.get("text", "")
                )
                segments.append(segment)
        else:
            # Create a single segment for the entire text
            segment = self._create_segment(
                id=0,
                seek=0.0,
                start=0.0,
                end=duration or 0.0,
                text=text
            )
            segments.append(segment)
        
        # Create words array if available
        words = []
        if "audio_events" in elevenlabs_result and elevenlabs_result["audio_events"]:
            for event in elevenlabs_result["audio_events"]:
                if event.get("type") == "word":
                    word = self._create_word(
                        word=event.get("text", ""),
                        start=event.get("start", 0.0),
                        end=event.get("end", 0.0),
                        probability=event.get("confidence", 1.0)
                    )
                    words.append(word)
        
        return self._format_openai_response(
            text=text,
            language=language,
            duration=duration,
            segments=segments,
            words=words if words else None
        )



if __name__ == "__main__":
    # Example usage
    stt = ElevenLabsSTT()
    result = stt.transcribe(r"C:\Users\koula\Downloads\audio_2025-05-12_22-30-47.ogg")
    print(result)
