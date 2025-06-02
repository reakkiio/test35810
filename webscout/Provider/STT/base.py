"""
Base class for STT providers with common functionality.
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from webscout.AIbase import STTProvider


class BaseSTTProvider(STTProvider):
    """
    Base class for STT providers with common functionality.
    
    This class implements common methods and utilities that can be used
    by all STT providers, following OpenAI Whisper API response format.
    """
    
    def __init__(self, timeout: int = 60):
        """Initialize the base STT provider.
        
        Args:
            timeout (int): Request timeout in seconds. Defaults to 60.
        """
        self.timeout = timeout
        self.temp_dir = tempfile.mkdtemp(prefix="webscout_stt_")
    
    def _validate_audio_file(self, audio_path: Union[str, Path]) -> Path:
        """Validate that the audio file exists and is readable.
        
        Args:
            audio_path (Union[str, Path]): Path to the audio file
            
        Returns:
            Path: Validated Path object
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return audio_path
    
    def _format_openai_response(
        self,
        text: str,
        language: Optional[str] = None,
        duration: Optional[float] = None,
        segments: Optional[list] = None,
        words: Optional[list] = None
    ) -> Dict[str, Any]:
        """Format response to match OpenAI Whisper API format.
        
        Args:
            text (str): Transcribed text
            language (Optional[str]): Detected language code
            duration (Optional[float]): Audio duration in seconds
            segments (Optional[list]): List of text segments with timestamps
            words (Optional[list]): List of words with timestamps
            
        Returns:
            Dict[str, Any]: OpenAI Whisper-compatible response
        """
        response = {
            "text": text.strip(),
            "task": "transcribe",
            "language": language or "en",
            "duration": duration
        }
        
        if segments:
            response["segments"] = segments
            
        if words:
            response["words"] = words
            
        return response
    
    def _create_segment(
        self,
        id: int,
        seek: float,
        start: float,
        end: float,
        text: str,
        tokens: Optional[list] = None,
        temperature: float = 0.0,
        avg_logprob: float = -0.5,
        compression_ratio: float = 1.0,
        no_speech_prob: float = 0.0
    ) -> Dict[str, Any]:
        """Create a segment object in OpenAI Whisper format.
        
        Args:
            id (int): Segment ID
            seek (float): Seek position
            start (float): Start time in seconds
            end (float): End time in seconds
            text (str): Segment text
            tokens (Optional[list]): Token list
            temperature (float): Temperature used
            avg_logprob (float): Average log probability
            compression_ratio (float): Compression ratio
            no_speech_prob (float): No speech probability
            
        Returns:
            Dict[str, Any]: Segment object
        """
        return {
            "id": id,
            "seek": seek,
            "start": start,
            "end": end,
            "text": text,
            "tokens": tokens or [],
            "temperature": temperature,
            "avg_logprob": avg_logprob,
            "compression_ratio": compression_ratio,
            "no_speech_prob": no_speech_prob
        }
    
    def _create_word(
        self,
        word: str,
        start: float,
        end: float,
        probability: float = 1.0
    ) -> Dict[str, Any]:
        """Create a word object in OpenAI Whisper format.
        
        Args:
            word (str): The word
            start (float): Start time in seconds
            end (float): End time in seconds
            probability (float): Word probability
            
        Returns:
            Dict[str, Any]: Word object
        """
        return {
            "word": word,
            "start": start,
            "end": end,
            "probability": probability
        }
    
    def save_transcription(
        self,
        transcription: Dict[str, Any],
        destination: str = None,
        format: str = "json"
    ) -> str:
        """Save transcription to a file.
        
        Args:
            transcription (Dict[str, Any]): Transcription result
            destination (str, optional): Destination path
            format (str): Output format ('json', 'txt', 'srt', 'vtt')
            
        Returns:
            str: Path to the saved file
        """
        import json
        
        if destination is None:
            timestamp = int(time.time())
            destination = os.path.join(
                os.getcwd(),
                f"transcription_{timestamp}.{format}"
            )
        
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        if format == "json":
            with open(destination, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(destination, 'w', encoding='utf-8') as f:
                f.write(transcription.get("text", ""))
        elif format == "srt":
            self._save_as_srt(transcription, destination)
        elif format == "vtt":
            self._save_as_vtt(transcription, destination)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return destination
    
    def _save_as_srt(self, transcription: Dict[str, Any], destination: str):
        """Save transcription as SRT subtitle format."""
        segments = transcription.get("segments", [])
        with open(destination, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
    
    def _save_as_vtt(self, transcription: Dict[str, Any], destination: str):
        """Save transcription as WebVTT format."""
        segments = transcription.get("segments", [])
        with open(destination, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start_time = self._format_timestamp(segment["start"], vtt=True)
                end_time = self._format_timestamp(segment["end"], vtt=True)
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
    
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format timestamp for subtitle formats.
        
        Args:
            seconds (float): Time in seconds
            vtt (bool): Whether to format for WebVTT (uses . instead of ,)
            
        Returns:
            str: Formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        separator = "." if vtt else ","
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"
