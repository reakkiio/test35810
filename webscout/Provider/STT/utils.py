"""
Utility functions for STT providers.
"""
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile


def validate_audio_format(audio_path: Union[str, Path]) -> bool:
    """
    Validate if the audio file format is supported.
    
    Args:
        audio_path (Union[str, Path]): Path to the audio file
        
    Returns:
        bool: True if format is supported, False otherwise
    """
    supported_formats = {
        '.mp3', '.wav', '.m4a', '.mp4', '.ogg', '.flac', '.webm', '.aac'
    }
    
    audio_path = Path(audio_path)
    return audio_path.suffix.lower() in supported_formats


def get_audio_duration(audio_path: Union[str, Path]) -> Optional[float]:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_path (Union[str, Path]): Path to the audio file
        
    Returns:
        Optional[float]: Duration in seconds, None if unable to determine
    """
    try:
        # Try using mutagen for audio metadata
        from mutagen import File
        audio_file = File(str(audio_path))
        if audio_file is not None and hasattr(audio_file, 'info'):
            return float(audio_file.info.length)
    except ImportError:
        pass
    except Exception:
        pass
    
    try:
        # Try using pydub as fallback
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except ImportError:
        pass
    except Exception:
        pass
    
    return None


def convert_audio_format(
    input_path: Union[str, Path],
    output_format: str = "wav",
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Convert audio file to a different format.
    
    Args:
        input_path (Union[str, Path]): Path to the input audio file
        output_format (str): Target format (wav, mp3, etc.)
        output_path (Optional[Union[str, Path]]): Output path, auto-generated if None
        
    Returns:
        Path: Path to the converted audio file
        
    Raises:
        ImportError: If pydub is not installed
        Exception: If conversion fails
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required for audio format conversion. Install with: pip install pydub")
    
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix(f'.{output_format}')
    else:
        output_path = Path(output_path)
    
    # Load and convert audio
    audio = AudioSegment.from_file(str(input_path))
    audio.export(str(output_path), format=output_format)
    
    return output_path


def split_audio_by_silence(
    audio_path: Union[str, Path],
    min_silence_len: int = 1000,
    silence_thresh: int = -40,
    keep_silence: int = 500
) -> List[Path]:
    """
    Split audio file by silence into smaller chunks.
    
    Args:
        audio_path (Union[str, Path]): Path to the audio file
        min_silence_len (int): Minimum length of silence in milliseconds
        silence_thresh (int): Silence threshold in dBFS
        keep_silence (int): Amount of silence to keep at the beginning and end
        
    Returns:
        List[Path]: List of paths to audio chunks
        
    Raises:
        ImportError: If pydub is not installed
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
    except ImportError:
        raise ImportError("pydub is required for audio splitting. Install with: pip install pydub")
    
    audio_path = Path(audio_path)
    audio = AudioSegment.from_file(str(audio_path))
    
    # Split audio on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    # Save chunks to temporary files
    chunk_paths = []
    temp_dir = tempfile.mkdtemp(prefix="webscout_stt_chunks_")
    
    for i, chunk in enumerate(chunks):
        chunk_path = Path(temp_dir) / f"chunk_{i:03d}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunk_paths.append(chunk_path)
    
    return chunk_paths


def merge_transcription_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple transcription results into a single result.
    
    Args:
        results (List[Dict[str, Any]]): List of transcription results
        
    Returns:
        Dict[str, Any]: Merged transcription result
    """
    if not results:
        return {
            "text": "",
            "task": "transcribe",
            "language": "en",
            "duration": 0.0,
            "segments": [],
            "words": []
        }
    
    if len(results) == 1:
        return results[0]
    
    # Merge text
    merged_text = " ".join(result.get("text", "").strip() for result in results if result.get("text"))
    
    # Use language from first result
    language = results[0].get("language", "en")
    
    # Sum durations
    total_duration = sum(result.get("duration", 0.0) for result in results)
    
    # Merge segments with time offset
    merged_segments = []
    time_offset = 0.0
    segment_id = 0
    
    for result in results:
        segments = result.get("segments", [])
        for segment in segments:
            merged_segment = segment.copy()
            merged_segment["id"] = segment_id
            merged_segment["start"] += time_offset
            merged_segment["end"] += time_offset
            merged_segment["seek"] += time_offset
            merged_segments.append(merged_segment)
            segment_id += 1
        
        # Update time offset for next result
        if segments:
            time_offset = segments[-1]["end"]
        else:
            time_offset += result.get("duration", 0.0)
    
    # Merge words with time offset
    merged_words = []
    time_offset = 0.0
    
    for result in results:
        words = result.get("words", [])
        for word in words:
            merged_word = word.copy()
            merged_word["start"] += time_offset
            merged_word["end"] += time_offset
            merged_words.append(merged_word)
        
        # Update time offset for next result
        if words:
            time_offset = words[-1]["end"]
        else:
            time_offset += result.get("duration", 0.0)
    
    return {
        "text": merged_text,
        "task": "transcribe",
        "language": language,
        "duration": total_duration,
        "segments": merged_segments,
        "words": merged_words if merged_words else None
    }


def extract_speakers_from_diarization(transcription: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract speaker information from diarized transcription.
    
    Args:
        transcription (Dict[str, Any]): Transcription result with speaker info
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping speaker IDs to their text segments
    """
    speakers = {}
    segments = transcription.get("segments", [])
    
    for segment in segments:
        speaker_id = segment.get("speaker", "unknown")
        text = segment.get("text", "").strip()
        
        if speaker_id not in speakers:
            speakers[speaker_id] = []
        
        if text:
            speakers[speaker_id].append(text)
    
    return speakers


def format_transcript_with_timestamps(
    transcription: Dict[str, Any],
    include_words: bool = False
) -> str:
    """
    Format transcription with timestamps for human reading.
    
    Args:
        transcription (Dict[str, Any]): Transcription result
        include_words (bool): Whether to include word-level timestamps
        
    Returns:
        str: Formatted transcript with timestamps
    """
    def format_time(seconds: float) -> str:
        """Format seconds as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    formatted_lines = []
    
    if include_words and transcription.get("words"):
        # Word-level formatting
        words = transcription["words"]
        for word in words:
            start_time = format_time(word["start"])
            end_time = format_time(word["end"])
            text = word["word"]
            formatted_lines.append(f"[{start_time} - {end_time}] {text}")
    else:
        # Segment-level formatting
        segments = transcription.get("segments", [])
        for segment in segments:
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"].strip()
            if text:
                formatted_lines.append(f"[{start_time} - {end_time}] {text}")
    
    return "\n".join(formatted_lines)


def clean_transcription_text(text: str) -> str:
    """
    Clean and normalize transcription text.
    
    Args:
        text (str): Raw transcription text
        
    Returns:
        str: Cleaned text
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common transcription artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after sentence endings
    
    # Capitalize first letter of sentences
    sentences = re.split(r'([.!?]+)', text)
    for i in range(0, len(sentences), 2):
        if sentences[i].strip():
            sentences[i] = sentences[i].strip().capitalize()
    
    text = ''.join(sentences)
    
    return text.strip()
