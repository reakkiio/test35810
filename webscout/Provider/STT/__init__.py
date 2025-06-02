"""
Speech-to-Text (STT) providers for Webscout.

This module provides various STT providers with OpenAI Whisper API compatibility.
All providers return transcription results in a standardized format that matches
the OpenAI Whisper API response structure.

Available Providers:
- ElevenLabsSTT: Uses ElevenLabs API for speech-to-text
- OpenAIWhisperSTT: Uses OpenAI Whisper API or compatible endpoints
- AsyncElevenLabsSTT: Async version of ElevenLabs provider
- AsyncOpenAIWhisperSTT: Async version of OpenAI Whisper provider

Example Usage:
    from webscout.Provider.STT import ElevenLabsSTT
    
    # Initialize the STT provider
    stt = ElevenLabsSTT()
    
    # Transcribe an audio file
    result = stt.transcribe("audio.mp3")
    print(result["text"])
    
    # Save transcription in different formats
    stt.save_transcription(result, "transcript.json", format="json")
    stt.save_transcription(result, "transcript.srt", format="srt")
"""

from .base import *
from .elevenlabs import *
from . import utils
