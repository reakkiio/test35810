# 🎤 Webscout Speech-to-Text (STT) Providers

A comprehensive collection of Speech-to-Text providers with OpenAI Whisper API compatibility. All providers return standardized responses that match the OpenAI Whisper API format, making it easy to switch between different STT services.

## 🌟 Features

- **OpenAI Whisper API Compatibility**: All providers return responses in OpenAI Whisper format
- **Multiple Providers**: Support for ElevenLabs, OpenAI Whisper, and more
- **Async Support**: Asynchronous versions of all providers
- **Multiple Output Formats**: JSON, text, SRT, VTT subtitle formats
- **Audio Format Support**: MP3, WAV, M4A, MP4, OGG, FLAC, WebM, AAC
- **Advanced Features**: Speaker diarization, timestamp granularities, language detection
- **Utility Functions**: Audio processing, format conversion, transcription merging

## 📦 Available Providers

### 1. ElevenLabsSTT
- Uses ElevenLabs API for high-quality transcription
- Supports speaker diarization
- Audio event tagging
- No API key required (unauthenticated mode)

### 2. OpenAIWhisperSTT
- Official OpenAI Whisper API support
- Multiple response formats
- Translation capabilities
- Timestamp granularities
- Requires OpenAI API key

## 🚀 Installation

```bash
pip install webscout
```

For audio processing features:
```bash
pip install webscout[audio]
# or
pip install pydub mutagen
```

## 💻 Basic Usage

### Quick Start

```python
from webscout.Provider.STT import ElevenLabsSTT

# Initialize the STT provider
stt = ElevenLabsSTT()

# Transcribe an audio file
result = stt.transcribe("audio.mp3")
print(result["text"])

# Access detailed information
print(f"Language: {result['language']}")
print(f"Duration: {result['duration']} seconds")

# Access segments with timestamps
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

### OpenAI Whisper Provider

```python
from webscout.Provider.STT import OpenAIWhisperSTT
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the provider
stt = OpenAIWhisperSTT()

# Transcribe with specific options
result = stt.transcribe(
    "audio.mp3",
    language="en",
    response_format="verbose_json",
    timestamp_granularities=["segment", "word"]
)

# Translate to English
translation = stt.translate("foreign_audio.mp3")
print(translation["text"])
```

### Async Usage

```python
import asyncio
from webscout.Provider.STT import AsyncElevenLabsSTT

async def transcribe_async():
    stt = AsyncElevenLabsSTT()
    result = await stt.transcribe("audio.mp3")
    return result["text"]

# Run async transcription
text = asyncio.run(transcribe_async())
print(text)
```

### Transcribe from URL

```python
from webscout.Provider.STT import ElevenLabsSTT

stt = ElevenLabsSTT()

# Transcribe audio from URL
result = stt.transcribe_from_url("https://example.com/audio.mp3")
print(result["text"])
```

## 📄 Output Formats

### Save in Different Formats

```python
from webscout.Provider.STT import ElevenLabsSTT

stt = ElevenLabsSTT()
result = stt.transcribe("audio.mp3")

# Save as JSON
stt.save_transcription(result, "transcript.json", format="json")

# Save as plain text
stt.save_transcription(result, "transcript.txt", format="txt")

# Save as SRT subtitles
stt.save_transcription(result, "transcript.srt", format="srt")

# Save as WebVTT subtitles
stt.save_transcription(result, "transcript.vtt", format="vtt")
```

### Response Format

All providers return responses in this standardized format:

```json
{
  "text": "Complete transcribed text",
  "task": "transcribe",
  "language": "en",
  "duration": 45.6,
  "segments": [
    {
      "id": 0,
      "seek": 0.0,
      "start": 0.0,
      "end": 5.2,
      "text": "First segment text",
      "tokens": [1234, 5678],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.0
    }
  ],
  "words": [
    {
      "word": "Hello",
      "start": 0.0,
      "end": 0.5,
      "probability": 0.99
    }
  ]
}
```

## 🔧 Advanced Features

### Audio Processing Utilities

```python
from webscout.Provider.STT import utils

# Validate audio format
is_valid = utils.validate_audio_format("audio.mp3")

# Get audio duration
duration = utils.get_audio_duration("audio.mp3")

# Convert audio format
wav_file = utils.convert_audio_format("audio.mp3", "wav")

# Split audio by silence
chunks = utils.split_audio_by_silence("long_audio.mp3")

# Transcribe chunks and merge results
results = []
for chunk in chunks:
    result = stt.transcribe(chunk)
    results.append(result)

merged_result = utils.merge_transcription_results(results)
```

### Speaker Diarization

```python
from webscout.Provider.STT import ElevenLabsSTT, utils

# Enable speaker diarization
stt = ElevenLabsSTT(diarize=True)
result = stt.transcribe("conversation.mp3")

# Extract speaker information
speakers = utils.extract_speakers_from_diarization(result)
for speaker_id, texts in speakers.items():
    print(f"Speaker {speaker_id}: {' '.join(texts)}")
```

### Formatted Output

```python
from webscout.Provider.STT import utils

# Format with timestamps
formatted = utils.format_transcript_with_timestamps(result)
print(formatted)

# Format with word-level timestamps
word_formatted = utils.format_transcript_with_timestamps(result, include_words=True)
print(word_formatted)

# Clean transcription text
clean_text = utils.clean_transcription_text(result["text"])
print(clean_text)
```

## ⚙️ Configuration

### ElevenLabs Provider Options

```python
from webscout.Provider.STT import ElevenLabsSTT

stt = ElevenLabsSTT(
    model_id="scribe_v1",           # Model to use
    allow_unauthenticated=True,     # Allow without API key
    tag_audio_events=True,          # Tag audio events
    diarize=True,                   # Enable speaker diarization
    timeout=60,                     # Request timeout
    proxies={"http": "proxy:8080"}  # Proxy configuration
)
```

### OpenAI Whisper Provider Options

```python
from webscout.Provider.STT import OpenAIWhisperSTT

stt = OpenAIWhisperSTT(
    api_key="your-api-key",         # OpenAI API key
    model="whisper-1",              # Model to use
    timeout=60,                     # Request timeout
    proxies={"http": "proxy:8080"}  # Proxy configuration
)
```

## 🛡️ Error Handling

```python
from webscout.Provider.STT import ElevenLabsSTT
from webscout import exceptions

stt = ElevenLabsSTT()

try:
    result = stt.transcribe("audio.mp3")
    print(result["text"])
except exceptions.FailedToGenerateResponseError as e:
    print(f"Transcription failed: {e}")
except exceptions.TimeoutE as e:
    print(f"Request timed out: {e}")
except exceptions.AuthenticationError as e:
    print(f"Authentication failed: {e}")
except FileNotFoundError as e:
    print(f"Audio file not found: {e}")
```

## 🌐 Proxy Support

```python
from webscout.Provider.STT import ElevenLabsSTT

# Use with proxy
stt = ElevenLabsSTT(proxies={
    'http': 'http://proxy.example.com:8080',
    'https': 'https://proxy.example.com:8080'
})

result = stt.transcribe("audio.mp3")
```

## 📝 Supported Audio Formats

- **MP3** - MPEG Audio Layer 3
- **WAV** - Waveform Audio File Format
- **M4A** - MPEG-4 Audio
- **MP4** - MPEG-4 Video (audio track)
- **OGG** - Ogg Vorbis
- **FLAC** - Free Lossless Audio Codec
- **WebM** - WebM Audio
- **AAC** - Advanced Audio Coding

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
