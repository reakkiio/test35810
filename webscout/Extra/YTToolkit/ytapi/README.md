<div align="center">
  <a href="https://github.com/OEvortex/Webscout">
    <img src="https://img.shields.io/badge/YTToolkit-YouTube%20Data%20Extraction-red?style=for-the-badge&logo=youtube&logoColor=white" alt="YTToolkit Logo">
  </a>
  <br/>
  <h1>YTToolkit</h1>
  <p><strong>Powerful YouTube Data Extraction Module for Webscout</strong></p>
  <p>
    Extract comprehensive YouTube data without API keys - channel information, video metadata, playlists, and more.
  </p>

  <!-- Badges -->
  <p>
    <a href="https://pypi.org/project/webscout/"><img src="https://img.shields.io/pypi/v/webscout.svg?style=flat-square&logo=pypi&label=PyPI" alt="PyPI Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/No%20API%20Key-Required-success?style=flat-square" alt="No API Key Required"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python" alt="Python Version"></a>
  </p>
</div>

## 🚀 Features

* **Channel Metadata Extraction**
  * Retrieve comprehensive channel information
  * Extract subscriber count, views, description
  * Fetch channel avatars and banners

* **Video Information**
  * Detailed video metadata retrieval
  * Stream and upload history tracking
  * Upcoming video detection
  * Thumbnail and embed code generation

* **Advanced Search Capabilities**
  * Trending videos across categories
  * Flexible search and filtering

* **No Official API Required**
  * Web scraping-based extraction
  * No API key needed

## 📦 Installation

Install as part of the Webscout package:

```bash
pip install webscout
```

## 💡 Quick Examples

### Channel Information

```python
from webscout import Channel

# Create a channel instance
channel = Channel('@PewDiePie')

# Access channel metadata
print(channel.name)          # Channel name
print(channel.subscribers)   # Subscriber count
print(channel.description)   # Channel description

# Get recent uploads
recent_videos = channel.uploads(10)  # Get 10 most recent video IDs

# Check live status
if channel.live:
    print(f"Currently streaming: {channel.streaming_now}")
```

### Video Extraction

```python
from webscout import Video

# Get video metadata
video = Video('https://www.youtube.com/watch?v=9bZkp7q19f0')
print(video.metadata)

# Get video thumbnails
print(video.thumbnail_urls)

# Get video embed code
print(video.embed_html)
```

### Trending Videos

```python
from webscout import Extras

# Get trending videos
trending = Extras.trending_videos()
music_videos = Extras.music_videos()
gaming_videos = Extras.gaming_videos()
```

### Search across YouTube

```python
from webscout import Search
print(Search.videos("LET ME IN WWE SONG"))
print(Search.channels("OEvortex"))
print(Search.playlists("OEvortex"))
```

## 📋 Detailed Usage

### Channel Class

The `Channel` class provides comprehensive access to YouTube channel data:

```python
from webscout import Channel

# Initialize with channel handle, ID, or URL
channel = Channel('@PewDiePie')

# Basic information
print(f"Name: {channel.name}")
print(f"ID: {channel.id}")
print(f"Subscribers: {channel.subscribers}")
print(f"Total Views: {channel.views}")
print(f"Country: {channel.country}")

# Media URLs
print(f"Avatar: {channel.avatar}")
print(f"Banner: {channel.banner}")
print(f"URL: {channel.url}")

# Content
print(f"Description: {channel.description}")
print(f"Social Links: {channel.socials}")

# Live status
if channel.live:
    print(f"Currently streaming: {channel.streaming_now}")

# Get videos
recent_uploads = channel.uploads(20)  # Get 20 most recent videos
```

### Video Class

The `Video` class extracts detailed information about YouTube videos:

```python
from webscout import Video

# Initialize with video ID or URL
video = Video('https://www.youtube.com/watch?v=9bZkp7q19f0')

# Get comprehensive metadata
metadata = video.metadata
print(f"Title: {metadata['title']}")
print(f"Views: {metadata['views']}")
print(f"Duration: {metadata['duration']} seconds")
print(f"Upload Date: {metadata['upload_date']}")
print(f"Author ID: {metadata['author_id']}")
print(f"Tags: {metadata['tags']}")

# Get thumbnails in different resolutions
thumbnails = video.thumbnail_urls
print(f"Default thumbnail: {thumbnails['default']}")
print(f"High quality thumbnail: {thumbnails['high']}")
print(f"Maximum resolution thumbnail: {thumbnails['maxres']}")

# Get embed code and URL
print(f"Embed HTML: {video.embed_html}")
print(f"Embed URL: {video.embed_url}")
```

### Search Class

The `Search` class allows you to search for videos, channels, and playlists:

```python
from webscout import Search

# Search for videos
video_results = Search.videos("Python tutorial", limit=5)
for video_id in video_results:
    print(f"Video ID: {video_id}")

# Search for channels
channel_results = Search.channels("coding", limit=3)
for channel_id in channel_results:
    print(f"Channel ID: {channel_id}")

# Search for playlists
playlist_results = Search.playlists("music mix", limit=3)
for playlist_id in playlist_results:
    print(f"Playlist ID: {playlist_id}")
```

### Extras Class

The `Extras` class provides access to trending videos and category-specific content:

```python
from webscout import Extras

# Get trending videos
trending = Extras.trending_videos(limit=10)
for video_id in trending:
    print(f"Trending Video ID: {video_id}")

# Get music videos
music = Extras.music_videos(limit=5)
for video_id in music:
    print(f"Music Video ID: {video_id}")

# Get gaming videos
gaming = Extras.gaming_videos(limit=5)
for video_id in gaming:
    print(f"Gaming Video ID: {video_id}")
```

## 🛠 Modules

- `channel.py`: Channel metadata and interaction
- `video.py`: Video information extraction with enhanced features
- `extras.py`: Trending and category-based video retrieval
- `query.py`: Advanced search capabilities
- `playlist.py`: Playlist metadata extraction

## ⚠️ Disclaimer

This module is designed for educational and research purposes. Please use responsibly and in accordance with YouTube's terms of service.

<div align="center">
  <a href="https://t.me/PyscoutAI"><img alt="Telegram Group" src="https://img.shields.io/badge/Telegram%20Group-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
  <a href="https://t.me/ANONYMOUS_56788"><img alt="Developer Telegram" src="https://img.shields.io/badge/Developer%20Contact-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
  <a href="https://youtube.com/@OEvortex"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white"></a>
  <a href="https://buymeacoffee.com/oevortex"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black"></a>
</div>
