import json
import platform
import subprocess
from typing import Union, Optional, Dict, Any, Iterable, Generator, List, Callable
import io # Import io for type checking
# Removed logging import and configuration

# Helper function to process a single chunk
def _process_chunk(
    chunk: str,
    intro_value: str,
    to_json: bool,
    skip_markers: List[str],
    strip_chars: Optional[str],
    yield_raw_on_error: bool,
) -> Union[str, Dict[str, Any], None]:
    """Internal helper to sanitize and potentially parse a single chunk."""
    if not isinstance(chunk, str):
        # Silently skip non-strings when processing an iterable
        return None

    sanitized_chunk = chunk

    # 1. Remove the prefix
    if intro_value and sanitized_chunk.startswith(intro_value):
        sanitized_chunk = sanitized_chunk[len(intro_value):]

    # 2. Strip characters/whitespace
    if strip_chars is not None:
        # Strip specified chars from both ends
        sanitized_chunk = sanitized_chunk.strip(strip_chars)
    else:
        # Default: strip only leading whitespace *after* prefix removal
        sanitized_chunk = sanitized_chunk.lstrip()

    # 3. Skip if empty or a marker (checked *after* stripping)
    if not sanitized_chunk or sanitized_chunk in skip_markers:
        return None

    # 4. Attempt JSON parsing if requested
    if to_json:
        try:
            return json.loads(sanitized_chunk)
        except json.JSONDecodeError:
            return sanitized_chunk if yield_raw_on_error else None
        except Exception: # Catch other potential JSON errors
             return sanitized_chunk if yield_raw_on_error else None
    else:
        # 5. Return sanitized string if no JSON parsing needed
        return sanitized_chunk

# Helper generator to decode bytes and split lines
def _decode_byte_stream(byte_iterator: Iterable[bytes]) -> Generator[str, None, None]:
    """Decodes bytes from an iterator, handles line splitting, and yields strings."""
    buffer = b""
    decoder = io.TextIOWrapper(io.BytesIO(buffer), encoding='utf-8', errors='ignore') # Use TextIOWrapper for robust decoding

    for chunk_bytes in byte_iterator:
        if not chunk_bytes:
            continue

        # Append new bytes to the buffer
        current_pos = decoder.tell()
        decoder.seek(0, io.SEEK_END)
        decoder.buffer.write(chunk_bytes) # type: ignore # Write bytes to underlying buffer
        decoder.seek(current_pos) # Reset position

        # Read lines
        line = decoder.readline()
        while line:
            if line.endswith('\n'):
                yield line.rstrip('\r\n') # Yield complete line without newline chars
            else:
                # Incomplete line, put it back by adjusting the read position
                decoder.seek(current_pos) # Go back to where we started reading this potential line
                break # Stop reading lines for now, wait for more bytes
            current_pos = decoder.tell() # Update position for next potential line read
            line = decoder.readline()

    # Yield any remaining data in the buffer after the loop finishes
    remaining = decoder.read()
    if remaining:
        yield remaining.rstrip('\r\n')

def sanitize_stream(
    data: Union[str, Iterable[str], Iterable[bytes]],
    intro_value: str = "data:",
    to_json: bool = True,
    skip_markers: Optional[List[str]] = None,
    strip_chars: Optional[str] = None,
    start_marker: Optional[str] = None,
    end_marker: Optional[str] = None,
    content_extractor: Optional[Callable[[Union[str, Dict[str, Any]]], Optional[Any]]] = None,
    yield_raw_on_error: bool = True,
) -> Generator[Any, None, None]:
    """
    Processes a single string chunk or an iterable stream of text chunks (like SSE).
    Removes prefixes, strips characters, skips markers, optionally parses as JSON,
    and yields the results.

    This function always returns a generator, even for single string input
    (yielding 0 or 1 item). It can automatically handle byte streams
    (like response.iter_content) and decode them. It can also attempt
    to parse a single input string as JSON if `to_json` is True.

    Args:
        data (Union[str, Iterable[str], Iterable[bytes]]):
            A single string chunk, an iterable yielding string chunks,
            or an iterable yielding bytes (like response.iter_content()).
        intro_value (str, optional): The prefix to remove from each chunk/line.
                                     Set to None or "" to disable prefix removal.
                                     Defaults to "data:".
        to_json (bool, optional): If True, attempt to parse the sanitized chunk as JSON.
                                  If False, yield the sanitized string. Defaults to True.
        skip_markers (Optional[List[str]], optional): A list of exact string values
                                                      (after prefix removal and stripping)
                                                      to skip yielding. E.g., ["[DONE]"].
                                                      Defaults to None.
        strip_chars (Optional[str], optional): Characters to strip from the beginning
                                               and end of the sanitized chunk *before*
                                               JSON parsing or marker checking. If None (default),
                                               only leading whitespace is stripped *after*
                                               prefix removal.
        start_marker (Optional[str], optional): If provided, processing and yielding will
                                                only begin *after* this exact marker string
                                                is encountered in the raw input. Defaults to None.
        end_marker (Optional[str], optional): If provided, processing and yielding will
                                              stop *before* this exact marker string is encountered.
        content_extractor (Optional[Callable]): A function that takes the processed chunk
                                                (string or dict) and returns the final item
                                                to yield. If None, the processed chunk is yielded.
        yield_raw_on_error (bool, optional): If True and to_json is True, yield the raw
                                             sanitized string chunk if JSON parsing fails.
                                             If False, skip chunks that fail parsing.
                                             Defaults to True.

    Yields:
        Generator[Any, None, None]: # Yields result of extractor or processed chunk
            Processed chunks (string or dictionary).
            Skips empty chunks, chunks matching skip_markers,
            or chunks failing JSON parse if yield_raw_on_error is False.

    Raises:
        TypeError: If the input `data` is neither a string nor a valid iterable.
    """
    effective_skip_markers = skip_markers or []
    processing_active = start_marker is None # Start processing immediately if no start_marker

    if isinstance(data, str):
        # --- Handle single string input (potentially non-streaming JSON or text) ---
        processed_item = None
        if to_json:
            try:
                # Try parsing the whole string as JSON first
                json_obj = json.loads(data)
                # If successful, treat this as the single item to process
                processed_item = json_obj
            except json.JSONDecodeError:
                # If not JSON, fall back to processing as a single text line
                pass # processed_item remains None

        if processed_item is None:
            # Process as a single text line (respecting start/end markers if relevant for single line)
            if not processing_active and data == start_marker:
                processing_active = True # Activate processing but don't yield marker
            elif processing_active and end_marker is not None and data == end_marker:
                processing_active = False # Deactivate processing

            if processing_active:
                 # Apply standard chunk processing (prefix, strip, skip markers)
                 processed_item = _process_chunk(
                     data, intro_value, to_json, effective_skip_markers, strip_chars, yield_raw_on_error
                 )

        # Apply content extractor if an item was processed
        if processed_item is not None:
            if content_extractor:
                try:
                    final_content = content_extractor(processed_item)
                    if final_content is not None: # Yield whatever the extractor returns if not None
                        yield final_content
                except Exception:
                    pass # Skip if extractor fails
            else: # Yield directly if no extractor
                 yield processed_item

    elif hasattr(data, '__iter__') or hasattr(data, '__aiter__'): # Check for iterables (sync/async)
        # --- Handle Streaming Input (Bytes or Strings) ---
        data_iter = iter(data) # Get iterator
        try:
            first_item = next(data_iter)
        except StopIteration:
            return # Empty iterable

        # Reconstruct the iterable including the first item
        from itertools import chain
        reconstructed_iter = chain([first_item], data_iter)

        # --- Choose processing path based on type ---
        if isinstance(first_item, bytes):
            # Process byte stream
            line_iterator = _decode_byte_stream(reconstructed_iter) # type: ignore
        elif isinstance(first_item, str):
            # Process string stream directly
            line_iterator = reconstructed_iter # type: ignore
        else:
             raise TypeError(f"Iterable must yield strings or bytes, not {type(first_item).__name__}")

        # --- Process the line iterator (now guaranteed to yield strings) ---
        for line in line_iterator:
            # Check start marker if not already active
            if not processing_active and start_marker is not None and line == start_marker:
                processing_active = True
                continue # Skip the marker itself
            # Check end marker if active
            if processing_active and end_marker is not None and line == end_marker:
                processing_active = False
                continue # Skip the marker itself
            # Process and yield if active
            if processing_active:
                processed = _process_chunk(
                    line, intro_value, to_json, effective_skip_markers, strip_chars, yield_raw_on_error
                )
                if processed is not None:
                    # Apply content extractor
                    if content_extractor:
                        try:
                            final_content = content_extractor(processed)
                            if final_content is not None: # Yield whatever the extractor returns if not None
                                yield final_content
                        except Exception:
                             pass # Skip if extractor fails
                    else: # Yield directly if no extractor
                         yield processed
    else:
        raise TypeError(f"Input must be a string or an iterable, not {type(data).__name__}")


def run_system_command(
    command: str,
    exit_on_error: bool = True,
    stdout_error: bool = True,
    help: str = None,
):
    """Run commands against system
    Args:
        command (str): shell command
        exit_on_error (bool, optional): Exit on error. Defaults to True.
        stdout_error (bool, optional): Print out the error. Defaults to True
        help (str, optional): Help info in case of exception. Defaults to None.
    Returns:
        tuple : (is_successful, object[Exception|Subprocess.run])
    """
    try:
        # Run the command and capture the output
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return (True, result)
    except subprocess.CalledProcessError as e:
        if exit_on_error:
            raise Exception(f"Command failed with exit code {e.returncode}") from e
        else:
            return (False, e)

class Updates:
    """Webscout latest release info"""

    url = "https://api.github.com/repos/OE-LUCIFER/Webscout/releases/latest"

    @property
    def latest_version(self):
        return self.latest(version=True)

    def executable(self, system: str = platform.system()) -> str:
        """Url pointing to executable for particular system

        Args:
            system (str, optional): system name. Defaults to platform.system().

        Returns:
            str: url
        """
        for entry in self.latest()["assets"]:
            if entry.get("target") == system:
                return entry.get("url")

    def latest(self, whole: bool = False, version: bool = False) -> dict:
        """Check Webscout latest version info

        Args:
            whole (bool, optional): Return whole json response. Defaults to False.
            version (bool, optional): return version only. Defaults to False.

        Returns:
            bool|dict: version str or whole dict info
        """
        import requests

        data = requests.get(self.url).json()
        if whole:
            return data

        elif version:
            return data.get("tag_name")

        else:
            sorted = dict(
                tag_name=data.get("tag_name"),
                tarball_url=data.get("tarball_url"),
                zipball_url=data.get("zipball_url"),
                html_url=data.get("html_url"),
                body=data.get("body"),
            )
            whole_assets = []
            for entry in data.get("assets"):
                url = entry.get("browser_download_url")
                assets = dict(url=url, size=entry.get("size"))
                if ".deb" in url:
                    assets["target"] = "Debian"
                elif ".exe" in url:
                    assets["target"] = "Windows"
                elif "macos" in url:
                    assets["target"] = "Mac"
                elif "linux" in url:
                    assets["target"] = "Linux"

                whole_assets.append(assets)
            sorted["assets"] = whole_assets

            return sorted

from .conversation import Conversation

from .optimizers import Optimizers

from .Extra.autocoder import AutoCoder

from .prompt_manager import AwesomePrompts
