import json
import platform
import subprocess
from typing import Union, Optional, Dict, Any, Iterable, Generator, List, Callable, Literal, Tuple
import io
from collections import deque
import codecs

# Expanded encoding types
EncodingType = Literal['utf-8', 'utf-16', 'utf-32', 'ascii', 'latin1', 'cp1252', 'iso-8859-1', 
                        'iso-8859-2', 'windows-1250', 'windows-1251', 'windows-1252', 'gbk', 'big5',
                        'shift_jis', 'euc-jp', 'euc-kr']

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
        return None

    sanitized_chunk = chunk

    # Check if chunk starts with intro_value + skip_marker combination
    if intro_value and skip_markers:
        for marker in skip_markers:
            combined_marker = f"{intro_value}{marker}"
            if sanitized_chunk.startswith(combined_marker):
                return None

    if intro_value and sanitized_chunk.startswith(intro_value):
        sanitized_chunk = sanitized_chunk[len(intro_value):]

    if strip_chars is not None:
        sanitized_chunk = sanitized_chunk.strip(strip_chars)
    else:
        sanitized_chunk = sanitized_chunk.lstrip()

    # Check both standalone skip_markers and stripped version
    if not sanitized_chunk or any(
        marker in sanitized_chunk or sanitized_chunk == marker
        for marker in skip_markers
    ):
        return None

    if to_json:
        try:
            return json.loads(sanitized_chunk)
        except (json.JSONDecodeError, Exception) as e:
            return sanitized_chunk if yield_raw_on_error else None
    
    return sanitized_chunk

def _decode_byte_stream(
    byte_iterator: Iterable[bytes], 
    encoding: EncodingType = 'utf-8',
    errors: str = 'replace'
) -> Generator[str, None, None]:
    """
    Realtime byte stream decoder with flexible encoding support.
    
    Args:
        byte_iterator: Iterator yielding bytes
        encoding: Character encoding to use
        errors: How to handle encoding errors ('strict', 'ignore', 'replace')
    """
    # Initialize decoder with the specified encoding
    try:
        decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
    except LookupError:
        # Fallback to utf-8 if the encoding is not supported
        decoder = codecs.getincrementaldecoder('utf-8')(errors=errors)
    
    # Process byte stream in realtime
    for chunk_bytes in byte_iterator:
        if not chunk_bytes:
            continue

        try:
            # Decode chunk with specified encoding
            text = decoder.decode(chunk_bytes, final=False)
            if text:
                yield text
        except UnicodeDecodeError:
            yield f"[Encoding Error: Could not decode bytes with {encoding}]\n"
    
    # Final flush
    try:
        final_text = decoder.decode(b'', final=True)
        if final_text:
            yield final_text
    except UnicodeDecodeError:
        yield f"[Encoding Error: Could not decode final bytes with {encoding}]\n"
def sanitize_stream(
    data: Union[str, bytes, Iterable[str], Iterable[bytes]],
    intro_value: str = "data:",
    to_json: bool = True,
    skip_markers: Optional[List[str]] = None,
    strip_chars: Optional[str] = None,
    start_marker: Optional[str] = None,
    end_marker: Optional[str] = None,
    content_extractor: Optional[Callable[[Union[str, Dict[str, Any]]], Optional[Any]]] = None,
    yield_raw_on_error: bool = True,
    encoding: EncodingType = 'utf-8',
    encoding_errors: str = 'replace',
    chunk_size: Optional[int] = None,
) -> Generator[Any, None, None]:
    """
    Realtime stream processor that handles string/byte streams with minimal latency.
    
    Features:
    - Direct realtime processing of byte streams
    - Optimized string handling and JSON parsing
    - Robust error handling and validation
    - Flexible encoding support
    - Drop-in replacement for response.iter_content/iter_lines
    
    Args:
        data: Input data (string, string iterator, or bytes iterator)
        intro_value: Prefix to remove from each chunk
        to_json: Whether to parse chunks as JSON
        skip_markers: Markers to skip
        strip_chars: Characters to strip
        start_marker: Processing start marker
        end_marker: Processing end marker
        content_extractor: Function to extract content
        yield_raw_on_error: Yield raw content on JSON errors
        encoding: Character encoding for byte streams ('utf-8', 'latin1', etc.)
        encoding_errors: How to handle encoding errors ('strict', 'ignore', 'replace')
        chunk_size: Chunk size for byte streams (None for default)
    
    Yields:
        Processed chunks (string or dictionary)
    
    Example:
        >>> # Process response content
        >>> for chunk in sanitize_stream(response.iter_content()):
        ...     process_chunk(chunk)
        
        >>> # Process a stream with specific encoding
        >>> for text in sanitize_stream(byte_stream, encoding='latin1', to_json=False):
        ...     process_text(text)
    """
    effective_skip_markers = skip_markers or []
    processing_active = start_marker is None

    if isinstance(data, (str, bytes)):
        # Optimized single string/bytes processing
        processed_item = None
        if isinstance(data, bytes):
            data = data.decode(encoding, errors=encoding_errors)
        if to_json:
            try:
                processed_item = json.loads(data)
            except json.JSONDecodeError:
                pass

        if processed_item is None:
            if not processing_active and data == start_marker:
                processing_active = True
            elif processing_active and end_marker is not None and data == end_marker:
                processing_active = False

            if processing_active:
                processed_item = _process_chunk(
                    data, intro_value, to_json, effective_skip_markers, 
                    strip_chars, yield_raw_on_error
                )

        if processed_item is not None:
            if content_extractor:
                try:
                    final_content = content_extractor(processed_item)
                    if final_content is not None:
                        yield final_content
                except Exception:
                    pass
            else:
                yield processed_item

    elif hasattr(data, '__iter__') or hasattr(data, 'iter_content'):
        # Efficient stream processing
        try:
            if hasattr(data, 'iter_content'):
                data = data.iter_content(chunk_size=chunk_size)
            first_item = next(iter(data))
        except StopIteration:
            return

        from itertools import chain
        stream = chain([first_item], data)

        # Choose processing path based on type
        if isinstance(first_item, bytes):
            line_iterator = _decode_byte_stream(
                stream, 
                encoding=encoding,
                errors=encoding_errors
            )
        elif isinstance(first_item, str):
            line_iterator = stream
        else:
            raise TypeError(f"Stream must yield strings or bytes, not {type(first_item).__name__}")

        # Process stream efficiently
        for line in line_iterator:
            if not processing_active and start_marker is not None and line == start_marker:
                processing_active = True
                continue
            if processing_active and end_marker is not None and line == end_marker:
                processing_active = False
                continue
            
            if processing_active:
                processed = _process_chunk(
                    line, intro_value, to_json, effective_skip_markers,
                    strip_chars, yield_raw_on_error
                )
                
                if processed is not None:
                    if content_extractor:
                        try:
                            final_content = content_extractor(processed)
                            if final_content is not None:
                                yield final_content
                        except Exception:
                            pass
                    else:
                        yield processed
    else:
        raise TypeError(f"Input must be a string or an iterable, not {type(data).__name__}")

from .conversation import Conversation
from .optimizers import Optimizers
from .Extra.autocoder import AutoCoder
from .prompt_manager import AwesomePrompts