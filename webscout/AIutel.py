import json
from typing import Union, Optional, Dict, Any, Iterable, Generator, List, Callable, Literal
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

    # Fast path for empty chunks
    if not chunk:
        return None

    # Use slicing for prefix removal (faster than startswith+slicing)
    sanitized_chunk = chunk
    if intro_value and len(chunk) >= len(intro_value) and chunk[:len(intro_value)] == intro_value:
        sanitized_chunk = chunk[len(intro_value):]

    # Optimize string stripping operations
    if strip_chars is not None:
        sanitized_chunk = sanitized_chunk.strip(strip_chars)
    else:
        # lstrip() is faster than strip() when we only need leading whitespace removed
        sanitized_chunk = sanitized_chunk.lstrip()

    # Skip empty chunks and markers
    if not sanitized_chunk or any(marker == sanitized_chunk for marker in skip_markers):
        return None

    # JSON parsing with optimized error handling
    if to_json:
        try:
            # Only strip before JSON parsing if needed
            if sanitized_chunk[0] not in '{[' or sanitized_chunk[-1] not in '}]':
                sanitized_chunk = sanitized_chunk.strip()
            return json.loads(sanitized_chunk)
        except (json.JSONDecodeError, Exception):
            return sanitized_chunk if yield_raw_on_error else None
    
    return sanitized_chunk

def _decode_byte_stream(
    byte_iterator: Iterable[bytes], 
    encoding: EncodingType = 'utf-8',
    errors: str = 'replace',
    buffer_size: int = 8192
) -> Generator[str, None, None]:
    """
    Realtime byte stream decoder with flexible encoding support.
    
    Args:
        byte_iterator: Iterator yielding bytes
        encoding: Character encoding to use
        errors: How to handle encoding errors ('strict', 'ignore', 'replace')
        buffer_size: Size of internal buffer for performance tuning
    """
    # Initialize decoder with the specified encoding
    try:
        decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
    except LookupError:
        # Fallback to utf-8 if the encoding is not supported
        decoder = codecs.getincrementaldecoder('utf-8')(errors=errors)
    
    # Process byte stream in realtime
    buffer = bytearray(buffer_size)
    buffer_view = memoryview(buffer)
    
    for chunk_bytes in byte_iterator:
        if not chunk_bytes:
            continue

        try:
            # Use buffer for processing if chunk size is appropriate
            if len(chunk_bytes) <= buffer_size:
                buffer[:len(chunk_bytes)] = chunk_bytes
                text = decoder.decode(buffer_view[:len(chunk_bytes)], final=False)
            else:
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
    data: Union[str, Iterable[str], Iterable[bytes]],
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
    buffer_size: int = 8192,
) -> Generator[Any, None, None]:
    """
    Optimized realtime stream processor that handles string/byte streams with minimal latency.
    
    Features:
    - Direct realtime processing of byte streams
    - Optimized string handling and JSON parsing
    - Robust error handling and validation
    - Flexible encoding support with memory-efficient buffering
    - High performance for large streams
    
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
        encoding: Character encoding for byte streams
        encoding_errors: How to handle encoding errors
        buffer_size: Size of internal processing buffer
    
    Yields:
        Processed chunks (string or dictionary)
    """
    effective_skip_markers = skip_markers or []
    processing_active = start_marker is None

    # Fast path for single string processing
    if isinstance(data, str):
        processed_item = None
        if processing_active:
            # Optimize JSON parsing for large strings
            if to_json:
                try:
                    # Use faster JSON parser for large strings
                    data = data.strip()
                    if data:
                        processed_item = json.loads(data)
                except json.JSONDecodeError:
                    processed_item = data if yield_raw_on_error else None
            else:
                processed_item = _process_chunk(
                    data, intro_value, False, effective_skip_markers, 
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
        return

    # Stream processing path
    if not hasattr(data, '__iter__'):
        raise TypeError(f"Input must be a string or an iterable, not {type(data).__name__}")
    
    try:
        iterator = iter(data)
        first_item = next(iterator, None)
        if first_item is None:
            return
            
        # Efficient streaming with itertools
        from itertools import chain
        stream = chain([first_item], iterator)

        # Determine if we're dealing with bytes or strings
        if isinstance(first_item, bytes):
            line_iterator = _decode_byte_stream(
                stream, 
                encoding=encoding,
                errors=encoding_errors,
                buffer_size=buffer_size
            )
        elif isinstance(first_item, str):
            line_iterator = stream
        else:
            raise TypeError(f"Stream must yield strings or bytes, not {type(first_item).__name__}")

        # Process stream with minimal allocations
        for line in line_iterator:
            if not line:
                continue
                
            # Handle markers efficiently
            if not processing_active and start_marker is not None:
                if line.strip() == start_marker:
                    processing_active = True
                continue
                
            if processing_active and end_marker is not None and line.strip() == end_marker:
                processing_active = False
                continue
            
            if processing_active:
                # Process chunk with optimized function
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
                            # Continue on extraction errors
                            pass
                    else:
                        yield processed
                        
    except Exception as e:
        # Log error but don't crash on stream processing exceptions
        import sys
        print(f"Stream processing error: {str(e)}", file=sys.stderr)


from .conversation import Conversation
from .optimizers import Optimizers
from .Extra.autocoder import AutoCoder
from .prompt_manager import AwesomePrompts