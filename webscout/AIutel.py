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
    Robust realtime stream processor that handles string/byte streams with correct marker extraction/skipping.
    Now handles split markers, partial chunks, and skips lines containing (not just equal to) skip markers.
    """
    effective_skip_markers = skip_markers or []
    processing_active = start_marker is None
    buffer = ""
    found_start = False if start_marker else True

    # Fast path for single string processing
    if isinstance(data, str):
        processed_item = None
        if processing_active:
            if to_json:
                try:
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

        for line in line_iterator:
            if not line:
                continue
            buffer += line
            while True:
                # Look for start marker if needed
                if not found_start and start_marker:
                    idx = buffer.find(start_marker)
                    if idx != -1:
                        found_start = True
                        buffer = buffer[idx + len(start_marker):]
                    else:
                        # Not found, keep buffering
                        buffer = buffer[-max(len(start_marker), 256):]  # avoid unbounded growth
                        break
                # Look for end marker if needed
                if found_start and end_marker:
                    idx = buffer.find(end_marker)
                    if idx != -1:
                        chunk = buffer[:idx]
                        buffer = buffer[idx + len(end_marker):]
                        processing_active = False
                    else:
                        chunk = buffer
                        buffer = ""
                        processing_active = True
                    # Process chunk if we are in active region
                    if chunk and processing_active:
                        # Split into lines for skip marker logic
                        for subline in chunk.splitlines():
                            # Remove intro_value prefix if present
                            if intro_value and subline.startswith(intro_value):
                                subline = subline[len(intro_value):]
                            # Strip chars if needed
                            if strip_chars is not None:
                                subline = subline.strip(strip_chars)
                            else:
                                subline = subline.lstrip()
                            # Skip if matches any skip marker (using 'in')
                            if any(marker in subline for marker in effective_skip_markers):
                                continue
                            # Skip empty
                            if not subline:
                                continue
                            # JSON parse if needed
                            if to_json:
                                try:
                                    if subline and (subline[0] in '{[' and subline[-1] in '}]'):
                                        parsed = json.loads(subline)
                                        result = parsed
                                    else:
                                        result = subline
                                except Exception:
                                    result = subline if yield_raw_on_error else None
                            else:
                                result = subline
                            if result is not None:
                                if content_extractor:
                                    try:
                                        final_content = content_extractor(result)
                                        if final_content is not None:
                                            yield final_content
                                    except Exception:
                                        pass
                                else:
                                    yield result
                    if not processing_active:
                        found_start = False
                    if idx == -1:
                        break
                elif found_start:
                    # No end marker, process all buffered content
                    chunk = buffer
                    buffer = ""
                    if chunk:
                        for subline in chunk.splitlines():
                            if intro_value and subline.startswith(intro_value):
                                subline = subline[len(intro_value):]
                            if strip_chars is not None:
                                subline = subline.strip(strip_chars)
                            else:
                                subline = subline.lstrip()
                            if any(marker in subline for marker in effective_skip_markers):
                                continue
                            if not subline:
                                continue
                            if to_json:
                                try:
                                    if subline and (subline[0] in '{[' and subline[-1] in '}]'):
                                        parsed = json.loads(subline)
                                        result = parsed
                                    else:
                                        result = subline
                                except Exception:
                                    result = subline if yield_raw_on_error else None
                            else:
                                result = subline
                            if result is not None:
                                if content_extractor:
                                    try:
                                        final_content = content_extractor(result)
                                        if final_content is not None:
                                            yield final_content
                                    except Exception:
                                        pass
                                else:
                                    yield result
                    break
                else:
                    break
    except Exception as e:
        import sys
        print(f"Stream processing error: {str(e)}", file=sys.stderr)


from .conversation import Conversation
from .optimizers import Optimizers
from .Extra.autocoder import AutoCoder
from .prompt_manager import AwesomePrompts