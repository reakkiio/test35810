import codecs
import json
import re
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Pattern,
    Union,
)

# Expanded encoding types
EncodingType = Literal['utf-8', 'utf-16', 'utf-32', 'ascii', 'latin1', 'cp1252', 'iso-8859-1',
                        'iso-8859-2', 'windows-1250', 'windows-1251', 'windows-1252', 'gbk', 'big5',
                        'shift_jis', 'euc-jp', 'euc-kr']

def _compile_regexes(patterns: Optional[List[Union[str, Pattern[str]]]]) -> Optional[List[Pattern[str]]]:
    """
    Compile regex patterns from strings or return compiled patterns as-is.
    
    Args:
        patterns: List of regex patterns as strings or compiled Pattern objects.
        
    Returns:
        List of compiled Pattern objects, or None if input is None.
        
    Raises:
        ValueError: If any pattern is invalid.
    """
    if not patterns:
        return None
    
    compiled_patterns = []
    for i, pattern in enumerate(patterns):
        try:
            if isinstance(pattern, str):
                compiled_patterns.append(re.compile(pattern))
            elif isinstance(pattern, Pattern):
                compiled_patterns.append(pattern)
            else:
                raise ValueError(f"Pattern at index {i} must be a string or compiled regex pattern, got {type(pattern)}")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern at index {i}: '{pattern}' - {str(e)}")
    
    return compiled_patterns

def _process_chunk(
    chunk: str,
    intro_value: str,
    to_json: bool,
    skip_markers: List[str],
    strip_chars: Optional[str],
    yield_raw_on_error: bool,
    error_handler: Optional[Callable[[Exception, str], Optional[Any]]] = None,
    skip_regexes: Optional[List[Pattern[str]]] = None,
    extract_regexes: Optional[List[Pattern[str]]] = None,
) -> Union[str, Dict[str, Any], None]:
    """
    Sanitizes and potentially parses a single chunk of text.

    This function performs several operations on the input chunk:
    - Removes a specified prefix (`intro_value`).
    - Strips leading/trailing characters (`strip_chars`).
    - Skips chunks matching specific markers (`skip_markers`).
    - Skips chunks matching regex patterns (`skip_regexes`).
    - Extracts content using regex capturing groups (`extract_regexes`).
    - Optionally parses the chunk as JSON (`to_json`).
    - Handles JSON parsing errors with an optional callback (`error_handler`).

    Args:
        chunk (str): The chunk of text to process.
        intro_value (str): The prefix to remove from the chunk.
        to_json (bool): If True, attempts to parse the chunk as JSON.
        skip_markers (List[str]): A list of markers; chunks matching these are skipped.
        strip_chars (Optional[str]): Characters to strip from the beginning and end of the chunk.
        yield_raw_on_error (bool): If True, returns the raw chunk when JSON parsing fails; otherwise, returns None.
        error_handler (Optional[Callable[[Exception, str], Optional[Any]]]): An optional callback function that is called when JSON parsing fails.
            It receives the exception and the sanitized chunk as arguments.  It should return a value to yield instead of the raw chunk, or None to ignore.
        skip_regexes (Optional[List[Pattern[str]]]): A list of compiled regex patterns; chunks matching any of these are skipped.
        extract_regexes (Optional[List[Pattern[str]]]): A list of compiled regex patterns for extracting content using capturing groups.

    """
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

    # Apply regex-based extraction first (if provided)
    if extract_regexes:
        extracted_content = None
        for regex in extract_regexes:
            match = regex.search(sanitized_chunk)
            if match:
                # If there are capturing groups, return the first group or all groups as a tuple
                if match.groups():
                    if len(match.groups()) == 1:
                        extracted_content = match.group(1)
                    else:
                        # Multiple groups - return as tuple converted to string for JSON compatibility
                        extracted_content = str(match.groups())
                else:
                    # No capturing groups, return the full match
                    extracted_content = match.group(0)
                break  # Use first matching extraction regex
        
        # If extract_regexes are provided but no match found, skip this chunk entirely
        if extracted_content is None:
            return None
        
        sanitized_chunk = extracted_content

    # Apply regex-based skipping (after extraction)
    if skip_regexes:
        if any(regex.search(sanitized_chunk) for regex in skip_regexes):
            return None

    # JSON parsing with optimized error handling
    if to_json:
        try:
            # Only strip before JSON parsing if both boundaries are incorrect
            if len(sanitized_chunk) >= 2 and sanitized_chunk[0] not in '{[' and sanitized_chunk[-1] not in '}]':
                sanitized_chunk = sanitized_chunk.strip()
            return json.loads(sanitized_chunk)
        except (json.JSONDecodeError, Exception) as e:
            if error_handler:
                try:
                    handled = error_handler(e, sanitized_chunk)
                    if handled is not None:
                        return handled
                except Exception:
                    pass
            return sanitized_chunk if yield_raw_on_error else None

    return sanitized_chunk

def _decode_byte_stream(
    byte_iterator: Iterable[bytes],
    encoding: EncodingType = 'utf-8',
    errors: str = 'replace',
    buffer_size: int = 8192
) -> Generator[str, None, None]:
    """
    Decodes a byte stream in realtime with flexible encoding support.

    This function takes an iterator of bytes and decodes it into a stream of strings
    using the specified character encoding. It handles encoding errors gracefully
    and can be tuned for performance with the `buffer_size` parameter.

    Args:
        byte_iterator (Iterable[bytes]): An iterator that yields chunks of bytes.
        encoding (EncodingType): The character encoding to use for decoding.
            Defaults to 'utf-8'.  Supports a wide range of encodings, including:
            'utf-8', 'utf-16', 'utf-32', 'ascii', 'latin1', 'cp1252', 'iso-8859-1',
            'iso-8859-2', 'windows-1250', 'windows-1251', 'windows-1252', 'gbk', 'big5',
            'shift_jis', 'euc-jp', 'euc-kr'.
        errors (str): Specifies how encoding errors should be handled.
            Options are 'strict' (raises an error), 'ignore' (skips the error), and
            'replace' (replaces the erroneous byte with a replacement character).
            Defaults to 'replace'.
        buffer_size (int): The size of the internal buffer used for decoding.

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

async def _decode_byte_stream_async(
    byte_iterator: Iterable[bytes],
    encoding: EncodingType = 'utf-8',
    errors: str = 'replace',
    buffer_size: int = 8192
) -> AsyncGenerator[str, None]:
    """
    Asynchronously decodes a byte stream with flexible encoding support.

    This function is the asynchronous counterpart to `_decode_byte_stream`. It takes
    an asynchronous iterator of bytes and decodes it into a stream of strings using
    the specified character encoding. It handles encoding errors gracefully and can
    be tuned for performance with the `buffer_size` parameter.

    Args:
        byte_iterator (Iterable[bytes]): An asynchronous iterator that yields chunks of bytes.
        encoding (EncodingType): The character encoding to use for decoding.
            Defaults to 'utf-8'.  Supports a wide range of encodings, including:
            'utf-8', 'utf-16', 'utf-32', 'ascii', 'latin1', 'cp1252', 'iso-8859-1',
            'iso-8859-2', 'windows-1250', 'windows-1251', 'windows-1252', 'gbk', 'big5',
            'shift_jis', 'euc-jp', 'euc-kr'.
        errors (str): Specifies how encoding errors should be handled.
            Options are 'strict' (raises an error), 'ignore' (skips the error), and
            'replace' (replaces the erroneous byte with a replacement character).
            Defaults to 'replace'.
        buffer_size (int): The size of the internal buffer used for decoding.
    """
    try:
        decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
    except LookupError:
        decoder = codecs.getincrementaldecoder('utf-8')(errors=errors)

    buffer = bytearray(buffer_size)
    buffer_view = memoryview(buffer)

    async for chunk_bytes in byte_iterator:
        if not chunk_bytes:
            continue
        try:
            if len(chunk_bytes) <= buffer_size:
                buffer[:len(chunk_bytes)] = chunk_bytes
                text = decoder.decode(buffer_view[:len(chunk_bytes)], final=False)
            else:
                text = decoder.decode(chunk_bytes, final=False)
            if text:
                yield text
        except UnicodeDecodeError:
            yield f"[Encoding Error: Could not decode bytes with {encoding}]\n"

    try:
        final_text = decoder.decode(b'', final=True)
        if final_text:
            yield final_text
    except UnicodeDecodeError:
        yield f"[Encoding Error: Could not decode final bytes with {encoding}]\n"

def _sanitize_stream_sync(
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
    line_delimiter: Optional[str] = None,
    error_handler: Optional[Callable[[Exception, str], Optional[Any]]] = None,
    skip_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    extract_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    raw: bool = False,
) -> Generator[Any, None, None]:
    """
    Processes a stream of data (strings or bytes) in real-time, applying various transformations and filtering.

    This function is designed to handle streaming data, allowing for operations such as
    prefix removal, JSON parsing, skipping lines based on markers, regex-based filtering,
    and extracting specific content. It also supports custom error handling for JSON parsing failures.

    Args:
        data: String, iterable of strings, or iterable of bytes to process.
        intro_value: Prefix indicating the start of meaningful data.
        to_json: Parse the chunk as JSON if True.
        skip_markers: Lines containing any of these markers are skipped.
        strip_chars: Characters to strip from each line.
        start_marker: Begin processing only after this marker is found.
        end_marker: Stop processing once this marker is found.
        content_extractor: Optional callable to transform parsed content before yielding.
        yield_raw_on_error: Yield raw lines when JSON parsing fails.
        encoding: Byte stream encoding.
        encoding_errors: How to handle encoding errors.
        buffer_size: Buffer size for byte decoding.
        line_delimiter: Delimiter used to split incoming text into lines. ``None``
            uses ``str.splitlines()``.
        error_handler: Callback invoked with ``(Exception, str)`` when JSON
            parsing fails. If the callback returns a value, it is yielded instead of the raw line.
        skip_regexes: List of regex patterns (strings or compiled) for skipping lines that match.
        extract_regexes: List of regex patterns (strings or compiled) for extracting content using capturing groups.
        raw: If True, yields the raw response as returned by the API, chunk by chunk (no processing).

    Yields:
        Any: Processed data, which can be a string, a dictionary (if `to_json` is True), or the result of `content_extractor`.

    Raises:
        TypeError: If the input `data` is not a string or an iterable.
        ValueError: If any regex pattern is invalid.
    """
    # --- RAW MODE: yield each chunk exactly as returned by the API ---
    if raw:
        if isinstance(data, str):
            yield data
            return
        elif hasattr(data, '__iter__'):
            for chunk in data:
                if isinstance(chunk, (bytes, bytearray)):
                    yield chunk.decode(encoding, encoding_errors)
                elif chunk is not None:
                    yield chunk
            return
        else:
            if data is not None:
                yield data
            return
    # --- END RAW MODE ---
    
    effective_skip_markers = skip_markers or []
    # Compile regex patterns
    compiled_skip_regexes = _compile_regexes(skip_regexes)
    compiled_extract_regexes = _compile_regexes(extract_regexes)
    
    processing_active = start_marker is None
    buffer = ""
    found_start = False if start_marker else True
    line_iterator: Iterable[str]

    if isinstance(data, str):
        # If data is a string, decide whether to split it into lines
        # or treat it as an iterable containing a single chunk.
        temp_lines: List[str]
        if line_delimiter is None:  # Default: split by newlines if present
            if '\n' in data or '\r' in data:
                temp_lines = data.splitlines()
            else:
                temp_lines = [data]  # Treat as a single line/chunk
        elif line_delimiter in data:  # Custom delimiter found in string
            temp_lines = data.split(line_delimiter)
        else:  # Custom delimiter not found, or string is effectively a single segment
            temp_lines = [data]
        line_iterator = iter(temp_lines)
    elif hasattr(data, '__iter__'):  # data is an iterable (but not a string)
        _iter = iter(data)
        first_item = next(_iter, None)

        if first_item is None:  # Iterable was empty
            return

        from itertools import chain
        # Reconstruct the full iterable including the first_item
        stream_input_iterable = chain([first_item], _iter)

        if isinstance(first_item, bytes):
            # Ensure stream_input_iterable is typed as Iterable[bytes] for _decode_byte_stream
            line_iterator = _decode_byte_stream(
                stream_input_iterable, # type: ignore
                encoding=encoding,
                errors=encoding_errors,
                buffer_size=buffer_size
            )
        elif isinstance(first_item, str):
            # Ensure stream_input_iterable is typed as Iterable[str]
            line_iterator = stream_input_iterable # type: ignore
        else:
            raise TypeError(f"Iterable must yield strings or bytes, not {type(first_item).__name__}")
    else:  # Not a string and not an iterable
        raise TypeError(f"Input must be a string or an iterable, not {type(data).__name__}")

    try:
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
                        for subline in (chunk.split(line_delimiter) if line_delimiter is not None else chunk.splitlines()):
                            result = _process_chunk(
                                subline,
                                intro_value,
                                to_json,
                                effective_skip_markers,
                                strip_chars,
                                yield_raw_on_error,
                                error_handler,
                                compiled_skip_regexes,
                                compiled_extract_regexes,
                            )
                            if result is None:
                                continue
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
                        for subline in (chunk.split(line_delimiter) if line_delimiter is not None else chunk.splitlines()):
                            result = _process_chunk(
                                subline,
                                intro_value,
                                to_json,
                                effective_skip_markers,
                                strip_chars,
                                yield_raw_on_error,
                                error_handler,
                                compiled_skip_regexes,
                                compiled_extract_regexes,
                            )
                            if result is None:
                                continue
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


async def _sanitize_stream_async(
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
    line_delimiter: Optional[str] = None,
    error_handler: Optional[Callable[[Exception, str], Optional[Any]]] = None,
    skip_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    extract_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    raw: bool = False,
) -> AsyncGenerator[Any, None]:
    """
    Asynchronously processes a stream of data (strings or bytes), applying transformations and filtering.

    This function is the asynchronous counterpart to `_sanitize_stream_sync`. It handles
    streaming data, allowing for operations such as prefix removal, JSON parsing,
    skipping lines based on markers, regex-based filtering, and extracting specific content.
    It also supports custom error handling for JSON parsing failures.

    Args:
        data: String, iterable of strings, or iterable of bytes to process.
        intro_value: Prefix indicating the start of meaningful data.
        to_json: Parse JSON content if ``True``.
        skip_markers: Lines containing any of these markers are skipped.
        strip_chars: Characters to strip from each line.
        start_marker: Begin processing only after this marker is found.
        end_marker: Stop processing once this marker is found.
        content_extractor: Optional callable to transform parsed content before yielding.
        yield_raw_on_error: Yield raw lines when JSON parsing fails.
        encoding: Byte stream encoding.
        encoding_errors: How to handle encoding errors.
        buffer_size: Buffer size for byte decoding.
        line_delimiter: Delimiter used to split incoming text into lines. ``None`` uses ``str.splitlines()``.
        error_handler: Callback invoked with ``(Exception, str)`` when JSON parsing fails. If the callback returns a value, it is yielded in place of the raw line.
        skip_regexes: List of regex patterns (strings or compiled) for skipping lines that match.
        extract_regexes: List of regex patterns (strings or compiled) for extracting content using capturing groups.
        raw: If True, yields the raw response as returned by the API, chunk by chunk (no processing).
    """
    # --- RAW MODE: yield each chunk exactly as returned by the API ---
    if raw:
        if isinstance(data, str):
            yield data
            return
        elif hasattr(data, "__aiter__"):
            async for chunk in data:
                if isinstance(chunk, (bytes, bytearray)):
                    yield chunk.decode(encoding, encoding_errors)
                elif chunk is not None:
                    yield chunk
            return
        elif hasattr(data, "__iter__"):
            for chunk in data:
                if isinstance(chunk, (bytes, bytearray)):
                    yield chunk.decode(encoding, encoding_errors)
                elif chunk is not None:
                    yield chunk
            return
        else:
            if data is not None:
                yield data
            return
    # --- END RAW MODE ---
    
    if isinstance(data, str):
        for item in _sanitize_stream_sync(
            data,
            intro_value=intro_value,
            to_json=to_json,
            skip_markers=skip_markers,
            strip_chars=strip_chars,
            start_marker=start_marker,
            end_marker=end_marker,
            content_extractor=content_extractor,
            yield_raw_on_error=yield_raw_on_error,
            encoding=encoding,
            encoding_errors=encoding_errors,
            buffer_size=buffer_size,
            line_delimiter=line_delimiter,
            error_handler=error_handler,
            skip_regexes=skip_regexes,
            extract_regexes=extract_regexes,
            raw=raw,
        ):
            yield item
        return

    if not hasattr(data, "__aiter__"):
        # Fallback to synchronous processing if possible
        for item in _sanitize_stream_sync(
            data,
            intro_value=intro_value,
            to_json=to_json,
            skip_markers=skip_markers,
            strip_chars=strip_chars,
            start_marker=start_marker,
            end_marker=end_marker,
            content_extractor=content_extractor,
            yield_raw_on_error=yield_raw_on_error,
            encoding=encoding,
            encoding_errors=encoding_errors,
            buffer_size=buffer_size,
            line_delimiter=line_delimiter,
            error_handler=error_handler,
            skip_regexes=skip_regexes,
            extract_regexes=extract_regexes,
            raw=raw,
        ):
            yield item
        return

    effective_skip_markers = skip_markers or []
    # Compile regex patterns
    compiled_skip_regexes = _compile_regexes(skip_regexes)
    compiled_extract_regexes = _compile_regexes(extract_regexes)
    
    processing_active = start_marker is None
    buffer = ""
    found_start = False if start_marker else True

    iterator = data.__aiter__()
    first_item = None
    async for first_item in iterator:
        break
    if first_item is None:
        return
    async def _chain(first, it):
        yield first
        async for x in it:
            yield x

    stream = _chain(first_item, iterator)

    if isinstance(first_item, bytes):
        line_iterator = _decode_byte_stream_async(
            stream,
            encoding=encoding,
            errors=encoding_errors,
            buffer_size=buffer_size,
        )
    elif isinstance(first_item, str):
        line_iterator = stream
    else:
        raise TypeError(
            f"Stream must yield strings or bytes, not {type(first_item).__name__}"
        )

    try:
        async for line in line_iterator:
            if not line:
                continue
            buffer += line
            while True:
                if not found_start and start_marker:
                    idx = buffer.find(start_marker)
                if idx != -1:
                    found_start = True
                    buffer = buffer[idx + len(start_marker) :]
                else:
                    buffer = buffer[-max(len(start_marker), 256) :]
                    break
            if found_start and end_marker:
                idx = buffer.find(end_marker)
                if idx != -1:
                    chunk = buffer[:idx]
                    buffer = buffer[idx + len(end_marker) :]
                    processing_active = False
                else:
                    chunk = buffer
                    buffer = ""
                    processing_active = True
                if chunk and processing_active:
                    for subline in (
                        chunk.split(line_delimiter)
                        if line_delimiter is not None
                        else chunk.splitlines()
                    ):
                        result = _process_chunk(
                            subline,
                            intro_value,
                            to_json,
                            effective_skip_markers,
                            strip_chars,
                            yield_raw_on_error,
                            error_handler,
                            compiled_skip_regexes,
                            compiled_extract_regexes,
                        )
                        if result is None:
                            continue
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
                chunk = buffer
                buffer = ""
                if chunk:
                    for subline in (
                        chunk.split(line_delimiter)
                        if line_delimiter is not None
                        else chunk.splitlines()
                    ):
                        result = _process_chunk(
                            subline,
                            intro_value,
                            to_json,
                            effective_skip_markers,
                            strip_chars,
                            yield_raw_on_error,
                            error_handler,
                            compiled_skip_regexes,
                            compiled_extract_regexes,
                        )
                        if result is None:
                            continue
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
        print(f"Async stream processing error: {str(e)}", file=sys.stderr)


def sanitize_stream(
    data: Union[
        str,
        bytes,
        Iterable[str],
        Iterable[bytes],
        AsyncIterable[str],
        AsyncIterable[bytes],
        dict,
        list,
        int,
        float,
        bool,
        None,
    ],
    intro_value: str = "data:",
    to_json: bool = True,
    skip_markers: Optional[List[str]] = None,
    strip_chars: Optional[str] = None,
    start_marker: Optional[str] = None,
    end_marker: Optional[str] = None,
    content_extractor: Optional[Callable[[Union[str, Dict[str, Any]]], Optional[Any]]] = None,
    yield_raw_on_error: bool = True,
    encoding: EncodingType = "utf-8",
    encoding_errors: str = "replace",
    buffer_size: int = 8192,
    line_delimiter: Optional[str] = None,
    error_handler: Optional[Callable[[Exception, str], Optional[Any]]] = None,
    skip_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    extract_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    object_mode: Literal["as_is", "json", "str"] = "json",
    raw: bool = False,
) -> Union[Generator[Any, None, None], AsyncGenerator[Any, None]]:
    """
    Processes streaming data (strings or bytes) in either synchronous or asynchronous mode.
    Now supports non-iterable and miscellaneous input types (dict, list, int, float, bool, None).
    Includes regex-based content filtering and extraction capabilities.

    Args:
        data: The data to be processed. Can be a string, bytes, a synchronous iterable of strings or bytes,
            an asynchronous iterable of strings or bytes, or a single object (dict, list, int, float, bool, None).
        intro_value (str): Prefix indicating the start of meaningful data. Defaults to "data:".
        to_json (bool): Parse JSON content if ``True``. Defaults to True.
        skip_markers (Optional[List[str]]): Lines containing any of these markers are skipped. Defaults to None.
        strip_chars (Optional[str]): Characters to strip from each line. Defaults to None.
        start_marker (Optional[str]): Begin processing only after this marker is found. Defaults to None.
        end_marker (Optional[str]): Stop processing once this marker is found. Defaults to None.
        content_extractor (Optional[Callable[[Union[str, Dict[str, Any]]], Optional[Any]]]):
            Optional callable to transform parsed content before yielding. Defaults to None.
        yield_raw_on_error (bool): Yield raw lines when JSON parsing fails. Defaults to True.
        encoding (EncodingType): Byte stream encoding. Defaults to "utf-8".
        encoding_errors (str): How to handle encoding errors. Defaults to "replace".
        buffer_size (int): Buffer size for byte decoding. Defaults to 8192.
        line_delimiter (Optional[str]): Delimiter used to split incoming text into lines.
            ``None`` uses ``str.splitlines()``. Defaults to None.
        error_handler (Optional[Callable[[Exception, str], Optional[Any]]]):
            Callback invoked with ``(Exception, str)`` when JSON parsing fails.
            If the callback returns a value, it is yielded in place of the raw line. Defaults to None.
        skip_regexes (Optional[List[Union[str, Pattern[str]]]]): List of regex patterns (strings or compiled) 
            for skipping lines that match any pattern. Defaults to None.
        extract_regexes (Optional[List[Union[str, Pattern[str]]]]): List of regex patterns (strings or compiled) 
            for extracting content using capturing groups. If multiple groups are captured, they are returned as a tuple string. Defaults to None.
        object_mode (Literal["as_is", "json", "str"]): How to handle non-string, non-iterable objects.
            "json" (default) yields as JSON string, "str" yields as str(obj), "as_is" yields the object as-is.
        raw (bool): If True, yields the raw response as returned by the API, chunk by chunk (no splitting or joining).

    Returns:
        Union[Generator[Any, None, None], AsyncGenerator[Any, None]]:
            A generator or an asynchronous generator yielding the processed data, or raw data if raw=True.
            
    Raises:
        ValueError: If any regex pattern is invalid.
    """    # --- RAW MODE: yield each chunk exactly as returned by the API ---
    if raw:
        def _raw_passthrough_sync(source_iter):
            for chunk in source_iter:
                if isinstance(chunk, (bytes, bytearray)):
                    # Decode bytes preserving all whitespace and newlines
                    yield chunk.decode(encoding, encoding_errors)
                elif chunk is not None:
                    # Yield string chunks as-is, preserving all formatting
                    yield chunk
                # Skip None chunks entirely
        async def _raw_passthrough_async(source_aiter):
            async for chunk in source_aiter:
                if isinstance(chunk, (bytes, bytearray)):
                    # Decode bytes preserving all whitespace and newlines
                    yield chunk.decode(encoding, encoding_errors)
                elif chunk is not None:
                    # Yield string chunks as-is, preserving all formatting
                    yield chunk
                # Skip None chunks entirely
        # Sync iterable (but not str/bytes)
        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
            return _raw_passthrough_sync(data)
        # Async iterable
        if hasattr(data, "__aiter__"):
            return _raw_passthrough_async(data)
        # Single string or bytes
        if isinstance(data, (bytes, bytearray)):
            def _yield_single():
                yield data.decode(encoding, encoding_errors)
            return _yield_single()
        else:
            def _yield_single():
                if data is not None:
                    yield data
            return _yield_single()
    # --- END RAW MODE ---

    text_attr = getattr(data, "text", None)
    content_attr = getattr(data, "content", None)

    # Handle None
    if data is None:
        def _empty_gen():
            if False:
                yield None
        return _empty_gen()

    # Handle bytes directly
    if isinstance(data, bytes):
        try:
            payload = data.decode(encoding, encoding_errors)
        except Exception:
            payload = str(data)
        return _sanitize_stream_sync(
            payload, intro_value, to_json, skip_markers, strip_chars,
            start_marker, end_marker, content_extractor, yield_raw_on_error,
            encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
            skip_regexes, extract_regexes, raw,
        )

    # Handle string directly
    if isinstance(data, str):
        return _sanitize_stream_sync(
            data, intro_value, to_json, skip_markers, strip_chars,
            start_marker, end_marker, content_extractor, yield_raw_on_error,
            encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
            skip_regexes, extract_regexes, raw,
        )

    # Handle dict, list, int, float, bool (non-iterable, non-string/bytes)
    if isinstance(data, (dict, list, int, float, bool)):
        if object_mode == "as_is":
            def _as_is_gen():
                yield data
            return _as_is_gen()
        elif object_mode == "str":
            return _sanitize_stream_sync(
                str(data), intro_value, to_json, skip_markers, strip_chars,
                start_marker, end_marker, content_extractor, yield_raw_on_error,
                encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
                skip_regexes, extract_regexes, raw,
            )
        else:  # "json"
            try:
                json_str = json.dumps(data)
            except Exception:
                json_str = str(data)
            return _sanitize_stream_sync(
                json_str, intro_value, to_json, skip_markers, strip_chars,
                start_marker, end_marker, content_extractor, yield_raw_on_error,
                encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
                skip_regexes, extract_regexes, raw,
            )

    # Handle file-like objects (optional, treat as string if .read exists)
    if hasattr(data, "read") and callable(data.read):
        try:
            file_content = data.read()
            if isinstance(file_content, bytes):
                file_content = file_content.decode(encoding, encoding_errors)
            return _sanitize_stream_sync(
                file_content, intro_value, to_json, skip_markers, strip_chars,
                start_marker, end_marker, content_extractor, yield_raw_on_error,
                encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
                skip_regexes, extract_regexes, raw,
            )
        except Exception:
            pass  # fallback to next

    # Handle .text or .content attributes
    if isinstance(text_attr, str):
        payload = text_attr
        return _sanitize_stream_sync(
            payload, intro_value, to_json, skip_markers, strip_chars,
            start_marker, end_marker, content_extractor, yield_raw_on_error,
            encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
            skip_regexes, extract_regexes, raw,
        )
    elif isinstance(content_attr, bytes):
        try:
            payload = content_attr.decode(encoding, encoding_errors)
        except Exception:
            payload = str(content_attr)
        return _sanitize_stream_sync(
            payload, intro_value, to_json, skip_markers, strip_chars,
            start_marker, end_marker, content_extractor, yield_raw_on_error,
            encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
            skip_regexes, extract_regexes, raw,
        )

    # Handle async iterables
    if hasattr(data, "__aiter__"):
        return _sanitize_stream_async(
            data, intro_value, to_json, skip_markers, strip_chars,
            start_marker, end_marker, content_extractor, yield_raw_on_error,
            encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
            skip_regexes, extract_regexes, raw,
        )
    # Handle sync iterables (but not strings/bytes)
    if hasattr(data, "__iter__"):
        return _sanitize_stream_sync(
            data, intro_value, to_json, skip_markers, strip_chars,
            start_marker, end_marker, content_extractor, yield_raw_on_error,
            encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
            skip_regexes, extract_regexes, raw,
        )
    # Fallback: treat as string
    return _sanitize_stream_sync(
        str(data), intro_value, to_json, skip_markers, strip_chars,
        start_marker, end_marker, content_extractor, yield_raw_on_error,
        encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
        skip_regexes, extract_regexes, raw,
    )

# --- Decorator version of sanitize_stream ---
import functools
import asyncio
from typing import overload

def _sanitize_stream_decorator(
    _func=None,
    *,
    intro_value: str = "data:",
    to_json: bool = True,
    skip_markers: Optional[List[str]] = None,
    strip_chars: Optional[str] = None,
    start_marker: Optional[str] = None,
    end_marker: Optional[str] = None,
    content_extractor: Optional[Callable[[Union[str, Dict[str, Any]]], Optional[Any]]] = None,
    yield_raw_on_error: bool = True,
    encoding: EncodingType = "utf-8",
    encoding_errors: str = "replace",
    buffer_size: int = 8192,
    line_delimiter: Optional[str] = None,
    error_handler: Optional[Callable[[Exception, str], Optional[Any]]] = None,
    skip_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    extract_regexes: Optional[List[Union[str, Pattern[str]]]] = None,
    object_mode: Literal["as_is", "json", "str"] = "json",
    raw: bool = False,
):
    """
    Decorator for sanitize_stream. Can be used as @sanitize_stream or @sanitize_stream(...).
    All arguments are the same as sanitize_stream().
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                return sanitize_stream(
                    result,
                    intro_value=intro_value,
                    to_json=to_json,
                    skip_markers=skip_markers,
                    strip_chars=strip_chars,
                    start_marker=start_marker,
                    end_marker=end_marker,
                    content_extractor=content_extractor,
                    yield_raw_on_error=yield_raw_on_error,
                    encoding=encoding,
                    encoding_errors=encoding_errors,
                    buffer_size=buffer_size,
                    line_delimiter=line_delimiter,
                    error_handler=error_handler,
                    skip_regexes=skip_regexes,
                    extract_regexes=extract_regexes,
                    object_mode=object_mode,
                    raw=raw,
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return sanitize_stream(
                    result,
                    intro_value=intro_value,
                    to_json=to_json,
                    skip_markers=skip_markers,
                    strip_chars=strip_chars,
                    start_marker=start_marker,
                    end_marker=end_marker,
                    content_extractor=content_extractor,
                    yield_raw_on_error=yield_raw_on_error,
                    encoding=encoding,
                    encoding_errors=encoding_errors,
                    buffer_size=buffer_size,
                    line_delimiter=line_delimiter,
                    error_handler=error_handler,
                    skip_regexes=skip_regexes,
                    extract_regexes=extract_regexes,
                    object_mode=object_mode,
                    raw=raw,
                )
            return sync_wrapper
    if _func is None:
        return decorator
    else:
        return decorator(_func)

# Alias for decorator usage
LITSTREAM = sanitize_stream

# Decorator aliases
sanitize_stream_decorator = _sanitize_stream_decorator
lit_streamer = _sanitize_stream_decorator

# Allow @sanitize_stream and @lit_streamer as decorators
sanitize_stream.__decorator__ = _sanitize_stream_decorator
LITSTREAM.__decorator__ = _sanitize_stream_decorator
lit_streamer.__decorator__ = _sanitize_stream_decorator

def __getattr__(name):
    if name == 'sanitize_stream':
        return sanitize_stream
    if name == 'LITSTREAM':
        return LITSTREAM
    if name == 'sanitize_stream_decorator':
        return _sanitize_stream_decorator
    if name == 'lit_streamer':
        return _sanitize_stream_decorator
    raise AttributeError(f"module {__name__} has no attribute {name}")
