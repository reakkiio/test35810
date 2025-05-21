import codecs
import json
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
    Union,
)

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
    error_handler: Optional[Callable[[Exception, str], Optional[Any]]] = None,
) -> Union[str, Dict[str, Any], None]:
    """
    Sanitizes and potentially parses a single chunk of text.

    This function performs several operations on the input chunk:
    - Removes a specified prefix (`intro_value`).
    - Strips leading/trailing characters (`strip_chars`).
    - Skips chunks matching specific markers (`skip_markers`).
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


    Args:
        chunk: Chunk of text to process.
        intro_value: Prefix to remove from the chunk.
        to_json: Parse the chunk as JSON if True.
        skip_markers: List of markers to skip.
        strip_chars: Characters to strip from the chunk.
        yield_raw_on_error: Whether to return the raw chunk on parse errors.
        error_handler: Optional callback ``Callable[[Exception, str], Optional[Any]]``
            invoked when JSON parsing fails. The callback should return a value to
            yield instead of the raw chunk, or ``None`` to ignore.
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

    # JSON parsing with optimized error handling
    if to_json:
        try:
            # Only strip before JSON parsing if needed
            if sanitized_chunk[0] not in '{[' or sanitized_chunk[-1] not in '}]':
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
) -> Generator[Any, None, None]:
    """
    Processes a stream of data (strings or bytes) in real-time, applying various transformations and filtering.

    This function is designed to handle streaming data, allowing for operations such as
    prefix removal, JSON parsing, skipping lines based on markers, and extracting specific content.
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
        line_delimiter: Delimiter used to split incoming text into lines. ``None``
            uses ``str.splitlines()``.
        error_handler: Callback invoked with ``(Exception, str)`` when JSON
            parsing fails. If the callback returns a value, it is yielded instead of the raw line.

    Yields:
        Any: Processed data, which can be a string, a dictionary (if `to_json` is True), or the result of `content_extractor`.

    Raises:
        TypeError: If the input `data` is not a string or an iterable.
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
                except Exception as e:
                    if error_handler:
                        try:
                            handled = error_handler(e, data)
                            if handled is not None:
                                processed_item = handled

                        except Exception:
                            pass
                    if processed_item is None:
                        processed_item = data if yield_raw_on_error else None
            else:
                processed_item = _process_chunk(
                    data, intro_value, False, effective_skip_markers,
                    strip_chars, yield_raw_on_error, error_handler
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
                        for subline in (chunk.split(line_delimiter) if line_delimiter is not None else chunk.splitlines()):
                            result = _process_chunk(
                                subline,
                                intro_value,
                                to_json,
                                effective_skip_markers,
                                strip_chars,
                                yield_raw_on_error,
                                error_handler,
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
) -> AsyncGenerator[Any, None]:
    """
    Asynchronously processes a stream of data (strings or bytes), applying transformations and filtering.

    This function is the asynchronous counterpart to `_sanitize_stream_sync`. It handles
    streaming data, allowing for operations such as prefix removal, JSON parsing,
    skipping lines based on markers, and extracting specific content. It also supports
    custom error handling for JSON parsing failures.

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
    """
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
        ):
            yield item
        return

    effective_skip_markers = skip_markers or []
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


def sanitize_stream(
    data: Union[
        str,
        Iterable[str],
        Iterable[bytes],
        AsyncIterable[str],
        AsyncIterable[bytes],
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
) -> Union[Generator[Any, None, None], AsyncGenerator[Any, None]]:
    """
    Processes streaming data (strings or bytes) in either synchronous or asynchronous mode.

    This function acts as a unified interface for handling both synchronous and
    asynchronous data streams. It automatically detects the type of input data and
    dispatches it to the appropriate processing function (`_sanitize_stream_sync` or
    `_sanitize_stream_async`).

    Args:
        data (Union[str, Iterable[str], Iterable[bytes], AsyncIterable[str], AsyncIterable[bytes]]):
            The data to be processed. Can be a string, a synchronous iterable of strings or bytes,
            or an asynchronous iterable of strings or bytes.
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

    Returns:
        Union[Generator[Any, None, None], AsyncGenerator[Any, None]]:
            A generator or an asynchronous generator yielding the processed data.
    """

    if hasattr(data, "__aiter__"):
        return _sanitize_stream_async(
            data, intro_value, to_json, skip_markers, strip_chars,
            start_marker, end_marker, content_extractor, yield_raw_on_error,
            encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
        )
    return _sanitize_stream_sync(
        data, intro_value, to_json, skip_markers, strip_chars,
        start_marker, end_marker, content_extractor, yield_raw_on_error,
        encoding, encoding_errors, buffer_size, line_delimiter, error_handler,
    )


from .conversation import Conversation  # noqa: E402,F401
from .Extra.autocoder import AutoCoder  # noqa: E402,F401
from .optimizers import Optimizers  # noqa: E402,F401
from .prompt_manager import AwesomePrompts  # noqa: E402,F401
