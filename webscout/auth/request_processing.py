"""
Request processing utilities for the Webscout API.
"""

import json
import time
import uuid
from typing import List, Dict, Any
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR
from fastapi.responses import StreamingResponse

from webscout.Provider.OPENAI.utils import ChatCompletion, Choice, ChatCompletionMessage, CompletionUsage
from webscout.Litlogger import Logger, LogLevel, LogFormat, ConsoleHandler
import sys

from .request_models import Message, ChatCompletionRequest
from .exceptions import APIError, clean_text

# Setup logger
logger = Logger(
    name="webscout.api",
    level=LogLevel.INFO,
    handlers=[ConsoleHandler(stream=sys.stdout)],
    fmt=LogFormat.DEFAULT
)


def process_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """Process and validate chat messages."""
    processed_messages = []

    for i, msg_in in enumerate(messages):
        try:
            message_dict_out = {"role": msg_in.role}

            if msg_in.content is None:
                message_dict_out["content"] = None
            elif isinstance(msg_in.content, str):
                message_dict_out["content"] = msg_in.content
            else:  # List[MessageContentParts]
                message_dict_out["content"] = [
                    part.model_dump(exclude_none=True) for part in msg_in.content
                ]

            if msg_in.name:
                message_dict_out["name"] = msg_in.name

            processed_messages.append(message_dict_out)

        except Exception as e:
            raise APIError(
                f"Invalid message at index {i}: {str(e)}",
                HTTP_422_UNPROCESSABLE_ENTITY,
                "invalid_request_error",
                param=f"messages[{i}]"
            )

    return processed_messages


def prepare_provider_params(chat_request: ChatCompletionRequest, model_name: str,
                          processed_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare parameters for the provider."""
    params = {
        "model": model_name,
        "messages": processed_messages,
        "stream": chat_request.stream,
    }

    # Add optional parameters if present
    optional_params = [
        "temperature", "max_tokens", "top_p", "presence_penalty",
        "frequency_penalty", "stop", "user"
    ]

    for param in optional_params:
        value = getattr(chat_request, param, None)
        if value is not None:
            params[param] = value

    return params


async def handle_streaming_response(provider: Any, params: Dict[str, Any], request_id: str) -> StreamingResponse:
    """Handle streaming chat completion response."""
    async def streaming():
        try:
            logger.debug(f"Starting streaming response for request {request_id}")
            completion_stream = provider.chat.completions.create(**params)

            # Check if it's iterable (generator, iterator, or other iterable types)
            if hasattr(completion_stream, '__iter__') and not isinstance(completion_stream, (str, bytes, dict)):
                try:
                    for chunk in completion_stream:
                        # Standardize chunk format before sending
                        if hasattr(chunk, 'model_dump'):  # Pydantic v2
                            chunk_data = chunk.model_dump(exclude_none=True)
                        elif hasattr(chunk, 'dict'):  # Pydantic v1
                            chunk_data = chunk.dict(exclude_none=True)
                        elif isinstance(chunk, dict):
                            chunk_data = chunk
                        else:  # Fallback for unknown chunk types
                            chunk_data = chunk
                        
                        # Clean text content in the chunk to remove control characters
                        if isinstance(chunk_data, dict) and 'choices' in chunk_data:
                            for choice in chunk_data.get('choices', []):
                                if isinstance(choice, dict):
                                    # Handle delta for streaming
                                    if 'delta' in choice and isinstance(choice['delta'], dict) and 'content' in choice['delta']:
                                        choice['delta']['content'] = clean_text(choice['delta']['content'])
                                    # Handle message for non-streaming
                                    elif 'message' in choice and isinstance(choice['message'], dict) and 'content' in choice['message']:
                                        choice['message']['content'] = clean_text(choice['message']['content'])
                        
                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                except TypeError as te:
                    logger.error(f"Error iterating over completion_stream: {te}")
                    # Fall back to treating as non-generator response
                    if hasattr(completion_stream, 'model_dump'):
                        response_data = completion_stream.model_dump(exclude_none=True)
                    elif hasattr(completion_stream, 'dict'):
                        response_data = completion_stream.dict(exclude_none=True)
                    else:
                        response_data = completion_stream
                    
                    # Clean text content in the response
                    if isinstance(response_data, dict) and 'choices' in response_data:
                        for choice in response_data.get('choices', []):
                            if isinstance(choice, dict):
                                if 'delta' in choice and isinstance(choice['delta'], dict) and 'content' in choice['delta']:
                                    choice['delta']['content'] = clean_text(choice['delta']['content'])
                                elif 'message' in choice and isinstance(choice['message'], dict) and 'content' in choice['message']:
                                    choice['message']['content'] = clean_text(choice['message']['content'])
                    
                    yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
            else:  # Non-generator response
                if hasattr(completion_stream, 'model_dump'):
                    response_data = completion_stream.model_dump(exclude_none=True)
                elif hasattr(completion_stream, 'dict'):
                    response_data = completion_stream.dict(exclude_none=True)
                else:
                    response_data = completion_stream
                
                # Clean text content in the response
                if isinstance(response_data, dict) and 'choices' in response_data:
                    for choice in response_data.get('choices', []):
                        if isinstance(choice, dict):
                            if 'delta' in choice and isinstance(choice['delta'], dict) and 'content' in choice['delta']:
                                choice['delta']['content'] = clean_text(choice['delta']['content'])
                            elif 'message' in choice and isinstance(choice['message'], dict) and 'content' in choice['message']:
                                choice['message']['content'] = clean_text(choice['message']['content'])
                
                yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming response for request {request_id}: {e}")
            error_message = clean_text(str(e))
            error_data = {
                "error": {
                    "message": error_message,
                    "type": "server_error",
                    "code": "streaming_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    return StreamingResponse(streaming(), media_type="text/event-stream")


async def handle_non_streaming_response(provider: Any, params: Dict[str, Any],
                                      request_id: str, start_time: float) -> Dict[str, Any]:
    """Handle non-streaming chat completion response."""
    try:
        logger.debug(f"Starting non-streaming response for request {request_id}")
        completion = provider.chat.completions.create(**params)

        if completion is None:
            # Return a valid OpenAI-compatible error response
            return ChatCompletion(
                id=request_id,
                created=int(time.time()),
                model=params.get("model", "unknown"),
                choices=[Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="No response generated."),
                    finish_reason="error"
                )],
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            ).model_dump(exclude_none=True)

        # Standardize response format
        if hasattr(completion, "model_dump"):  # Pydantic v2
            response_data = completion.model_dump(exclude_none=True)
        elif hasattr(completion, "dict"):  # Pydantic v1
            response_data = completion.dict(exclude_none=True)
        elif isinstance(completion, dict):
            response_data = completion
        else:
            raise APIError(
                "Invalid response format from provider",
                HTTP_500_INTERNAL_SERVER_ERROR,
                "provider_error"
            )
        
        # Clean text content in the response to remove control characters
        if isinstance(response_data, dict) and 'choices' in response_data:
            for choice in response_data.get('choices', []):
                if isinstance(choice, dict) and 'message' in choice:
                    if isinstance(choice['message'], dict) and 'content' in choice['message']:
                        choice['message']['content'] = clean_text(choice['message']['content'])

        elapsed = time.time() - start_time
        logger.info(f"Completed non-streaming request {request_id} in {elapsed:.2f}s")

        return response_data

    except Exception as e:
        logger.error(f"Error in non-streaming response for request {request_id}: {e}")
        error_message = clean_text(str(e))
        raise APIError(
            f"Provider error: {error_message}",
            HTTP_500_INTERNAL_SERVER_ERROR,
            "provider_error"
        )
