import requests
import json
import time
import uuid
from typing import List, Dict, Optional, Union, Generator, Any

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

# Attempt to import LitAgent, fallback if not available
try:
    from webscout.litagent import LitAgent
except ImportError:
    # Define a dummy LitAgent if webscout is not installed or accessible
    class LitAgent:
        def generate_fingerprint(self, browser: str = "chrome") -> Dict[str, Any]:
            # Return minimal default headers if LitAgent is unavailable
            print("Warning: LitAgent not found. Using default minimal headers.")
            return {
                "accept": "*/*",
                "accept_language": "en-US,en;q=0.9",
                "platform": "Windows",
                "sec_ch_ua": '"Not/A)Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
                "browser_type": browser,
            }

# --- Glider Client ---

class Completions(BaseCompletions):
    def __init__(self, client: 'Glider'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 2049,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation.
        Mimics openai.chat.completions.create
        """
        # Prepare the payload for Glider API
        payload = {
            "messages": messages,
            "model": self._client.convert_model_name(model),
        }

        # Add optional parameters if provided
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        if temperature is not None:
            payload["temperature"] = temperature

        if top_p is not None:
            payload["top_p"] = top_p

        # Add any additional parameters
        payload.update(kwargs)

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload)
        else:
            return self._create_non_stream(request_id, created_time, model, payload)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=self._client.timeout
            )
            response.raise_for_status()

            # Track token usage across chunks
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()

                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str == "[DONE]":
                            # Format the final [DONE] marker in OpenAI format
                            # print("data: [DONE]")
                            break

                        try:
                            data = json.loads(json_str)
                            choice_data = data.get('choices', [{}])[0]
                            delta_data = choice_data.get('delta', {})
                            finish_reason = choice_data.get('finish_reason')

                            # Update token counts if available
                            usage_data = data.get('usage', {})
                            if usage_data:
                                prompt_tokens = usage_data.get('prompt_tokens', prompt_tokens)
                                completion_tokens = usage_data.get('completion_tokens', completion_tokens)
                                total_tokens = usage_data.get('total_tokens', total_tokens)

                            # Create the delta object
                            delta = ChoiceDelta(
                                content=delta_data.get('content'),
                                role=delta_data.get('role'),
                                tool_calls=delta_data.get('tool_calls')
                            )

                            # Create the choice object
                            choice = Choice(
                                index=choice_data.get('index', 0),
                                delta=delta,
                                finish_reason=finish_reason,
                                logprobs=choice_data.get('logprobs')
                            )

                            # Create the chunk object
                            chunk = ChatCompletionChunk(
                                id=request_id,
                                choices=[choice],
                                created=created_time,
                                model=model,
                                system_fingerprint=data.get('system_fingerprint')
                            )

                            # Convert to dict for proper formatting
                            chunk_dict = chunk.to_dict()

                            # Add usage information to match OpenAI format
                            # Even if we don't have real token counts, include estimated usage
                            # This matches the format in the examples
                            usage_dict = {
                                "prompt_tokens": prompt_tokens or 10,
                                "completion_tokens": completion_tokens or (len(delta_data.get('content', '')) if delta_data.get('content') else 0),
                                "total_tokens": total_tokens or (10 + (len(delta_data.get('content', '')) if delta_data.get('content') else 0)),
                                "estimated_cost": None
                            }

                            # Update completion_tokens and total_tokens as we receive more content
                            if delta_data.get('content'):
                                completion_tokens += 1
                                total_tokens = prompt_tokens + completion_tokens
                                usage_dict["completion_tokens"] = completion_tokens
                                usage_dict["total_tokens"] = total_tokens

                            chunk_dict["usage"] = usage_dict

                            # Format the response in OpenAI format exactly as requested
                            # We need to print the raw string and also yield the chunk object
                            # This ensures both the console output and the returned object are correct
                            # print(f"data: {json.dumps(chunk_dict)}")

                            # Return the chunk object for internal processing
                            yield chunk
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line: {json_str}")
                            continue
        except requests.exceptions.RequestException as e:
            print(f"Error during Glider stream request: {e}")
            raise IOError(f"Glider request failed: {e}") from e
        except Exception as e:
            print(f"Error processing Glider stream: {e}")
            raise

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                timeout=self._client.timeout
            )
            response.raise_for_status()
            data = response.json()

            choices_data = data.get('choices', [])
            usage_data = data.get('usage', {})

            choices = []
            for choice_d in choices_data:
                message_d = choice_d.get('message', {})
                message = ChatCompletionMessage(
                    role=message_d.get('role', 'assistant'),
                    content=message_d.get('content', '')
                )
                choice = Choice(
                    index=choice_d.get('index', 0),
                    message=message,
                    finish_reason=choice_d.get('finish_reason', 'stop')
                )
                choices.append(choice)

            usage = CompletionUsage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0)
            )

            completion = ChatCompletion(
                id=request_id,
                choices=choices,
                created=created_time,
                model=data.get('model', model),
                usage=usage,
            )
            return completion

        except requests.exceptions.RequestException as e:
            print(f"Error during Glider non-stream request: {e}")
            raise IOError(f"Glider request failed: {e}") from e
        except Exception as e:
            print(f"Error processing Glider response: {e}")
            raise

class Chat(BaseChat):
    def __init__(self, client: 'Glider'):
        self.completions = Completions(client)

class Glider(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for Glider.so API.

    Usage:
        client = Glider()
        response = client.chat.completions.create(
            model="chat-llama-3-1-70b",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    AVAILABLE_MODELS = [
        "chat-llama-3-1-8b",
        "chat-llama-3-2-3b",
        "chat-deepseek-r1-qwen-32b",
        "chat-qwen-2-5-7b",
        "chat-qwen-qwq-32b",
        "deepseek-ai/DeepSeek-R1",
    ]

    # No model mapping needed as we use the model names directly

    def __init__(self, timeout: Optional[int] = None, browser: str = "chrome"):
        """
        Initialize the Glider client.

        Args:
            timeout: Request timeout in seconds (None for no timeout)
            browser: Browser to emulate in user agent
        """
        self.timeout = timeout
        self.api_endpoint = "https://glider.so/api/chat"
        self.session = requests.Session()

        agent = LitAgent()
        fingerprint = agent.generate_fingerprint(browser)

        self.headers = {
            "Accept": fingerprint["accept"],
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": fingerprint["accept_language"],
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Origin": "https://glider.so",
            "Pragma": "no-cache",
            "Referer": "https://glider.so/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Sec-CH-UA": fingerprint["sec_ch_ua"] or '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{fingerprint["platform"]}"',
            "User-Agent": fingerprint["user_agent"],
        }
        self.session.headers.update(self.headers)
        self.chat = Chat(self)

    def convert_model_name(self, model: str) -> str:
        """
        Convert model names to ones supported by Glider.

        Args:
            model: Model name to convert

        Returns:
            Glider model name
        """
        # If the model is already a valid Glider model, return it
        if model in self.AVAILABLE_MODELS:
            return model

        # Default to the most capable model
        print(f"Warning: Unknown model '{model}'. Using 'chat-llama-3-1-70b' instead.")
        return "chat-llama-3-1-70b"
