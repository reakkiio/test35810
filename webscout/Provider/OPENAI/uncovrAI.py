import time
import uuid
import re
import json
import cloudscraper
from typing import List, Dict, Optional, Union, Generator, Any

from webscout.litagent import LitAgent
from .base import BaseChat, BaseCompletions, OpenAICompatibleProvider
from .utils import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChatCompletionMessage,
    ChoiceDelta,
    CompletionUsage,
    format_prompt,
    get_system_prompt,
    get_last_user_message
)

# ANSI escape codes for formatting
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"

class Completions(BaseCompletions):
    def __init__(self, client: 'UncovrAI'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Create a chat completion using the UncovrAI API.

        Args:
            model: The model to use for completion
            messages: A list of messages in the conversation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            temperature: Controls randomness (mapped to UncovrAI's temperature)
            top_p: Controls diversity (not directly used by UncovrAI)
            **kwargs: Additional parameters

        Returns:
            A ChatCompletion object or a generator of ChatCompletionChunk objects
        """
        # Validate model
        if model not in self._client.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self._client.AVAILABLE_MODELS}")

        # Map temperature to UncovrAI's scale (0-100)
        # Default to 32 (medium) if not provided
        uncovr_temperature = 32
        if temperature is not None:
            # Map from 0-1 scale to 0-100 scale
            uncovr_temperature = int(temperature * 100)
            # Ensure it's within bounds
            uncovr_temperature = max(0, min(100, uncovr_temperature))

        # Map creativity from kwargs or use default
        creativity = kwargs.get("creativity", "medium")

        # Get focus and tools from kwargs or use defaults
        selected_focus = kwargs.get("selected_focus", ["web"])
        selected_tools = kwargs.get("selected_tools", ["quick-cards"])

        # Generate request ID and timestamp
        request_id = str(uuid.uuid4())
        created_time = int(time.time())

        # Format the conversation using utility functions
        conversation_prompt = format_prompt(messages, add_special_tokens=False, do_continue=True)

        # Prepare the request payload
        payload = {
            "content": conversation_prompt,
            "chatId": self._client.chat_id,
            "userMessageId": str(uuid.uuid4()),
            "ai_config": {
                "selectedFocus": selected_focus,
                "selectedTools": selected_tools,
                "agentId": "chat",
                "modelId": model,
                "temperature": uncovr_temperature,
                "creativity": creativity
            }
        }

        # Handle streaming response
        if stream:
            return self._handle_streaming_response(
                payload=payload,
                model=model,
                request_id=request_id,
                created_time=created_time
            )

        # Handle non-streaming response
        return self._handle_non_streaming_response(
            payload=payload,
            model=model,
            request_id=request_id,
            created_time=created_time
        )

    def _handle_streaming_response(
        self,
        *,
        payload: Dict[str, Any],
        model: str,
        request_id: str,
        created_time: int
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Handle streaming response from UncovrAI API."""
        try:
            with self._client.session.post(
                self._client.url,
                json=payload,
                stream=True,
                timeout=self._client.timeout
            ) as response:
                if response.status_code != 200:
                    # If we get a non-200 response, try refreshing our identity once
                    if response.status_code in [403, 429]:
                        self._client.refresh_identity()
                        # Retry with new identity
                        with self._client.session.post(
                            self._client.url,
                            json=payload,
                            stream=True,
                            timeout=self._client.timeout
                        ) as retry_response:
                            if not retry_response.ok:
                                raise IOError(
                                    f"Failed to generate response after identity refresh - "
                                    f"({retry_response.status_code}, {retry_response.reason}) - "
                                    f"{retry_response.text}"
                                )
                            response = retry_response
                    else:
                        raise IOError(f"Request failed with status code {response.status_code}")

                # Process the streaming response
                streaming_text = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            line = line.decode('utf-8')

                            # Use regex to match content messages
                            content_match = re.match(r'^0:\s*"?(.*?)"?$', line)
                            if content_match:  # Content message
                                content = content_match.group(1)
                                # Format the content to handle escape sequences
                                content = self._client.format_text(content)
                                streaming_text += content

                                # Create a chunk for this part of the response
                                delta = ChoiceDelta(content=content)
                                choice = Choice(
                                    index=0,
                                    delta=delta,
                                    finish_reason=None
                                )
                                chunk = ChatCompletionChunk(
                                    id=request_id,
                                    choices=[choice],
                                    created=created_time,
                                    model=model
                                )

                                yield chunk

                            # Check for error messages
                            error_match = re.match(r'^2:\[{"type":"error","error":"(.*?)"}]$', line)
                            if error_match:
                                error_msg = error_match.group(1)
                                raise IOError(f"API Error: {error_msg}")

                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

                # Yield a final chunk with finish_reason="stop"
                delta = ChoiceDelta()
                choice = Choice(
                    index=0,
                    delta=delta,
                    finish_reason="stop"
                )
                chunk = ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=created_time,
                    model=model
                )
                yield chunk

        except Exception as e:
            print(f"{RED}Error during UncovrAI streaming request: {e}{RESET}")
            raise IOError(f"UncovrAI streaming request failed: {e}") from e

    def _handle_non_streaming_response(
        self,
        *,
        payload: Dict[str, Any],
        model: str,
        request_id: str,
        created_time: int
    ) -> ChatCompletion:
        """Handle non-streaming response from UncovrAI API."""
        try:
            response = self._client.session.post(
                self._client.url,
                json=payload,
                timeout=self._client.timeout
            )

            if response.status_code != 200:
                if response.status_code in [403, 429]:
                    self._client.refresh_identity()
                    response = self._client.session.post(
                        self._client.url,
                        json=payload,
                        timeout=self._client.timeout
                    )
                    if not response.ok:
                        raise IOError(
                            f"Failed to generate response after identity refresh - "
                            f"({response.status_code}, {response.reason}) - "
                            f"{response.text}"
                        )
                else:
                    raise IOError(f"Request failed with status code {response.status_code}")

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode('utf-8')
                        content_match = re.match(r'^0:\s*"?(.*?)"?$', line)
                        if content_match:
                            content = content_match.group(1)
                            full_response += content

                        # Check for error messages
                        error_match = re.match(r'^2:\[{"type":"error","error":"(.*?)"}]$', line)
                        if error_match:
                            error_msg = error_match.group(1)
                            raise IOError(f"API Error: {error_msg}")

                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

            # Format the full response to handle escape sequences
            full_response = self._client.format_text(full_response)

            # Create message, choice, and usage objects
            message = ChatCompletionMessage(
                role="assistant",
                content=full_response
            )

            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )

            # Estimate token usage (this is approximate)
            prompt_tokens = len(payload["content"]) // 4
            completion_tokens = len(full_response) // 4
            total_tokens = prompt_tokens + completion_tokens

            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )

            # Create the completion object
            completion = ChatCompletion(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                usage=usage,
            )

            return completion

        except Exception as e:
            print(f"{RED}Error during UncovrAI non-stream request: {e}{RESET}")
            raise IOError(f"UncovrAI request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'UncovrAI'):
        self.completions = Completions(client)

class UncovrAI(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for Uncovr AI API.

    Usage:
        client = UncovrAI()
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """

    AVAILABLE_MODELS = [
        "default",
        "gpt-4o-mini",
        "gemini-2-flash",
        "gemini-2-flash-lite",
        "groq-llama-3-1-8b",
        "o3-mini",
        "deepseek-r1-distill-qwen-32b",
        # The following models are not available in the free plan:
        # "claude-3-7-sonnet",
        # "gpt-4o",
        # "claude-3-5-sonnet-v2",
        # "deepseek-r1-distill-llama-70b",
        # "gemini-2-flash-lite-preview",
        # "qwen-qwq-32b"
    ]

    def __init__(
        self,
        timeout: int = 30,
        browser: str = "chrome",
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None,
        proxies: dict = {}
    ):
        """
        Initialize the UncovrAI client.

        Args:
            timeout: Request timeout in seconds
            browser: Browser name for LitAgent to generate fingerprint
            chat_id: Optional chat ID (will generate one if not provided)
            user_id: Optional user ID (will generate one if not provided)
            proxies: Optional proxy configuration
        """
        self.url = "https://uncovr.app/api/workflows/chat"
        self.timeout = timeout

        # Initialize LitAgent for user agent generation
        self.agent = LitAgent()

        # Use fingerprinting to create a consistent browser identity
        self.fingerprint = self.agent.generate_fingerprint(browser)

        # Use the fingerprint for headers
        self.headers = {
            "Accept": self.fingerprint["accept"],
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": self.fingerprint["accept_language"],
            "Content-Type": "application/json",
            "Origin": "https://uncovr.app",
            "Referer": "https://uncovr.app/",
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"] or '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }

        # Use cloudscraper to bypass Cloudflare protection
        self.session = cloudscraper.create_scraper()
        self.session.headers.update(self.headers)
        self.session.proxies.update(proxies)

        # Set chat and user IDs
        self.chat_id = chat_id or str(uuid.uuid4())
        self.user_id = user_id or f"user_{str(uuid.uuid4())[:8].upper()}"

        # Initialize chat interface
        self.chat = Chat(self)

    def refresh_identity(self, browser: str = None):
        """
        Refreshes the browser identity fingerprint.

        Args:
            browser: Specific browser to use for the new fingerprint
        """
        browser = browser or self.fingerprint.get("browser_type", "chrome")
        self.fingerprint = self.agent.generate_fingerprint(browser)

        # Update headers with new fingerprint
        self.headers.update({
            "Accept": self.fingerprint["accept"],
            "Accept-Language": self.fingerprint["accept_language"],
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"] or self.headers["Sec-CH-UA"],
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
        })

        # Update session headers
        for header, value in self.headers.items():
            self.session.headers[header] = value

        return self.fingerprint

    def format_text(self, text: str) -> str:
        """
        Format text by replacing escaped newlines with actual newlines.

        Args:
            text: Text to format

        Returns:
            Formatted text
        """
        # Use a more comprehensive approach to handle all escape sequences
        try:
            # First handle double backslashes to avoid issues
            text = text.replace('\\\\', '\\')

            # Handle common escape sequences
            text = text.replace('\\n', '\n')
            text = text.replace('\\r', '\r')
            text = text.replace('\\t', '\t')
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")

            # Handle any remaining escape sequences using JSON decoding
            try:
                # Add quotes to make it a valid JSON string
                json_str = f'"{text}"'
                # Use json module to decode all escape sequences
                decoded = json.loads(json_str)
                return decoded
            except json.JSONDecodeError:
                # If JSON decoding fails, return the text with the replacements we've already done
                return text
        except Exception as e:
            # If any error occurs, return the original text
            print(f"{RED}Warning: Error formatting text: {e}{RESET}")
            return text
