import requests
import json
import re
import uuid
import time
from typing import Dict, Optional, Generator, Union, Any

from webscout.AIbase import AISearch
from webscout import exceptions
from webscout.litagent import LitAgent


class Response:
    """A wrapper class for SCIRA API responses.

    This class automatically converts response objects to their text representation
    when printed or converted to string.

    Attributes:
        text (str): The text content of the response

    Example:
        >>> response = Response("Hello, world!")
        >>> print(response)
        Hello, world!
        >>> str(response)
        'Hello, world!'
    """
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Scira(AISearch):
    """A class to interact with the SCIRA AI search API.

    SCIRA provides a powerful search interface that returns AI-generated responses
    based on web content. It supports both streaming and non-streaming responses.

    Basic Usage:
        >>> from webscout import Scira
        >>> ai = Scira()
        >>> # Non-streaming example
        >>> response = ai.search("What is Python?")
        >>> print(response)
        Python is a high-level programming language...

        >>> # Streaming example
        >>> for chunk in ai.search("Tell me about AI", stream=True):
        ...     print(chunk, end="", flush=True)
        Artificial Intelligence is...

        >>> # Raw response format
        >>> for chunk in ai.search("Hello", stream=True, raw=True):
        ...     print(chunk)
        {'text': 'Hello'}
        {'text': ' there!'}

    Args:
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
        proxies (dict, optional): Proxy configuration for requests. Defaults to None.
        model (str, optional): Model to use for the search. Defaults to "scira-default".
        group (str, optional): Group parameter. Defaults to "web".
    """

    AVAILABLE_MODELS = {
        "scira-default": "Grok3-mini", # thinking model
        "scira-grok-3": "Grok3",
        "scira-anthropic": "Sonnet 3.7 thinking",
        "scira-vision" : "Grok2-Vision", # vision model
        "scira-4.1-mini": "GPT4.1-mini",
        "scira-qwq": "QWQ-32B",
        "scira-o4-mini": "o4-mini",
        "scira-google": "gemini 2.5 flash"
    }
    def __init__(
        self,
        timeout: int = 60,
        proxies: Optional[dict] = None,
        model: str = "scira-default",
        deepsearch: bool = False
    ):
        """Initialize the SCIRA API client.

        Args:
            timeout (int, optional): Request timeout in seconds. Defaults to 60.
            proxies (dict, optional): Proxy configuration for requests. Defaults to None.
            model (str, optional): Model to use for the search. Defaults to "scira-default" (Grok3).
            deepsearch (bool, optional): Whether to use deep search mode. If True, uses "extreme" group for more comprehensive results. If False, uses "web" group for faster results. Defaults to False.

        Example:
            >>> ai = Scira(timeout=120)  # Longer timeout
            >>> ai = Scira(proxies={'http': 'http://proxy.com:8080'})  # With proxy
            >>> ai = Scira(model="scira-claude")  # Use Claude model
            >>> ai = Scira(deepsearch=True)  # Use deep search mode
        """
        # Validate model
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model: {model}. Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.session = requests.Session()
        self.api_endpoint = "https://scira.ai/api/search"
        self.timeout = timeout
        self.proxies = proxies
        self.model = model

        # Set group based on deepsearch parameter
        self.group = "extreme" if deepsearch else "web"
        self.last_response = {}

        # Set headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "User-Agent": LitAgent().random()
        }

        self.session.headers.update(self.headers)

    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[Response, Generator[Union[Dict[str, str], Response], None, None]]:
        """Search using the SCIRA API and get AI-generated responses.

        This method sends a search query to SCIRA and returns the AI-generated response.
        It supports both streaming and non-streaming modes, as well as raw response format.

        Args:
            prompt (str): The search query or prompt to send to the API.
            stream (bool, optional): If True, yields response chunks as they arrive.
                                   If False, returns complete response. Defaults to False.
            raw (bool, optional): If True, returns raw response dictionaries with 'text' key.
                                If False, returns Response objects that convert to text automatically.
                                Defaults to False.

        Returns:
            Union[Response, Generator[Union[Dict[str, str], Response], None, None]]:
                - If stream=False: Returns complete response
                - If stream=True: Yields response chunks as they arrive

        Raises:
            exceptions.APIConnectionError: If there's an issue connecting to the API
            exceptions.APIResponseError: If the API returns an error response

        Example:
            >>> ai = Scira()
            >>> # Non-streaming example
            >>> response = ai.search("What is Python?")
            >>> print(response)
            Python is a high-level programming language...

            >>> # Streaming example
            >>> for chunk in ai.search("Tell me about AI", stream=True):
            ...     print(chunk, end="", flush=True)
            Artificial Intelligence is...
        """
        # Create a unique message ID
        message_id = str(uuid.uuid4()).replace("-", "")[:16]
        self.user_id = str(uuid.uuid4()).replace("-", "")[:16]
        # Prepare the payload
        payload = {
            "id": message_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "parts": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "model": self.model,
            "group": self.group,
            "user_id": self.user_id,
            "timezone": "Asia/Calcutta"
        }

        try:
            # Send the request
            response = self.session.post(
                self.api_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                stream=True,
                timeout=self.timeout,
                proxies=self.proxies
            )

            # Check for successful response
            if response.status_code != 200:
                raise exceptions.APIResponseError(
                    f"API returned error status: {response.status_code}"
                )

            # Store the last response
            self.last_response = {"status_code": response.status_code}

            # Handle streaming response
            if stream:
                return self._handle_streaming_response(response, raw)

            # Handle non-streaming response
            return self._handle_non_streaming_response(response, raw)

        except requests.RequestException as e:
            raise exceptions.APIConnectionError(f"Error connecting to API: {str(e)}")

    def _handle_streaming_response(
        self,
        response: requests.Response,
        raw: bool
    ) -> Generator[Union[Dict[str, str], Response], None, None]:
        """Handle streaming response from the API.

        Args:
            response (requests.Response): The streaming response from the API
            raw (bool): Whether to return raw response dictionaries

        Yields:
            Union[Dict[str, str], Response]: Response chunks as they arrive
        """
        for line in response.iter_lines():
            if line:
                try:
                    # Decode the line
                    decoded_line = line.decode("utf-8")

                    # Check if this is a line starting with "0:" (content)
                    if re.match(r'^0:', decoded_line):
                        # Extract the content after "0:"
                        content = re.sub(r'^0:', '', decoded_line)
                        # Remove surrounding quotes if present
                        content = re.sub(r'^"(.*)"$', r'\1', content)
                        # Replace escaped newlines with actual newlines
                        content = content.replace('\\n', '\n')

                        if raw:
                            yield {"text": content}
                        else:
                            yield Response(content)
                except Exception:
                    # Skip lines that can't be processed
                    pass

    def _handle_non_streaming_response(
        self,
        response: requests.Response,
        raw: bool
    ) -> Union[Dict[str, str], Response]:
        """Handle non-streaming response from the API.

        Args:
            response (requests.Response): The response from the API
            raw (bool): Whether to return raw response dictionary

        Returns:
            Union[Dict[str, str], Response]: Complete response
        """
        full_text = ""

        for line in response.iter_lines():
            if line:
                try:
                    # Decode the line
                    decoded_line = line.decode("utf-8")

                    # Check if this is a line starting with "0:" (content)
                    if re.match(r'^0:', decoded_line):
                        # Extract the content after "0:"
                        content = re.sub(r'^0:', '', decoded_line)
                        # Remove surrounding quotes if present
                        content = re.sub(r'^"(.*)"$', r'\1', content)
                        # Replace escaped newlines with actual newlines
                        content = content.replace('\\n', '\n')
                        full_text += content
                except Exception:
                    # Skip lines that can't be processed
                    pass

        if raw:
            return {"text": full_text}
        else:
            return Response(full_text)

    @staticmethod
    def clean_content(text: str) -> str:
        """Clean the response content by removing unnecessary formatting.

        Args:
            text (str): The text to clean

        Returns:
            str: The cleaned text
        """
        # Remove any extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', text)
        # Remove any trailing whitespace
        cleaned_text = cleaned_text.strip()

        return cleaned_text


if __name__ == "__main__":
    from rich import print
    ai = Scira()
    user_query = input(">>> ")
    response = ai.search(user_query, stream=True, raw=False)
    for chunk in response:
        print(chunk, end="", flush=True)
