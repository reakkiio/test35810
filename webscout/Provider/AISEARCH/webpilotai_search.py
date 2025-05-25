import requests
import json
import re
from typing import Dict, Optional, Generator, Union, Any

from webscout.AIbase import AISearch, SearchResponse
from webscout import exceptions
from webscout.litagent import LitAgent


class webpilotai(AISearch):
    """A class to interact with the webpilotai (WebPilot) AI search API.
    
    webpilotai provides a web-based comprehensive search SearchResponse interface that returns AI-generated 
    SearchResponses with source references and related questions. It supports both streaming and
    non-streaming SearchResponses.
    
    Basic Usage:
        >>> from webscout import webpilotai
        >>> ai = webpilotai()
        >>> # Non-streaming example
        >>> response = ai.search("What is Python?")
        >>> print(response)
        Python is a high-level programming language...
        
        >>> # Streaming example
        >>> for chunk in ai.search("Tell me about AI", stream=True):
        ...     print(chunk, end="", flush=True)
        Artificial Intelligence is...
        
        >>> # Raw SearchResponse format
        >>> for chunk in ai.search("Hello", stream=True, raw=True):
        ...     print(chunk)
        {'text': 'Hello'}
        {'text': ' there!'}
    
    Args:
        timeout (int, optional): Request timeout in seconds. Defaults to 90.
        proxies (dict, optional): Proxy configuration for requests. Defaults to None.
    """

    def __init__(
        self,
        timeout: int = 90,
        proxies: Optional[dict] = None,
    ):
        """Initialize the webpilotai API client.
        
        Args:
            timeout (int, optional): Request timeout in seconds. Defaults to 90.
            proxies (dict, optional): Proxy configuration for requests. Defaults to None.
        
        Example:
            >>> ai = webpilotai(timeout=120)  # Longer timeout
            >>> ai = webpilotai(proxies={'http': 'http://proxy.com:8080'})  # With proxy
        """
        self.session = requests.Session()
        self.api_endpoint = "https://api.webpilotai.com/rupee/v1/search"
        self.timeout = timeout
        self.last_SearchResponse = {}
        
        # The 'Bearer null' is part of the API's expected headers
        self.headers = {
            'Accept': 'application/json, text/plain, */*, text/event-stream',
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': 'Bearer null',
            'Origin': 'https://www.webpilot.ai',
            'Referer': 'https://www.webpilot.ai/',
            'User-Agent': LitAgent().random(),
        }
        
        self.session.headers.update(self.headers)
        self.proxies = proxies

    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]:
        """Search using the webpilotai API and get AI-generated SearchResponses.
        
        This method sends a search query to webpilotai and returns the AI-generated SearchResponse.
        It supports both streaming and non-streaming modes, as well as raw SearchResponse format.
        
        Args:
            prompt (str): The search query or prompt to send to the API.
            stream (bool, optional): If True, yields SearchResponse chunks as they arrive.
                                   If False, returns complete SearchResponse. Defaults to False.
            raw (bool, optional): If True, returns raw SearchResponse dictionaries with 'text' key.
                                If False, returns SearchResponse objects that convert to text automatically.
                                Defaults to False.
        
        Returns:
            Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]: 
                - If stream=False: Returns complete SearchResponse as SearchResponse object
                - If stream=True: Yields SearchResponse chunks as either Dict or SearchResponse objects
        
        Raises:
            APIConnectionError: If the API request fails
        
        Examples:
            Basic search:
            >>> ai = webpilotai()
            >>> response = ai.search("What is Python?")
            >>> print(response)
            Python is a programming language...
            
            Streaming SearchResponse:
            >>> for chunk in ai.search("Tell me about AI", stream=True):
            ...     print(chunk, end="")
            Artificial Intelligence...
            
            Raw SearchResponse format:
            >>> for chunk in ai.search("Hello", stream=True, raw=True):
            ...     print(chunk)
            {'text': 'Hello'}
            {'text': ' there!'}
        """
        payload = {
            "q": prompt,
            "threadId": ""  # Empty for new search
        }
        
        def for_stream():
            full_SearchResponse_content = ""
            current_event_name = None
            current_data_buffer = []
            
            try:
                with self.session.post(
                    self.api_endpoint,
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    proxies=self.proxies
                ) as response:
                    if not response.ok:
                        raise exceptions.APIConnectionError(
                            f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                        )
                    
                    # Process the stream line by line
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:  # Empty line indicates end of an event
                            if current_data_buffer:
                                # Process the completed event
                                full_data = "\n".join(current_data_buffer)
                                if current_event_name == "message":
                                    try:
                                        data_payload = json.loads(full_data)
                                        # Check structure based on the API SearchResponse
                                        if data_payload.get('type') == 'data':
                                            content_chunk = data_payload.get('data', {}).get('content', "")
                                            if content_chunk:
                                                full_SearchResponse_content += content_chunk
                                                
                                                # Yield the new content chunk
                                                if raw:
                                                    yield {"text": content_chunk}
                                                else:
                                                    yield SearchResponse(content_chunk)
                                    except json.JSONDecodeError:
                                        pass
                                    except Exception as e:
                                        # Handle exceptions gracefully in stream processing
                                        pass
                                
                                # Reset for the next event
                                current_event_name = None
                                current_data_buffer = []
                            continue
                        
                        # Parse SSE fields
                        if line.startswith('event:'):
                            current_event_name = line[len('event:'):].strip()
                        elif line.startswith('data:'):
                            data_part = line[len('data:'):]
                            # Remove leading space if present (common in SSE)
                            if data_part.startswith(' '):
                                data_part = data_part[1:]
                            current_data_buffer.append(data_part)
                    
                    # Process any remaining data in buffer if stream ended without blank line
                    if current_data_buffer and current_event_name == "message":
                        try:
                            full_data = "\n".join(current_data_buffer)
                            data_payload = json.loads(full_data)
                            if data_payload.get('type') == 'data':
                                content_chunk = data_payload.get('data', {}).get('content', "")
                                if content_chunk and len(content_chunk) > len(full_SearchResponse_content):
                                    delta = content_chunk[len(full_SearchResponse_content):]
                                    full_SearchResponse_content += delta
                                    
                                    if raw:
                                        yield {"text": delta}
                                    else:
                                        yield SearchResponse(delta)
                        except (json.JSONDecodeError, Exception):
                            pass
                
            except requests.exceptions.Timeout:
                raise exceptions.APIConnectionError("Request timed out")
            except requests.exceptions.RequestException as e:
                raise exceptions.APIConnectionError(f"Request failed: {e}")

        def for_non_stream():
            full_SearchResponse = ""
            for chunk in for_stream():
                if raw:
                    yield chunk
                else:
                    full_SearchResponse += str(chunk)
            
            if not raw:
                # Format the SearchResponse for better readability
                formatted_SearchResponse = self.format_SearchResponse(full_SearchResponse)
                self.last_response = SearchResponse(formatted_SearchResponse)
                return self.last_SearchResponse

        return for_stream() if stream else for_non_stream()
    
    @staticmethod
    def format_SearchResponse(text: str) -> str:
        """Format the SearchResponse text for better readability.
        
        Args:
            text (str): The raw SearchResponse text
        
        Returns:
            str: Formatted text with improved structure
        """
        # Clean up formatting
        # Remove excessive newlines
        clean_text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure consistent spacing around sections
        clean_text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1\n\n\2', clean_text)
        
        # Clean up any leftover HTML or markdown artifacts
        clean_text = re.sub(r'<[^>]*>', '', clean_text)
        
        # Remove trailing whitespace on each line
        clean_text = '\n'.join(line.rstrip() for line in clean_text.split('\n'))
        
        return clean_text.strip()


if __name__ == "__main__":
    from rich import print
    
    ai = webpilotai()
    r = ai.search(input(">>> "), stream=True, raw=False)
    for chunk in r:
        print(chunk, end="", flush=True)
