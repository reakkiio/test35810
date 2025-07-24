import requests
import json
from typing import Any, Dict, Generator, Optional, Union

from webscout.AIbase import AISearch, SearchResponse
from webscout import exceptions
from webscout.litagent import LitAgent
from webscout.AIutel import sanitize_stream


class PERPLEXED(AISearch):
    """A class to interact with the PERPLEXED stream search API.
    
    PERPLEXED provides an AI-powered search interface that returns emotionally intelligent
    responses based on web content. It supports both streaming and non-streaming responses.
    
    Basic Usage:
        >>> from webscout import PERPLEXED
        >>> ai = PERPLEXED()
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
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        proxies (dict, optional): Proxy configuration for requests. Defaults to None.
    
    Attributes:
        api_endpoint (str): The PERPLEXED API endpoint URL.
        stream_chunk_size (int): Size of chunks when streaming responses.
        timeout (int): Request timeout in seconds.
        headers (dict): HTTP headers used in requests.
    """

    def __init__(
        self,
        timeout: int = 30,
        proxies: Optional[dict] = None,
    ):
        """Initialize the PERPLEXED API client.
        
        Args:
            timeout (int, optional): Request timeout in seconds. Defaults to 30.
            proxies (dict, optional): Proxy configuration for requests. Defaults to None.
        
        Example:
            >>> ai = PERPLEXED(timeout=60)  # Longer timeout
            >>> ai = PERPLEXED(proxies={'http': 'http://proxy.com:8080'})  # With proxy
        """
        self.session = requests.Session()
        self.api_endpoint = "https://d21l5c617zttgr.cloudfront.net/stream_search"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://d37ozmhmvu2kcg.cloudfront.net",
            "referer": "https://d37ozmhmvu2kcg.cloudfront.net/",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "sec-gpc": "1",
            "user-agent": LitAgent().random()
        }
        self.session.headers.update(self.headers)
        self.proxies = proxies

    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]:
        """Search using the PERPLEXED API and get AI-generated responses.
        
        This method sends a search query to PERPLEXED and returns the AI-generated response.
        It supports both streaming and non-streaming modes, as well as raw response format.
        
        Args:
            prompt (str): The search query or prompt to send to the API.
            stream (bool, optional): If True, yields response chunks as they arrive.
                                   If False, returns complete response. Defaults to False.
            raw (bool, optional): If True, returns raw response dictionaries with 'text' key.
                                If False, returns SearchResponse objects that convert to text automatically.
                                Defaults to False.
        
        Returns:
            Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]: 
                - If stream=False: Returns complete response as SearchResponse object
                - If stream=True: Yields response chunks as either Dict or SearchResponse objects
        
        Raises:
            APIConnectionError: If the API request fails
        
        Examples:
            Basic search:
            >>> ai = PERPLEXED()
            >>> response = ai.search("What is Python?")
            >>> print(response)
            Python is a programming language...
            
            Streaming response:
            >>> for chunk in ai.search("Tell me about AI", stream=True):
            ...     print(chunk, end="")
            Artificial Intelligence...
            
            Raw response format:
            >>> for chunk in ai.search("Hello", stream=True, raw=True):
            ...     print(chunk)
            {'text': 'Hello'}
            {'text': ' there!'}
            
            Error handling:
            >>> try:
            ...     response = ai.search("My question")
            ... except exceptions.APIConnectionError as e:
            ...     print(f"API error: {e}")
        """
        payload = {
            "user_prompt": prompt
        }

        def extract_answer_content(data):
            """Extract answer content from PERPLEXED response."""
            print(f"[DEBUG] extract_answer_content received: {type(data)}")
            if isinstance(data, dict):
                print(f"[DEBUG] Dict keys: {list(data.keys())}")
                print(f"[DEBUG] success: {data.get('success')}")
                print(f"[DEBUG] stage: {data.get('stage')}")
                print(f"[DEBUG] answer present: {'answer' in data}")
                answer_val = data.get('answer', 'NOT_FOUND')
                print(f"[DEBUG] answer value: {repr(answer_val[:100] if isinstance(answer_val, str) and len(answer_val) > 100 else answer_val)}")
                
                # Check if this is the final answer - answer field exists and is not empty
                if data.get("success") and "answer" in data and data["answer"]:
                    print(f"[DEBUG] Returning answer content (length: {len(data['answer'])})")
                    return data["answer"]
                # Check if this is a stage update with no answer yet
                elif data.get("success") and data.get("stage"):
                    print(f"[DEBUG] Skipping stage update: {data.get('stage')}")
                    return None  # Skip stage updates without answers
                else:
                    print(f"[DEBUG] No matching condition, returning None")
            else:
                print(f"[DEBUG] Data is not dict, returning None")
            return None

        def for_stream():
            try:
                with self.session.post(
                    self.api_endpoint,
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    proxies=self.proxies,
                ) as response:
                    if not response.ok:
                        raise exceptions.APIConnectionError(
                            f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                        )

                    # Process the streaming response manually
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            full_response += line

                    # Split by the separator to get individual JSON chunks
                    chunks = full_response.split("[/PERPLEXED-SEPARATOR]")
                    
                    for chunk_text in chunks:
                        if chunk_text.strip():
                            try:
                                # Parse the JSON chunk
                                chunk_data = json.loads(chunk_text.strip())
                                
                                if raw:
                                    # For raw mode, yield the entire JSON string
                                    yield {"text": chunk_text.strip()}
                                else:
                                    # For non-raw mode, extract the answer if available
                                    answer_content = extract_answer_content(chunk_data)
                                    if answer_content:
                                        yield SearchResponse(answer_content)
                                        
                            except json.JSONDecodeError:
                                # Skip invalid JSON chunks
                                continue

            except requests.exceptions.RequestException as e:
                raise exceptions.APIConnectionError(f"Request failed: {e}")

        def for_non_stream():
            if raw:
                # For raw mode, yield each chunk as it comes
                for chunk in for_stream():
                    yield chunk
            else:
                # For non-raw mode, accumulate all chunks and return final response
                full_response = ""
                for chunk in for_stream():
                    full_response += str(chunk)
                
                if full_response:
                    self.last_response = SearchResponse(full_response)
                else:
                    # Return empty response if no content was extracted
                    self.last_response = SearchResponse("")
                
                return self.last_response
        
        if stream:
            return for_stream()
        else:
            # For non-streaming mode, we need to consume the generator and return the result
            result = for_non_stream()
            # If result is a generator (which it shouldn't be), consume it
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                try:
                    # This shouldn't happen with our current implementation, but just in case
                    return list(result)[0] if list(result) else SearchResponse("")
                except:
                    return SearchResponse("")
            return result


if __name__ == "__main__":
    from rich import print
    ai = PERPLEXED()
    
    # Test with raw=False to see debug output
    print("=== Testing with raw=False ===")
    response = ai.search(input(">>> "), stream=False, raw=False)
    print("Final response:", response)
    print("Response type:", type(response))