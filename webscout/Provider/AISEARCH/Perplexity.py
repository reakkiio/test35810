import json
import random
from uuid import uuid4
from typing import Dict, Optional, Generator, Union, Any
from curl_cffi import requests

from webscout.AIbase import AISearch, SearchResponse
from webscout import exceptions
from webscout.litagent import LitAgent


class Perplexity(AISearch):
    """A class to interact with the Perplexity AI search API.
    
    Perplexity provides a powerful search interface that returns AI-generated responses
    based on web content. It supports both streaming and non-streaming responses,
    multiple search modes, and model selection.
    
    Basic Usage:
        >>> from webscout import Perplexity
        >>> ai = Perplexity()
        >>> # Non-streaming example
        >>> response = ai.search("What is Python?")
        >>> print(response)
        Python is a high-level programming language...
        
        >>> # Streaming example
        >>> for chunk in ai.search("Tell me about AI", stream=True):
        ...     print(chunk, end="", flush=True)
        Artificial Intelligence is...
        
        >>> # Pro search with specific model (requires authentication via cookies)
        >>> cookies = {"perplexity-user": "your_cookie_value"}
        >>> ai_pro = Perplexity(cookies=cookies)
        >>> response = ai_pro.search("Latest AI research", mode="pro", model="gpt-4o")
        >>> print(response)
        
        >>> # Raw response format
        >>> for chunk in ai.search("Hello", stream=True, raw=True):
        ...     print(chunk)
        {'text': 'Hello'}
        {'text': ' there!'}
    
    Args:
        cookies (dict, optional): Cookies to use for authentication. Defaults to None.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
        proxies (dict, optional): Proxy configuration for requests. Defaults to None.
    """
    
    def __init__(
        self,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        proxies: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Perplexity client.
        
        Args:
            cookies (dict, optional): Cookies to use for authentication. Defaults to None.
            timeout (int, optional): Request timeout in seconds. Defaults to 60.
            proxies (dict, optional): Proxy configuration for requests. Defaults to None.
        """
        self.timeout = timeout
        self.agent = LitAgent()
        self.session = requests.Session(headers={
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'max-age=0',
            'dnt': '1',
            'priority': 'u=0, i',
            'sec-ch-ua': '"Not;A=Brand";v="24", "Chromium";v="128"',
            'sec-ch-ua-arch': '"x86"',
            'sec-ch-ua-bitness': '"64"',
            'sec-ch-ua-full-version': '"128.0.6613.120"',
            'sec-ch-ua-full-version-list': '"Not;A=Brand";v="24.0.0.0", "Chromium";v="128.0.6613.120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua-platform-version': '"19.0.0"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': self.agent.random(),
        }, cookies=cookies or {}, impersonate='chrome')
        
        # Apply proxies if provided
        if proxies:
            self.session.proxies.update(proxies)
        
        # Initialize session with socket.io
        self.timestamp = format(random.getrandbits(32), '08x')
        
        # Get socket.io session ID
        response = self.session.get(f'https://www.perplexity.ai/socket.io/?EIO=4&transport=polling&t={self.timestamp}')
        self.sid = json.loads(response.text[1:])['sid']
        
        # Initialize socket.io connection
        assert (self.session.post(
            f'https://www.perplexity.ai/socket.io/?EIO=4&transport=polling&t={self.timestamp}&sid={self.sid}',
            data='40{"jwt":"anonymous-ask-user"}'
        )).text == 'OK'
        
        # Get session info
        self.session.get('https://www.perplexity.ai/api/auth/session')
        
        # Set default values
        self.copilot = 0 if not cookies else float('inf')
        self.file_upload = 0 if not cookies else float('inf')
    
    def _extract_answer(self, response):
        """
        Extract the answer from the response.
        
        Args:
            response (dict): The response from Perplexity AI.
            
        Returns:
            str: The extracted answer text.
        """
        if not response:
            return ""
        
        # Find the FINAL step in the text array
        final_step = None
        if 'text' in response and isinstance(response['text'], list):
            for step in response['text']:
                if step.get('step_type') == 'FINAL' and 'content' in step and 'answer' in step['content']:
                    final_step = step
                    break
        
        if not final_step:
            return ""
        
        try:
            # Parse the answer JSON string
            answer_json = json.loads(final_step['content']['answer'])
            return answer_json.get('answer', '')
        except (json.JSONDecodeError, KeyError):
            return ""
    
    def search(
        self,
        prompt: str,
        mode: str = 'auto',
        model: Optional[str] = None,
        sources: Optional[list] = None,
        stream: bool = False,
        raw: bool = False,
        language: str = 'en-US',
        follow_up: Optional[Dict[str, Any]] = None,
        incognito: bool = False
    ) -> Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]:
        """Search using the Perplexity API and get AI-generated responses.
        
        This method sends a search query to Perplexity and returns the AI-generated response.
        It supports both streaming and non-streaming modes, as well as raw response format.
        
        Args:
            prompt (str): The search query or prompt to send to the API.
            mode (str, optional): Search mode. Options: 'auto', 'pro', 'reasoning', 'deep research'. 
                                Defaults to 'auto'.
            model (str, optional): Model to use. Available models depend on the mode. Defaults to None.
            sources (list, optional): Sources to use. Options: 'web', 'scholar', 'social'. 
                                    Defaults to ['web'].
            stream (bool, optional): If True, yields response chunks as they arrive.
                                   If False, returns complete response. Defaults to False.
            raw (bool, optional): If True, returns raw response dictionaries.
                                If False, returns Response objects that convert to text automatically.
                                Defaults to False.
            language (str, optional): Language to use. Defaults to 'en-US'.
            follow_up (dict, optional): Follow-up information. Defaults to None.
            incognito (bool, optional): Whether to use incognito mode. Defaults to False.
            
        Returns:
            If stream=True: Generator yielding response chunks as they arrive
            If stream=False: Complete response
            
        Raises:
            ValueError: If invalid mode or model is provided
            exceptions.APIConnectionError: If connection to API fails
            exceptions.FailedToGenerateResponseError: If response generation fails
        """
        if sources is None:
            sources = ['web']
        
        # Validate inputs
        if mode not in ['auto', 'pro', 'reasoning', 'deep research']:
            raise ValueError('Search modes -> ["auto", "pro", "reasoning", "deep research"]')
        
        if not all([source in ('web', 'scholar', 'social') for source in sources]):
            raise ValueError('Sources -> ["web", "scholar", "social"]')
        
        # Check if model is valid for the selected mode
        valid_models = {
            'auto': [None],
            'pro': [None, 'sonar', 'gpt-4.5', 'gpt-4o', 'claude 3.7 sonnet', 'gemini 2.0 flash', 'grok-2'],
            'reasoning': [None, 'r1', 'o3-mini', 'claude 3.7 sonnet'],
            'deep research': [None]
        }
        
        if mode in valid_models and model not in valid_models[mode] and model is not None:
            raise ValueError(f'Invalid model for {mode} mode. Valid models: {valid_models[mode]}')
        
        # Prepare request data
        json_data = {
            'query_str': prompt,
            'params': {
                'attachments': follow_up['attachments'] if follow_up else [],
                'frontend_context_uuid': str(uuid4()),
                'frontend_uuid': str(uuid4()),
                'is_incognito': incognito,
                'language': language,
                'last_backend_uuid': follow_up['backend_uuid'] if follow_up else None,
                'mode': 'concise' if mode == 'auto' else 'copilot',
                'model_preference': {
                    'auto': {
                        None: 'turbo'
                    },
                    'pro': {
                        None: 'pplx_pro',
                        'sonar': 'experimental',
                        'gpt-4.5': 'gpt45',
                        'gpt-4o': 'gpt4o',
                        'claude 3.7 sonnet': 'claude2',
                        'gemini 2.0 flash': 'gemini2flash',
                        'grok-2': 'grok'
                    },
                    'reasoning': {
                        None: 'pplx_reasoning',
                        'r1': 'r1',
                        'o3-mini': 'o3mini',
                        'claude 3.7 sonnet': 'claude37sonnetthinking'
                    },
                    'deep research': {
                        None: 'pplx_alpha'
                    }
                }[mode][model],
                'source': 'default',
                'sources': sources,
                'version': '2.18'
            }
        }
        
        try:
            # Make the request
            resp = self.session.post(
                'https://www.perplexity.ai/rest/sse/perplexity_ask', 
                json=json_data, 
                stream=True,
                timeout=self.timeout
            )
            
            if resp.status_code != 200:
                raise exceptions.APIConnectionError(f"API returned status code {resp.status_code}")
            
            # Define streaming response handler
            def stream_response():
                for chunk in resp.iter_lines(delimiter=b'\r\n\r\n'):
                    content = chunk.decode('utf-8')
                    if content.startswith('event: message\r\n'):
                        content_json = json.loads(content[len('event: message\r\ndata: '):])
                        if 'text' in content_json:
                            try:
                                # If text is a string, try to parse it as JSON
                                if isinstance(content_json['text'], str):
                                    content_json['text'] = json.loads(content_json['text'])
                            except json.JSONDecodeError:
                                pass
                        
                        if raw:
                            yield content_json
                        else:
                            # For non-raw responses, extract text from each chunk
                            if 'text' in content_json and isinstance(content_json['text'], list):
                                for step in content_json['text']:
                                    if step.get('type') == 'answer' and 'value' in step:
                                        yield SearchResponse(step['value'])
                                    elif step.get('type') == 'thinking' and 'value' in step:
                                        yield SearchResponse(step['value'])
                    elif content.startswith('event: end_of_stream\r\n'):
                        return
            
            # Handle streaming or non-streaming response
            if stream:
                return stream_response()
            else:
                chunks = []
                final_response = None
                
                for chunk in resp.iter_lines(delimiter=b'\r\n\r\n'):
                    content = chunk.decode('utf-8')
                    if content.startswith('event: message\r\n'):
                        content_json = json.loads(content[len('event: message\r\ndata: '):])
                        if 'text' in content_json:
                            try:
                                # If text is a string, try to parse it as JSON
                                if isinstance(content_json['text'], str):
                                    content_json['text'] = json.loads(content_json['text'])
                            except json.JSONDecodeError:
                                pass
                        chunks.append(content_json)
                        final_response = content_json
                    elif content.startswith('event: end_of_stream\r\n'):
                        # Process the final response to extract the answer
                        if final_response:
                            answer_text = self._extract_answer(final_response)
                            return SearchResponse(answer_text) if not raw else final_response
                        elif chunks:
                            answer_text = self._extract_answer(chunks[-1])
                            return SearchResponse(answer_text) if not raw else chunks[-1]
                        else:
                            return SearchResponse("") if not raw else {}
                
                # If we get here, something went wrong
                raise exceptions.FailedToGenerateResponseError("Failed to get complete response")
                
        except requests.RequestsError as e:
            raise exceptions.APIConnectionError(f"Connection error: {str(e)}")
        except json.JSONDecodeError:
            raise exceptions.FailedToGenerateResponseError("Failed to parse response JSON")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Error: {str(e)}")


if __name__ == "__main__":
    # Simple test
    ai = Perplexity()
    response = ai.search("What is Python?")
    print(response)
    