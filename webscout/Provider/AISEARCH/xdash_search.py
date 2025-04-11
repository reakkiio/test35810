import json
import re
import requests
from urllib.parse import quote
from typing import Dict, Optional, Generator, Union, Any

from webscout.AIbase import AISearch
from webscout import exceptions
from webscout.litagent import LitAgent


class Response:
    """A wrapper class for XDash API responses.
    
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


class XDash(AISearch):
    """A class to interact with the XDash AI search API.
    
    XDash provides a powerful search interface that returns AI-generated responses
    based on web content with additional related questions.
    
    Basic Usage:
        >>> from webscout import XDash
        >>> ai = XDash()
        >>> # Non-streaming example
        >>> response = ai.search("What is Python?")
        >>> print(response)
        Python is a high-level programming language...
        
        >>> # Raw response format
        >>> response = ai.search("Hello", raw=True)
        >>> print(response)
        {'result': 'Hello there!'}
    
    Args:
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        proxies (dict, optional): Proxy configuration for requests. Defaults to None.
    """

    def __init__(
        self,
        timeout: int = 30,
        proxies: Optional[dict] = None,
    ):
        """Initialize the XDash API client.
        
        Args:
            timeout (int, optional): Request timeout in seconds. Defaults to 30.
            proxies (dict, optional): Proxy configuration for requests. Defaults to None.
        """
        self.session = requests.Session()
        self.api_endpoint = "https://www.xdash.ai/api/aiquery"
        self.timeout = timeout
        self.last_response = {}
        
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "User-Agent": LitAgent().random(),
        }
        
        self.session.headers.update(self.headers)
        self.proxies = proxies
        
        # Default values for search parameters
        self.search_uuid = "sniqIUPOkv8RIj6TWnB1j"
        self.visitor_uuid = "89072f2dde4f20c276ed5dd9242eaa4f"
        self.token = "U2FsdGVkX1994yT0p52bEy373unUukq49cSd9K7QMjQ="

    def split_and_format(self, input_text: str) -> Dict[str, Any]:
        """Split and format the input text based on specific tags.
        
        Args:
            input_text (str): The raw response text from XDash API
        
        Returns:
            Dict[str, Any]: Parsed response with answer, llm, and related sections
        """
        llm_tag = "__LLM_RESPONSE__"
        related_tag = "__RELATED_QUESTIONS__"
        
        llm_index = input_text.find(llm_tag)
        related_index = input_text.find(related_tag)
        
        answer_text = input_text[:llm_index].strip()
        llm_text = input_text[llm_index + len(llm_tag):related_index].strip()
        related_text = input_text[related_index + len(related_tag):].strip()
        
        # Remove citation patterns
        llm_text = re.sub(r'\s*\[citation:\d+\]\s*', '', llm_text)
        
        return {
            "answer": json.loads(answer_text) if answer_text else None,
            "llm": llm_text,
            "related": json.loads(related_text) if related_text else None
        }

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean HTML entities from text.
        
        Args:
            text (str): Text with HTML entities
        
        Returns:
            str: Cleaned text
        """
        entities = {
            "&amp;": "&",
            "&#x27;": "'",
            "&quot;": '"',
            "&lt;": "<",
            "&gt;": ">",
            "&nbsp;": " ",
            "&apos;": "'",
            "&#39;": "'"
        }
        
        def replace_entity(match):
            entity = match.group(0)
            return entities.get(entity, entity)
        
        return re.sub(r'&[#A-Za-z0-9]+;', replace_entity, text)

    def format_output(self, data: Dict[str, Any]) -> str:
        """Format the output data into a readable string.
        
        Args:
            data (Dict[str, Any]): Parsed response data
        
        Returns:
            str: Formatted response text
        """
        parts = []
        
        # Add LLM response if available
        if data.get("llm"):
            parts.append(data["llm"])
        
        # Add answer snippets if available
        # if data.get("answer"):
        #     answer_parts = []
        #     for a in data["answer"]:
        #         answer_parts.append(f"*{a['name']}*\n{self.clean_text(a['snippet'])}\n🔗 {a['url']}")
        #     parts.append("\n\n".join(answer_parts))
        
        # # Add related questions if available
        # if data.get("related"):
        #     related_parts = ["*Related:*"]
        #     for r in data["related"]:
        #         related_parts.append(f" • {r['question']}")
        #     parts.append("\n".join(related_parts))
        
        return "\n\n".join(filter(None, parts))

    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[Response, Dict[str, str]]:
        """Search using the XDash API and get AI-generated responses.
        
        Args:
            prompt (str): The search query or prompt to send to the API.
            stream (bool, optional): Parameter kept for compatibility, 
                                    XDash doesn't support streaming. Defaults to False.
            raw (bool, optional): If True, returns raw response dictionary.
                                 If False, returns Response objects. Defaults to False.
        
        Returns:
            Union[Response, Dict[str, str]]: 
                - If raw=False: Returns formatted response as Response object
                - If raw=True: Returns raw response dictionary
        
        Raises:
            APIConnectionError: If the API request fails
        """
        payload = {
            "query": prompt,
            "search_uuid": self.search_uuid,
            "visitor_uuid": self.visitor_uuid,
            "token": self.token
        }
        
        # Update referer with the current query
        self.headers["Referer"] = f"https://www.xdash.ai/search?q={quote(prompt)}&rid={self.search_uuid}"
        self.session.headers.update(self.headers)
        
        try:
            response = self.session.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout,
                proxies=self.proxies
            )
            
            response.raise_for_status()
            
            response_text = response.text
            parsed_data = self.split_and_format(response_text)
            formatted_response = self.format_output(parsed_data)
            
            self.last_response = formatted_response
            
            if raw:
                return {"result": formatted_response}
            else:
                return Response(formatted_response)
        
        except requests.exceptions.RequestException as e:
            raise exceptions.APIConnectionError(f"Request failed: {e}")


if __name__ == "__main__":
    from rich import print
    
    ai = XDash()
    response = ai.search(input(">>> "), raw=False)
    print(response)
