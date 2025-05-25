import requests
import hashlib
import json
import random
import time
import re
from typing import Dict, Optional, Generator, Union, Any

from webscout.AIbase import AISearch, SearchResponse
from webscout import exceptions
from webscout.litagent import LitAgent


class Hika(AISearch):
    """A class to interact with the Hika AI search API."""

    def __init__(
        self,
        timeout: int = 60,
        proxies: Optional[dict] = None,
        language: str = "en",
        # model: str = "deepseek-r1",
        
    ):
        self.session = requests.Session()
        self.base_url = "https://api.hika.fyi/api/"
        self.endpoint = "kbase/web"
        self.timeout = timeout
        self.language = language
        # self.model = model
        self.last_response = {}
        
        self.headers = {
            "Content-Type": "application/json",
            "Origin": "https://hika.fyi",
            "Referer": "https://hika.fyi/",
            "User-Agent": LitAgent().random()
        }
        
        self.session.headers.update(self.headers)
        self.proxies = proxies

    def generate_id(self):
        """Generate a unique ID and hash for the request."""
        uid = ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') for _ in range(10))
        uid += hex(int(time.time()))[2:]
        hash_id = hashlib.sha256(f"#{uid}*".encode()).hexdigest()
        return {"uid": uid, "hashId": hash_id}

    def clean_text(self, text):
        """Clean all XML tags and control markers from text.
        
        Args:
            text (str): The text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove XML tags and special markers
        # First remove <r> tag at the beginning
        text = text.lstrip("<r>")
        
        # Remove any remaining XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove [DONE] marker at the end
        text = re.sub(r'\[DONE\]\s*$', '', text)
        
        return text

    def search(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse], None, None]]:
        """Search using the Hika API and get AI-generated responses."""
        if not prompt or len(prompt) < 2:
            raise exceptions.APIConnectionError("Search query must be at least 2 characters long")
        
        # Generate ID for this request
        id_data = self.generate_id()
        uid, hash_id = id_data["uid"], id_data["hashId"]
        
        # Update headers with request-specific values
        request_headers = {
            **self.headers,
            "x-hika": hash_id,
            "x-uid": uid
        }
        
        # Prepare payload (fix: stream as string, add search_language)
        payload = {
            "keyword": prompt,
            "language": self.language,
            "search_language": self.language,
            "stream": "true"  # Must be string, not boolean
        }

        def for_stream():
            try:
                with self.session.post(
                    f"{self.base_url}{self.endpoint}",
                    json=payload,
                    headers=request_headers,
                    stream=True,
                    timeout=self.timeout,
                    proxies=self.proxies
                ) as response:
                    if not response.ok:
                        raise exceptions.APIConnectionError(
                            f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                        )
                    
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                # Handle chunk and references
                                if "chunk" in data:
                                    chunk = data["chunk"]
                                    if "[DONE]" in chunk:
                                        continue
                                    clean_chunk = self.clean_text(chunk)
                                    if clean_chunk:
                                        if raw:
                                            yield {"text": clean_chunk}
                                        else:
                                            yield SearchResponse(clean_chunk)
                                elif "references" in data:
                                    # Optionally yield references if raw requested
                                    if raw:
                                        yield {"references": data["references"]}
                            except json.JSONDecodeError:
                                pass
                                
            except requests.exceptions.RequestException as e:
                raise exceptions.APIConnectionError(f"Request failed: {e}")

        def for_non_stream():
            full_response = ""
            for chunk in for_stream():
                if raw:
                    yield chunk
                else:
                    full_response += str(chunk)
            
            if not raw:
                # Clean up the response text one final time
                cleaned_response = self.format_response(full_response)
                self.last_response = SearchResponse(cleaned_response)
                return self.last_response

        return for_stream() if stream else for_non_stream()
    
    def format_response(self, text: str) -> str:
        """Format the response text for better readability."""
        if not text:
            return ""
            
        # First clean any tags or markers
        cleaned_text = self.clean_text(text)
        
        # Remove any empty lines
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        
        # Remove any trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text


if __name__ == "__main__":
    from rich import print
    ai = Hika()
    try:
        response = ai.search(input(">>> "), stream=True, raw=False)
        for chunk in response:
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")