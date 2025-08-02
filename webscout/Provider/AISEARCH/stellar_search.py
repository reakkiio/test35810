import requests
import re
from typing import Dict, Optional, Generator, Union, Any
from webscout.AIbase import AISearch, SearchResponse
from webscout import exceptions
from webscout.litagent import LitAgent
from webscout.AIutel import sanitize_stream

class Stellar(AISearch):
    """AI Search provider for stellar.chatastra.ai"""
    def __init__(self, timeout: int = 30, proxies: Optional[dict] = None):
        self.api_endpoint = "https://stellar.chatastra.ai/search/x1GUVzl"
        self.timeout = timeout
        self.proxies = proxies
        self.session = requests.Session()
        self.headers = {
            "accept": "text/x-component",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "multipart/form-data; boundary=----WebKitFormBoundaryQsWD5Qs3QqDkNBPH",
            "dnt": "1",
            "next-action": "efc2643ed9bafe182a010b58ebea17f068ad3985",
            "next-router-state-tree": "%5B%22%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2C%22%2F%22%2C%22refresh%22%5D%7D%2Cnull%2Cnull%2Ctrue%5D",
            "origin": "https://stellar.chatastra.ai",
            "priority": "u=1, i",
            "referer": "https://stellar.chatastra.ai/search/x1GUVzl",
            "sec-ch-ua": '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": LitAgent().random(),
            "cookie": "__client_uat=0; __client_uat_K90aduOv=0",
        }
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies = proxies

    def _make_payload(self, prompt: str) -> bytes:        # This is a static payload for the demo; in production, generate dynamically as needed
        boundary = "----WebKitFormBoundaryQsWD5Qs3QqDkNBPH"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"1\"\r\n\r\n{{\"id\":\"71bb616ba5b7cbcac2308fe0c249a9f2d51825b7\",\"bound\":null}}\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"2\"\r\n\r\n{{\"id\":\"8bcca1d0cb933b14fefde88dacb2865be3d1d525\",\"bound\":null}}\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"3_input\"\r\n\r\n{prompt}\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"3_id\"\r\n\r\nx1GUVzl\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"3_userId\"\r\n\r\nnull\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"0\"\r\n\r\n[{{\"action\":\"$F1\",\"options\":{{\"onSetAIState\":\"$F2\"}}}},{{\"messages\":[],\"chatId\":\"\"}},\"$K3\"]\r\n",
            f"--{boundary}--\r\n"
        ]
        return "".join(parts).encode("utf-8")

    @staticmethod
    def _stellar_extractor(chunk: Union[str, bytes, Dict[str, Any]]) -> Optional[str]:
        """
        Extracts content from the Stellar stream format with focused pattern matching.
        
        Prioritizes the primary diff pattern to avoid duplication and focuses on
        incremental content building from stellar.chatastra.ai streaming response.
        """
        if isinstance(chunk, bytes):
            try:
                chunk = chunk.decode('utf-8', errors='replace')
            except Exception:
                return None
        if not isinstance(chunk, str):
            return None
        
        # Primary pattern: Hex key diff format (most reliable for streaming)
        # Matches: 16:{"diff":[0,"AI"],"next":"$@18"}
        primary_pattern = r'[0-9a-f]+:\{"diff":\[0,"([^"]*?)"\]'
        primary_matches = re.findall(primary_pattern, chunk)
        
        if primary_matches:
            # Join the matches and clean up
            extracted_text = ''.join(primary_matches)
            
            # Handle escape sequences properly
            extracted_text = extracted_text.replace('\\n', '\n')
            extracted_text = extracted_text.replace('\\r', '\r')
            extracted_text = extracted_text.replace('\\"', '"')
            extracted_text = extracted_text.replace('\\t', '\t')
            extracted_text = extracted_text.replace('\\/', '/')
            extracted_text = extracted_text.replace('\\\\', '\\')
            
            # Clean up markdown formatting
            extracted_text = extracted_text.replace('\\*', '*')
            extracted_text = extracted_text.replace('\\#', '#')
            extracted_text = extracted_text.replace('\\[', '[')
            extracted_text = extracted_text.replace('\\]', ']')
            extracted_text = extracted_text.replace('\\(', '(')
            extracted_text = extracted_text.replace('\\)', ')')
            
            return extracted_text if extracted_text.strip() else None
        
        # # Fallback: Look for Ta24 content blocks (complete responses)
        # if ':Ta24,' in chunk:
        #     ta24_pattern = r':Ta24,([^}]*?)(?:\d+:|$)'
        #     ta24_matches = re.findall(ta24_pattern, chunk)
        #     if ta24_matches:
        #         extracted_text = ''.join(ta24_matches)
        #         # Basic cleanup
        #         extracted_text = extracted_text.replace('\\n', '\n')
        #         extracted_text = extracted_text.replace('\\"', '"')
        #         return extracted_text.strip() if extracted_text.strip() else None
        
        # # Secondary fallback: Direct diff patterns without hex prefix
        # fallback_pattern = r'\{"diff":\[0,"([^"]*?)"\]'
        # fallback_matches = re.findall(fallback_pattern, chunk)
        # if fallback_matches:
        #     extracted_text = ''.join(fallback_matches)
        #     extracted_text = extracted_text.replace('\\n', '\n')
        #     extracted_text = extracted_text.replace('\\"', '"')
        #     return extracted_text if extracted_text.strip() else None
        
        return None

    def search(self, prompt: str, stream: bool = False, raw: bool = False) -> Union[SearchResponse, Generator[Union[Dict[str, str], SearchResponse, str], None, None]]:
        payload = self._make_payload(prompt)
        try:
            response = self.session.post(
                self.api_endpoint,
                data=payload,
                timeout=self.timeout,
                proxies=self.proxies,
                stream=stream,
            )
            if not response.ok:
                raise exceptions.APIConnectionError(f"Failed to get response: {response.status_code} {response.text}")

            def _yield_stream():
                # Use sanitize_stream for real-time extraction from the response iterator
                processed_stream = sanitize_stream(
                    data=response.iter_lines(decode_unicode=True),
                    intro_value=None,
                    to_json=False,
                    content_extractor=self._stellar_extractor
                )
                full_response = ""
                for content in processed_stream:
                    if content and isinstance(content, str):
                        full_response += content
                        if raw:
                            yield {"text": content}
                        else:
                            yield content
                # Do NOT yield SearchResponse(full_response) in streaming mode to avoid duplicate output

            if stream:
                return _yield_stream()
            else:
                # Use sanitize_stream for the full response text
                processed_stream = sanitize_stream(
                    data=response.text.splitlines(),
                    intro_value=None,
                    to_json=False,
                    content_extractor=self._stellar_extractor
                )
                full_response = ""
                for content in processed_stream:
                    if content and isinstance(content, str):
                        full_response += content
                if raw:
                    return {"text": full_response}
                else:
                    return SearchResponse(full_response)
        except requests.RequestException as e:
            raise exceptions.APIConnectionError(f"Request failed: {e}")

if __name__ == "__main__":
    from rich import print
    ai = Stellar()
    user_query = input(">>> ")
    response = ai.search(user_query, stream=True, raw=False)
    for chunk in response:
        print(chunk, end="", flush=True)
