import re
import json
from curl_cffi import CurlError
from curl_cffi.requests import Session
from typing import Union, Any, Dict, Generator, Optional
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class WritingMate(Provider):
    AVAILABLE_MODELS = [
        "claude-3-haiku-20240307",
        "gemini-1.5-flash-latest",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "google/gemini-flash-1.5-8b-exp",
        "gpt-4o-mini"
    ]
    """
    Provider for WritingMate streaming API.
    """
    api_endpoint = "https://chat.writingmate.ai/api/chat/tools-stream"

    def __init__(
        self,
        cookies_path: str = "cookies.json",
        is_conversation: bool = True,
        max_tokens: int = 4096,
        timeout: int = 60,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {}, # Added proxies parameter
        history_offset: int = 10250, # Added history_offset parameter
        act: str = None,
        system_prompt: str = "You are a friendly, helpful AI assistant.",
        model: str = "gpt-4o-mini"
    ):
        self.cookies_path = cookies_path
        # Load cookies into a dictionary for curl_cffi
        self.cookies = self._load_cookies_dict(cookies_path) 
        # Initialize curl_cffi Session
        self.session = Session() 
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.model = model
        if self.model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {self.model}. Choose from {self.AVAILABLE_MODELS}")
        self.last_response = {}
        self.agent = LitAgent() # Initialize LitAgent
        self.headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8",
            # Content-Type might be application/json based on body, but API expects text/plain? Keep for now.
            "Content-Type": "text/plain;charset=UTF-8", 
            "Origin": "https://chat.writingmate.ai",
            "Referer": "https://chat.writingmate.ai/chat",
            # Remove Cookie header, pass cookies via parameter
            # "Cookie": self.cookies, 
            "DNT": "1",
            "sec-ch-ua": "\"Microsoft Edge\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-GPC": "1",
            "User-Agent": self.agent.random() # Use LitAgent
        }
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies 
        
        self.__available_optimizers = (
            m for m in dir(Optimizers)
            if callable(getattr(Optimizers, m)) and not m.startswith("__")
        )
        Conversation.intro = (
            AwesomePrompts().get_act(act, raise_not_found=True, default=None, case_insensitive=True)
            if act else intro or Conversation.intro
        )
        self.conversation = Conversation(is_conversation, max_tokens, filepath, update_file)
        # Apply history offset
        self.conversation.history_offset = history_offset 

    # Keep original _load_cookies if needed elsewhere, or remove
    # def _load_cookies(self, path: str) -> str:
    #     try:
    #         with open(path, 'r') as f:
    #             data = json.load(f)
    #         return '; '.join(f"{c['name']}={c['value']}" for c in data)
    #     except (FileNotFoundError, json.JSONDecodeError):
    #         raise RuntimeError(f"Failed to load cookies from {path}")

    # New method to load cookies as a dictionary
    def _load_cookies_dict(self, path: str) -> Dict[str, str]:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # Ensure data is a list of cookie objects
            if not isinstance(data, list):
                 raise ValueError("Cookie file should contain a list of cookie objects.")
            return {c['name']: c['value'] for c in data if 'name' in c and 'value' in c}
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to load cookies from {path}: {e}")

    @staticmethod
    def _writingmate_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from the WritingMate stream format '0:"..."'."""
        if isinstance(chunk, str):
            # Regex to find the pattern 0:"<content>"
            match = re.search(r'0:"(.*?)"(?=,|$)', chunk) # Look for 0:"...", possibly followed by comma or end of string
            if match:
                # Decode potential unicode escapes like \u00e9 and handle escaped quotes/backslashes
                content = match.group(1).encode().decode('unicode_escape')
                return content.replace('\\\\', '\\').replace('\\"', '"')
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = True, # Defaulting stream to True as per original
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False
    ) -> Union[Dict[str,Any], Generator[Any,None,None]]:
        # ... existing prompt generation and optimizer logic ...
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                # Use the correct exception type
                raise exceptions.FailedToGenerateResponseError(f"Unknown optimizer: {optimizer}")

        # Body seems to be JSON, let curl_cffi handle serialization
        body = {
            "chatSettings": {
                "model": self.model,
                "prompt": self.system_prompt,
                "temperature": 0.5,
                "contextLength": 4096,
                "includeProfileContext": True,
                "includeWorkspaceInstructions": True,
                "embeddingsProvider": "openai"
            },
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt}
            ],
            "selectedTools": []
        }

        def for_stream():
            try:
                # Use curl_cffi session post, pass cookies dict
                response = self.session.post(
                    self.api_endpoint, 
                    headers=self.headers, 
                    cookies=self.cookies, # Pass cookies dict
                    json=body, # Pass body as json
                    stream=True, 
                    timeout=self.timeout,
                    impersonate="chrome120" # Add impersonate
                    # http_version=CurlHttpVersion.V1_1 # Add if HTTP/2 errors occur
                )
                if not response.ok:
                    raise exceptions.FailedToGenerateResponseError(
                        f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                    )
                streaming_text = ""
                # Use sanitize_stream with the custom extractor
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value=None, # No simple prefix
                    to_json=False,    # Content is not JSON
                    content_extractor=self._writingmate_extractor, # Use the specific extractor
                    raw=raw
                )

                for content_chunk in processed_stream:
                    # Always yield as string, even in raw mode
                    if isinstance(content_chunk, bytes):
                        content_chunk = content_chunk.decode('utf-8', errors='ignore')
                    if raw:
                        yield content_chunk
                    else:
                        if content_chunk and isinstance(content_chunk, str):
                            streaming_text += content_chunk
                            yield dict(text=content_chunk)

                self.last_response.update(dict(text=streaming_text))
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e: # Catch other potential exceptions
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")

        def for_non_stream():
            for _ in for_stream():
                pass
            return self.last_response

        effective_stream = stream if stream is not None else True 
        return for_stream() if effective_stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False, # Default stream to False as per original chat method
        optimizer: str = None,
        conversationally: bool = False,
        raw: bool = False,  # Added raw parameter
    ) -> Union[str, Generator[str,None,None]]:
        if stream:
            def text_stream():
                for response in self.ask(
                    prompt, stream=True, raw=raw,
                    optimizer=optimizer, conversationally=conversationally
                ):
                    if raw:
                        yield response
                    else:
                        yield self.get_message(response)
            return text_stream()
        else:
            response_data = self.ask(
                prompt,
                stream=False,
                raw=raw,
                optimizer=optimizer,
                conversationally=conversationally,
            )
            if raw:
                return response_data
            if isinstance(response_data, dict):
                return self.get_message(response_data)
            else:
                full_text = "".join(self.get_message(chunk) for chunk in response_data if isinstance(chunk, dict))
                return full_text


    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        # Ensure text exists before processing
        # Formatting is now mostly handled by the extractor
        text = response.get("text", "") 
        formatted_text = text # Keep newline replacement if needed: .replace('\\n', '\n')
        return formatted_text
    
if __name__ == "__main__":
    from rich import print
    try:
        ai = WritingMate(cookies_path="cookies.json", proxies={}, timeout=120) # Example with proxies and timeout
        # Get input within the try block
        user_input = input(">>> ") 
        response = ai.chat(user_input, stream=True)
        print("[bold green]Assistant:[/bold green]")
        for chunk in response:
            print(chunk, end="", flush=True)
        print() # Add a newline at the end
    except RuntimeError as e:
        print(f"[bold red]Error initializing WritingMate:[/bold red] {e}")
    except exceptions.FailedToGenerateResponseError as e:
        print(f"[bold red]Error during chat:[/bold red] {e}")
    except Exception as e:
        print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
