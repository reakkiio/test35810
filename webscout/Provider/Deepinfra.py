from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import os
from typing import Any, Dict, Optional, Generator, Union, List

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent

class DeepInfra(Provider):
    """
    A class to interact with the DeepInfra API with LitAgent user-agent.
    """

    AVAILABLE_MODELS = [
        # "anthropic/claude-3-7-sonnet-latest",  # >>>> NOT WORKING

        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Turbo",
        "deepseek-ai/DeepSeek-V3",

        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-3-27b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-4b-it",
        # "google/gemini-1.5-flash",  # >>>> NOT WORKING
        # "google/gemini-1.5-flash-8b",  # >>>> NOT WORKING
        # "google/gemini-2.0-flash-001",  # >>>> NOT WORKING

        # "Gryphe/MythoMax-L2-13b",  # >>>> NOT WORKING

        # "meta-llama/Llama-3.2-1B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Llama-3.2-3B-Instruct",  # >>>> NOT WORKING
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # "meta-llama/Llama-3.2-90B-Vision-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Llama-3.2-11B-Vision-Instruct",  # >>>> NOT WORKING
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        # "meta-llama/Meta-Llama-3-70B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3-8B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3.1-70B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",  # >>>> NOT WORKING
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        # "meta-llama/Meta-Llama-3.1-405B-Instruct",  # >>>> NOT WORKING

        "microsoft/phi-4",
        "microsoft/Phi-4-multimodal-instruct",
        "microsoft/WizardLM-2-8x22B",
        # "mistralai/Mixtral-8x7B-Instruct-v0.1",  # >>>> NOT WORKING
        # "mistralai/Mistral-7B-Instruct-v0.3",  # >>>> NOT WORKING
        # "mistralai/Mistral-Nemo-Instruct-2407",  # >>>> NOT WORKING
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        # "NousResearch/Hermes-3-Llama-3.1-405B",  # >>>> NOT WORKING
        # "NovaSky-AI/Sky-T1-32B-Preview",  # >>>> NOT WORKING
        "Qwen/QwQ-32B",
        # "Qwen/Qwen2.5-7B-Instruct",  # >>>> NOT WORKING
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        # "Sao10K/L3.1-70B-Euryale-v2.2",  # >>>> NOT WORKING
        # "Sao10K/L3.3-70B-Euryale-v2.3",  # >>>> NOT WORKING
    ]

    @staticmethod
    def _deepinfra_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from DeepInfra stream JSON objects."""
        if isinstance(chunk, dict):
            return chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        return None

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2049,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        system_prompt: str = "You are a helpful assistant.",
        browser: str = "chrome" # Note: browser fingerprinting might be less effective with impersonate
    ):
        """Initializes the DeepInfra API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.url = "https://api.deepinfra.com/v1/openai/chat/completions"

        # Initialize LitAgent (keep if needed for other headers or logic)
        self.agent = LitAgent()
        # Fingerprint generation might be less relevant with impersonate
        self.fingerprint = self.agent.generate_fingerprint(browser)

        # Use the fingerprint for headers (keep relevant ones)
        self.headers = {
            "Accept": self.fingerprint["accept"], # Keep Accept
            "Accept-Language": self.fingerprint["accept_language"], # Keep Accept-Language
            "Content-Type": "application/json",
            "Cache-Control": "no-cache", # Keep Cache-Control
            "Origin": "https://deepinfra.com", # Keep Origin
            "Pragma": "no-cache", # Keep Pragma
            "Referer": "https://deepinfra.com/", # Keep Referer
            "Sec-Fetch-Dest": "empty", # Keep Sec-Fetch-*
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "X-Deepinfra-Source": "web-embed", # Keep custom headers
        }

        # Initialize curl_cffi Session
        self.session = Session()
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly
        self.system_prompt = system_prompt
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model

        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            )
            if act
            else intro or Conversation.intro
        )

        self.conversation = Conversation(
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset

    def refresh_identity(self, browser: str = None):
        """
        Refreshes the browser identity fingerprint.

        Args:
            browser: Specific browser to use for the new fingerprint
        """
        browser = browser or self.fingerprint.get("browser_type", "chrome")
        self.fingerprint = self.agent.generate_fingerprint(browser)

        # Update headers with new fingerprint (only relevant ones)
        self.headers.update({
            "Accept": self.fingerprint["accept"],
            "Accept-Language": self.fingerprint["accept_language"],
        })

        # Update session headers
        self.session.headers.update(self.headers) # Update only relevant headers

        return self.fingerprint

    def ask(
        self,
        prompt: str,
        stream: bool = False, # API supports streaming
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        # Payload construction
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt},
            ],
            "stream": stream # Pass stream argument to payload
        }

        def for_stream():
            streaming_text = "" # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.url,
                    # headers are set on the session
                    data=json.dumps(payload),
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors

                # Use sanitize_stream
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value="data:",
                    to_json=True,     # Stream sends JSON
                    skip_markers=["[DONE]"],
                    content_extractor=self._deepinfra_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                for content_chunk in processed_stream:
                    # content_chunk is the string extracted by _deepinfra_extractor
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        resp = dict(text=content_chunk)
                        yield resp if not raw else content_chunk

            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {str(e)}") from e
            finally:
                # Update history after stream finishes or fails
                if streaming_text:
                    self.last_response = {"text": streaming_text}
                    self.conversation.update_chat_history(prompt, streaming_text)


        def for_non_stream():
            try:
                # Use curl_cffi session post with impersonate for non-streaming
                response = self.session.post(
                    self.url,
                    # headers are set on the session
                    data=json.dumps(payload),
                    timeout=self.timeout,
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors

                response_text = response.text # Get raw text

                # Use sanitize_stream to parse the non-streaming JSON response
                processed_stream = sanitize_stream(
                    data=response_text,
                    to_json=True, # Parse the whole text as JSON
                    intro_value=None,
                    # Extractor for non-stream structure
                    content_extractor=lambda chunk: chunk.get("choices", [{}])[0].get("message", {}).get("content") if isinstance(chunk, dict) else None,
                    yield_raw_on_error=False
                )
                # Extract the single result
                content = next(processed_stream, None)
                content = content if isinstance(content, str) else "" # Ensure it's a string

                self.last_response = {"text": content}
                self.conversation.update_chat_history(prompt, content)
                return self.last_response if not raw else content # Return dict or raw string

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError, JSONDecodeError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {e} - {err_text}") from e


        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream_chat():
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt, stream=True, raw=False, # Ensure ask yields dicts
                optimizer=optimizer, conversationally=conversationally
            )
            for response_dict in gen:
                yield self.get_message(response_dict) # get_message expects dict

        def for_non_stream_chat():
            # ask() returns dict or str when not streaming
            response_data = self.ask(
                prompt, stream=False, raw=False, # Ensure ask returns dict
                optimizer=optimizer, conversationally=conversationally
            )
            return self.get_message(response_data) # get_message expects dict

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in DeepInfra.AVAILABLE_MODELS:
        try:
            test_ai = DeepInfra(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            for chunk in response:
                response_text += chunk

            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip().encode('utf-8', errors='ignore').decode('utf-8')
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
