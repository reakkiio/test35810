from curl_cffi.requests import Session
from curl_cffi import CurlError
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class TogetherAI(Provider):
    """
    A class to interact with the TogetherAI API.
    """

    AVAILABLE_MODELS = [
        "moonshotai/Kimi-K2-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "togethercomputer/MoA-1",
        "deepseek-ai/DeepSeek-V3",
        "Rrrr/MiniMaxAI/MiniMax-M1-40k-eb978d0c",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "meta-llama/Llama-3-8b-chat-hf",
        "togethercomputer/MoA-1-Turbo",
        "eddiehou/meta-llama/Llama-3.1-405B",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "google/gemma-3n-E4B-it",
        "arcee-ai/AFM-4.5B-Preview",
        "moonshotai/Kimi-K2-Instruct-tgl-testing",
        "lgai/exaone-3-5-32b-instruct",
        "lgai/exaone-deep-32b",
        "meta-llama/Llama-3-70b-chat-hf",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "google/gemma-2-27b-it",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen3-235B-A22B-fp8-tput",
        "deepseek-ai/DeepSeek-R1",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "Qwen/Qwen2-VL-72B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "meta-llama/Llama-Vision-Free",
        "perplexity-ai/r1-1776",
        "scb10x/scb10x-llama3-1-typhoon2-70b-instruct",
        "Qwen/QwQ-32B",
        "arcee-ai/maestro-reasoning",
        "togethercomputer/Refuel-Llm-V2-Small",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "arcee-ai/coder-large",
        "arcee-ai/virtuoso-large",
        "arcee_ai/arcee-spotlight",
        "arcee-ai/arcee-blitz",
        "deepseek-ai/DeepSeek-R1-0528-tput",
        "arcee-ai/virtuoso-medium-v2",
        "arcee-ai/caller",
        "marin-community/marin-8b-instruct",
        "google/gemma-3-27b-it",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "moonshotai/Kimi-K2-Instruct-B200",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "scb10x/scb10x-typhoon-2-1-gemma3-12b",
        "togethercomputer/Refuel-Llm-V2",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "yan/deepseek-ai-deepseek-v3",
        "Rrrr/MiniMaxAI/MiniMax-M1-40k-eb978d0c",
        "Rrrr/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-e35bdf03",
        "Rrrr/ChatGPT-5",
        "Rrrr/MeowGPT-3.5",
        "Rrrr/Llama-3.3-70B-32k-Instruct-Reference-2262472f-08cfe871",
        "Rrrr/nim/meta/llama-3.1-8b-instruct-7ac4b754",
        "blackbox/meta-llama-3-1-8b",
        "Rrrr/meta-llama/Llama-3-70b-chat-hf-6f9ad551",
        "Rrrr/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-03dc18e1",
        "Rrrr/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-6c92f39d",
        "roberizk@gmail.com/meta-llama/Meta-Llama-3-70B-Instruct-6feb41f7",
        "roberizk@gmail.com/meta-llama/Llama-3-70b-chat-hf-26ee936b",
        "roberizk@gmail.com/meta-llama/Meta-Llama-3-8B-Instruct-8ced8839"
    ]

    @staticmethod
    def _togetherai_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from TogetherAI stream JSON objects."""
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
        model: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        system_prompt: str = "You are a helpful assistant.",
        browser: str = "chrome"
    ):
        """Initializes the TogetherAI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.api_endpoint = "https://api.together.xyz/v1/chat/completions"
        self.activation_endpoint = "https://www.codegeneration.ai/activate-v2"

        # Initialize LitAgent
        self.agent = LitAgent()
        self.fingerprint = self.agent.generate_fingerprint(browser)

        # Use the fingerprint for headers
        self.headers = {
            "Accept": self.fingerprint["accept"],
            "Accept-Language": self.fingerprint["accept_language"],
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Origin": "https://www.codegeneration.ai",
            "Pragma": "no-cache",
            "Referer": "https://www.codegeneration.ai/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": self.fingerprint["user_agent"],
        }

        # Initialize curl_cffi Session
        self.session = Session()
        self.session.headers.update(self.headers)
        self.session.proxies = proxies
        self.system_prompt = system_prompt
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self._api_key_cache = None

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

        # Update headers with new fingerprint
        self.headers.update({
            "Accept": self.fingerprint["accept"],
            "Accept-Language": self.fingerprint["accept_language"],
            "User-Agent": self.fingerprint["user_agent"],
        })

        # Update session headers
        self.session.headers.update(self.headers)

        return self.fingerprint

    def get_activation_key(self) -> str:
        """Get API key from activation endpoint"""
        if self._api_key_cache:
            return self._api_key_cache

        try:
            response = self.session.get(
                self.activation_endpoint,
                headers={"Accept": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            activation_data = response.json()
            self._api_key_cache = activation_data["openAIParams"]["apiKey"]
            return self._api_key_cache
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Failed to get activation key: {e}")

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """
        Sends a prompt to the TogetherAI API and returns the response.
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")
        if not self.headers.get("Authorization"):
            api_key = self.get_activation_key()
            self.headers["Authorization"] = f"Bearer {api_key}"
            self.session.headers.update(self.headers)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt},
            ],
            "stream": stream
        }
        def for_stream():
            streaming_text = ""
            try:
                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome110"
                )
                response.raise_for_status()
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value="data:",
                    to_json=True,
                    skip_markers=["[DONE]"],
                    content_extractor=self._togetherai_extractor,
                    yield_raw_on_error=False,
                    raw=raw
                )
                for content_chunk in processed_stream:
                    if isinstance(content_chunk, bytes):
                        content_chunk = content_chunk.decode('utf-8', errors='ignore')
                    if content_chunk is None:
                        continue
                    if raw:
                        yield content_chunk
                    else:
                        if content_chunk and isinstance(content_chunk, str):
                            streaming_text += content_chunk
                            resp = dict(text=content_chunk)
                            yield resp
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {str(e)}") from e
            finally:
                if streaming_text:
                    self.last_response = {"text": streaming_text}
                    self.conversation.update_chat_history(prompt, streaming_text)
        def for_non_stream():
            try:
                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=self.timeout,
                    impersonate="chrome110"
                )
                response.raise_for_status()
                response_text = response.text
                processed_stream = sanitize_stream(
                    data=response_text,
                    to_json=True,
                    intro_value=None,
                    content_extractor=lambda chunk: chunk.get("choices", [{}])[0].get("message", {}).get("content") if isinstance(chunk, dict) else None,
                    yield_raw_on_error=False,
                    raw=raw
                )
                content = next((c for c in processed_stream if c is not None), None)
                content = content if isinstance(content, str) else ""
                self.last_response = {"text": content}
                self.conversation.update_chat_history(prompt, content)
                return self.last_response if not raw else content
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e:
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {e} - {err_text}") from e
        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        raw: bool = False,  # Added raw parameter
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream_chat():
            gen = self.ask(
                prompt, stream=True, raw=raw,
                optimizer=optimizer, conversationally=conversationally
            )
            for response in gen:
                if raw:
                    yield response
                else:
                    yield self.get_message(response)
        def for_non_stream_chat():
            response_data = self.ask(
                prompt, stream=False, raw=raw,
                optimizer=optimizer, conversationally=conversationally
            )
            if raw:
                return response_data
            else:
                return self.get_message(response_data)
        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        """Retrieves message only from response"""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]


if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in TogetherAI.AVAILABLE_MODELS:
        try:
            test_ai = TogetherAI(model=model, timeout=60)
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