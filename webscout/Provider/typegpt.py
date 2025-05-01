from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
from typing import Union, Any, Dict, Generator

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class TypeGPT(Provider):
    """
    A class to interact with the TypeGPT.net API. Improved to match webscout standards.
    """
    AVAILABLE_MODELS = [
        # Working Models (based on testing)
        "gpt-4o-mini-2024-07-18",
        "chatgpt-4o-latest",
        "deepseek-r1",
        "deepseek-v3",
        "uncensored-r1",
        "Image-Generator",
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 4000,  # Set a reasonable default
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gpt-4o-mini-2024-07-18",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.5,
        presence_penalty: int = 0,
        frequency_penalty: int = 0,
        top_p: float = 1,
    ):
        """Initializes the TypeGPT API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(self.AVAILABLE_MODELS)}")

        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://chat.typegpt.net/api/openai/v1/chat/completions"
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.top_p = top_p
        self.headers = {
            "authority": "chat.typegpt.net",
            "accept": "application/json, text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://chat.typegpt.net",
            "referer": "https://chat.typegpt.net/",
            "user-agent": LitAgent().random()
        }

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
        self.conversation = Conversation(is_conversation, self.max_tokens_to_sample, filepath, update_file)
        self.conversation.history_offset = history_offset
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator[Any, None, None]]:
        """Sends a prompt to the TypeGPT.net API and returns the response."""
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise exceptions.FailedToGenerateResponseError(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt}
            ],
            "stream": stream,
            "model": self.model,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens_to_sample,
        }

        def for_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint, 
                    headers=self.headers, 
                    json=payload, 
                    stream=True, 
                    timeout=self.timeout,
                    impersonate="chrome120"
                )
            except CurlError as ce:
                raise exceptions.FailedToGenerateResponseError(
                    f"Network connection failed (CurlError). Check your firewall or antivirus settings. Original error: {ce}"
                ) from ce

            response.raise_for_status() # Check for HTTP errors first

            streaming_text = ""
            # Use sanitize_stream
            processed_stream = sanitize_stream(
                data=response.iter_content(chunk_size=None), # Pass byte iterator
                intro_value="data:",
                to_json=True,     # Stream sends JSON
                skip_markers=["[DONE]"],
                content_extractor=lambda chunk: chunk.get('choices', [{}])[0].get('delta', {}).get('content') if isinstance(chunk, dict) else None,
                yield_raw_on_error=False # Skip non-JSON or lines where extractor fails
            )

            for content_chunk in processed_stream:
                # content_chunk is the string extracted by the content_extractor
                if content_chunk and isinstance(content_chunk, str):
                    streaming_text += content_chunk
                    yield dict(text=content_chunk) if not raw else content_chunk
                    # Update last_response incrementally
                    self.last_response = dict(text=streaming_text)

            # Update conversation history after stream finishes
            if streaming_text: # Only update if something was received
                 self.conversation.update_chat_history(prompt, streaming_text)


        def for_non_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=self.timeout,
                    impersonate="chrome120"
                )
            except CurlError as ce:
                raise exceptions.FailedToGenerateResponseError(
                    f"Network connection failed (CurlError). Check your firewall or antivirus settings. Original error: {ce}"
                ) from ce

            response.raise_for_status() # Check for HTTP errors

            try:
                response_text = response.text # Get raw text

                # Use sanitize_stream for non-streaming JSON response
                processed_stream = sanitize_stream(
                    data=response_text,
                    to_json=True, # Parse the whole text as JSON
                    intro_value=None,
                    # Extractor for non-stream structure
                    content_extractor=lambda chunk: chunk.get('choices', [{}])[0].get('message', {}).get('content') if isinstance(chunk, dict) else None,
                    yield_raw_on_error=False
                )

                # Extract the single result
                content = ""
                for extracted_content in processed_stream:
                    content = extracted_content if isinstance(extracted_content, str) else ""

                self.last_response = {"text": content} # Store in expected format
                self.conversation.update_chat_history(prompt, content)
                return self.last_response
            except (json.JSONDecodeError, Exception) as je: # Catch potential JSON errors or others
                 raise exceptions.FailedToGenerateResponseError(f"Failed to decode JSON response: {je} - Response text: {response.text}")


        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response string or stream."""
        if stream:
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt, stream=True, raw=False, # Ensure ask yields dicts
                optimizer=optimizer, conversationally=conversationally
            )
            for chunk_dict in gen:
                 # get_message expects a dict
                yield self.get_message(chunk_dict) 
        else:
            # ask() returns a dict when not streaming
            response_dict = self.ask(
                prompt, stream=False, 
                optimizer=optimizer, conversationally=conversationally
            )
            return self.get_message(response_dict)

    def get_message(self, response: Dict[str, Any]) -> str:
        """Retrieves message from response."""
        if isinstance(response, dict):
            assert isinstance(response, dict), "Response should be of dict data-type only"
            # Handle potential unicode escapes in the final text
            text = response.get("text", "")
            try:
                # Attempt to decode escapes, return original if fails
                return text.encode('utf-8').decode('unicode_escape')
            except UnicodeDecodeError:
                return text 
        else:
            # This case should ideally not be reached if ask() behaves as expected
            raise TypeError(f"Invalid response type: {type(response)}. Expected dict.")

if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)
    
    # Test all available models
    working = 0
    total = len(TypeGPT.AVAILABLE_MODELS)
    
    for model in TypeGPT.AVAILABLE_MODELS:
        try:
            test_ai = TypeGPT(model=model, timeout=60)
            # Test stream first
            response_stream = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            print(f"\r{model:<50} {'Streaming...':<10}", end="", flush=True)
            for chunk in response_stream:
                response_text += chunk
                # Optional: print chunks as they arrive for visual feedback
                # print(chunk, end="", flush=True) 
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip() # Already decoded in get_message
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗ (Stream)"
                display_text = "Empty or invalid stream response"
            print(f"\r{model:<50} {status:<10} {display_text}")
            
            # Optional: Add non-stream test if needed, but stream test covers basic functionality
            # print(f"\r{model:<50} {'Non-Stream...':<10}", end="", flush=True)
            # response_non_stream = test_ai.chat("Say 'Hi' again", stream=False)
            # if not response_non_stream or len(response_non_stream.strip()) == 0:
            #      print(f"\r{model:<50} {'✗ (Non-Stream)':<10} Empty non-stream response")


        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
