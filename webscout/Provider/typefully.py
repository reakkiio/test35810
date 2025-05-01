from typing import Optional, Union, Any, Dict
import re
from uuid import uuid4

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent
# Replace requests with curl_cffi
from curl_cffi.requests import Session # Import Session
from curl_cffi import CurlError # Import CurlError

class TypefullyAI(Provider):
    """
    A class to interact with the Typefully AI API.

    Attributes:
        system_prompt (str): The system prompt to define the assistant's role.
        model (str): The model identifier to use for completions.
        output_length (int): Maximum length of the generated output.

    Examples:
        >>> from webscout.Provider.typefully import TypefullyAI
        >>> ai = TypefullyAI()
        >>> response = ai.chat("What's the weather today?")
        >>> print(response)
        'The weather today is sunny with a high of 75°F.'
    """
    AVAILABLE_MODELS = ["openai:gpt-4o-mini", "openai:gpt-4o", "anthropic:claude-3-5-haiku-20241022", "groq:llama-3.3-70b-versatile"]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 600,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        system_prompt: str = "You're a helpful assistant.",
        model: str = "openai:gpt-4o-mini",
    ):
        """
        Initializes the TypefullyAI API with given parameters.

        Args:
            is_conversation (bool): Whether the provider is in conversation mode.
            max_tokens (int): Maximum number of tokens to sample.
            timeout (int): Timeout for API requests.
            intro (str): Introduction message for the conversation.
            filepath (str): Filepath for storing conversation history.
            update_file (bool): Whether to update the conversation history file.
            proxies (dict): Proxies for the API requests.
            history_offset (int): Offset for conversation history.
            act (str): Act for the conversation.
            system_prompt (str): The system prompt to define the assistant's role.
            model (str): The model identifier to use.

        Examples:
            >>> ai = TypefullyAI(system_prompt="You are a friendly assistant.")
            >>> print(ai.system_prompt)
            'You are a friendly assistant.'
        """
        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://typefully.com/tools/ai/api/completion"
        self.timeout = timeout
        self.last_response = {}
        self.system_prompt = system_prompt
        self.model = model
        self.output_length = max_tokens

        # Initialize LitAgent for user agent generation
        self.agent = LitAgent()

        self.headers = {
            "authority": "typefully.com",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://typefully.com",
            "referer": "https://typefully.com/tools/ai/chat-gpt-alternative",
            "sec-ch-ua": '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "user-agent": self.agent.random()  # Use LitAgent to generate a random user agent
        }

        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Use proxies directly, not session.proxies.update
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

    @staticmethod
    def _typefully_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from the Typefully stream format '0:"..."'."""
        if isinstance(chunk, str):
            match = re.search(r'0:"(.*?)"(?=,|$)', chunk) # Look for 0:"...", possibly followed by comma or end of string
            if match:
                # Decode potential unicode escapes like \u00e9 and handle escaped quotes/backslashes
                content = match.group(1).encode().decode('unicode_escape')
                return content.replace('\\\\', '\\').replace('\\"', '"')
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a prompt to the Typefully AI API and returns the response.

        Args:
            prompt (str): The prompt to send to the API.
            stream (bool): Whether to stream the response.
            raw (bool): Whether to return the raw response.
            optimizer (str): Optimizer to use for the prompt.
            conversationally (bool): Whether to generate the prompt conversationally.

        Returns:
            Dict[str, Any]: The API response.

        Examples:
            >>> ai = TypefullyAI()
            >>> response = ai.ask("Tell me a joke!")
            >>> print(response)
            {'text': 'Why did the scarecrow win an award? Because he was outstanding in his field!'}
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )

        payload = {
            "prompt": conversation_prompt,
            "systemPrompt": self.system_prompt,
            "modelIdentifier": self.model,
            "outputLength": self.output_length
        }

        def for_stream():
            try: # Add try block for CurlError
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint, 
                    headers=self.headers, 
                    json=payload, 
                    stream=True, 
                    timeout=self.timeout,
                    impersonate="chrome120" # Add impersonate
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
                    content_extractor=self._typefully_extractor, # Use the specific extractor
                    end_marker="e:", # Stop processing if "e:" line is encountered (adjust if needed)
                )

                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        yield content_chunk if raw else dict(text=content_chunk)
                # Update history and last response after stream finishes
                self.last_response.update(dict(text=streaming_text))
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e: # Catch other potential exceptions
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")

        def for_non_stream():
            # This function implicitly uses the updated for_stream
            for _ in for_stream():
                pass
            # Ensure last_response is updated by for_stream before returning
            return self.last_response 

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> str:
        """
        Generates a response from the Typefully AI API.

        Args:
            prompt (str): The prompt to send to the API.
            stream (bool): Whether to stream the response.
            optimizer (str): Optimizer to use for the prompt.
            conversationally (bool): Whether to generate the prompt conversationally.

        Returns:
            str: The API response.

        Examples:
            >>> ai = TypefullyAI()
            >>> response = ai.chat("What's the weather today?")
            >>> print(response)
            'The weather today is sunny with a high of 75°F.'
        """

        def for_stream():
            for response in self.ask(
                prompt, True, optimizer=optimizer, conversationally=conversationally
            ):
                yield self.get_message(response)

        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt,
                    False,
                    optimizer=optimizer,
                    conversationally=conversationally,
                )
            )

        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        """
        Extracts the message from the API response.

        Args:
            response (dict): The API response.

        Returns:
            str: The message content.

        Examples:
            >>> ai = TypefullyAI()
            >>> response = ai.ask("Tell me a joke!")
            >>> message = ai.get_message(response)
            >>> print(message)
            'Why did the scarecrow win an award? Because he was outstanding in his field!'
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        # Handle potential unicode escapes in the final text
        # Formatting is now handled by the extractor
        text = response.get("text", "")
        try:
            formatted_text = text.replace('\\n', '\n').replace('\\n\\n', '\n\n')
            return formatted_text
        except Exception: # Catch potential errors during newline replacement
             return text # Return original text if formatting fails


if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)
    
    # Test all available models
    working = 0
    total = len(TypefullyAI.AVAILABLE_MODELS)
    
    for model in TypefullyAI.AVAILABLE_MODELS:
        try:
            test_ai = TypefullyAI(model=model, timeout=60)
            # Test stream first
            response_stream = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            print(f"\r{model:<50} {'Streaming...':<10}", end="", flush=True)
            for chunk in response_stream:
                response_text += chunk
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip() # Already formatted in get_message
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗ (Stream)"
                display_text = "Empty or invalid stream response"
            print(f"\r{model:<50} {status:<10} {display_text}")
            
            # Optional: Add non-stream test if needed
            # print(f"\r{model:<50} {'Non-Stream...':<10}", end="", flush=True)
            # response_non_stream = test_ai.chat("Say 'Hi' again", stream=False)
            # if not response_non_stream or len(response_non_stream.strip()) == 0:
            #      print(f"\r{model:<50} {'✗ (Non-Stream)':<10} Empty non-stream response")

        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
