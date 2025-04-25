from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
from typing import Union, Any, Dict, Generator, Optional, List
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent as Lit

class TextPollinationsAI(Provider):
    """
    A class to interact with the Pollinations AI API.
    """

    AVAILABLE_MODELS = [
        "openai",              # OpenAI GPT-4.1-nano (Azure) - vision capable
        "openai-large",        # OpenAI GPT-4.1 mini (Azure) - vision capable
        "openai-reasoning",    # OpenAI o4-mini (Azure) - vision capable, reasoning
        "qwen-coder",          # Qwen 2.5 Coder 32B (Scaleway)
        "llama",               # Llama 3.3 70B (Cloudflare)
        "llamascout",          # Llama 4 Scout 17B (Cloudflare)
        "mistral",             # Mistral Small 3 (Scaleway) - vision capable
        "unity",               # Unity Mistral Large (Scaleway) - vision capable, uncensored
        "midijourney",         # Midijourney (Azure)
        "rtist",               # Rtist (Azure)
        "searchgpt",           # SearchGPT (Azure) - vision capable
        "evil",                # Evil (Scaleway) - vision capable, uncensored
        "deepseek-reasoning",  # DeepSeek-R1 Distill Qwen 32B (Cloudflare) - reasoning
        "deepseek-reasoning-large", # DeepSeek R1 - Llama 70B (Scaleway) - reasoning
        "phi",                 # Phi-4 Instruct (Cloudflare) - vision and audio capable
        "llama-vision",        # Llama 3.2 11B Vision (Cloudflare) - vision capable
        "gemini",              # gemini-2.5-flash-preview-04-17 (Azure) - vision and audio capable
        "hormoz",              # Hormoz 8b (Modal)
        "hypnosis-tracy",      # Hypnosis Tracy 7B (Azure) - audio capable
        "deepseek",            # DeepSeek-V3 (DeepSeek)
        "sur",                 # Sur AI Assistant (Mistral) (Scaleway) - vision capable
        "openai-audio",        # OpenAI GPT-4o-audio-preview (Azure) - vision and audio capable
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 8096, # Note: max_tokens is not directly used by this API endpoint
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "openai-large",
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        """Initializes the TextPollinationsAI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://text.pollinations.ai/openai"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt

        self.headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': Lit().random(),
            'Content-Type': 'application/json',
            # Add sec-ch-ua headers if needed for impersonation consistency
        }

        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly

        self.__available_optimizers = (
            method for method in dir(Optimizers)
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

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Generator[Any, None, None]]:
        """Chat with AI"""
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt}
            ],
            "model": self.model,
            "stream": stream,
        }

        # Add function calling parameters if provided
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        def for_stream():
            try: # Add try block for CurlError
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint,
                    # headers are set on the session
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome120" # Add impersonate
                )

                if not response.ok:
                    raise exceptions.FailedToGenerateResponseError(
                        f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                    )

                full_response = ""
                # Iterate over bytes and decode manually
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        line = line_bytes.decode('utf-8').strip()
                        if line == "data: [DONE]":
                            break
                        if line.startswith('data: '):
                            try:
                                json_data = json.loads(line[6:])
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    choice = json_data['choices'][0]
                                    if 'delta' in choice:
                                        if 'content' in choice['delta'] and choice['delta']['content'] is not None:
                                            content = choice['delta']['content']
                                            full_response += content
                                            # Yield dict or raw string
                                            yield content if raw else dict(text=content)
                                        elif 'tool_calls' in choice['delta']:
                                            # Handle tool calls in streaming response
                                            tool_calls = choice['delta']['tool_calls']
                                            # Yield dict or raw list
                                            yield tool_calls if raw else dict(tool_calls=tool_calls)
                            except json.JSONDecodeError:
                                continue
                            except UnicodeDecodeError:
                                continue

                # Update history and last response after stream finishes
                # Note: last_response might only contain text, not tool calls if they occurred
                self.last_response.update(dict(text=full_response))
                if full_response: # Only update history if text was received
                    self.conversation.update_chat_history(
                        prompt, full_response # Use the fully aggregated text
                    )
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}") from e


        def for_non_stream():
            # Aggregate the stream using the updated for_stream logic
            final_content = ""
            tool_calls_aggregated = None # To store potential tool calls
            for chunk_data in for_stream():
                if isinstance(chunk_data, dict):
                    if "text" in chunk_data:
                        final_content += chunk_data["text"]
                    elif "tool_calls" in chunk_data:
                        # Aggregate tool calls (simple aggregation, might need refinement)
                        if tool_calls_aggregated is None:
                            tool_calls_aggregated = []
                        tool_calls_aggregated.extend(chunk_data["tool_calls"])
                elif isinstance(chunk_data, str): # Handle raw stream case
                    final_content += chunk_data
                # Handle raw tool calls list if raw=True
                elif isinstance(chunk_data, list) and raw:
                     if tool_calls_aggregated is None:
                         tool_calls_aggregated = []
                     tool_calls_aggregated.extend(chunk_data)


            # last_response and history are updated within for_stream (for text)
            # Return a dict containing text and/or tool_calls
            result = {}
            if final_content:
                result["text"] = final_content
            if tool_calls_aggregated:
                result["tool_calls"] = tool_calls_aggregated
            self.last_response = result # Update last_response with aggregated result
            return self.last_response


        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response as a string"""
        def for_stream():
            for response in self.ask(
                prompt, True, optimizer=optimizer, conversationally=conversationally,
                tools=tools, tool_choice=tool_choice
            ):
                yield self.get_message(response)

        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt,
                    False,
                    optimizer=optimizer,
                    conversationally=conversationally,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            )

        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        """Retrieves message only from response"""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        if "text" in response:
            return response["text"]
        elif "tool_calls" in response:
            # For tool calls, return a string representation
            return json.dumps(response["tool_calls"])

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    # Test all available models
    working = 0
    total = len(TextPollinationsAI.AVAILABLE_MODELS)

    for model in TextPollinationsAI.AVAILABLE_MODELS:
        try:
            test_ai = TextPollinationsAI(model=model, timeout=60)
            # Test stream first
            response_stream = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            print(f"\r{model:<50} {'Streaming...':<10}", end="", flush=True)
            for chunk in response_stream:
                response_text += chunk

            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip()
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
