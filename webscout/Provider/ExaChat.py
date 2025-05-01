from curl_cffi import CurlError
from curl_cffi.requests import Session, Response # Import Response
import json
import uuid
from typing import Any, Dict, Union, Optional, List
from datetime import datetime
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider 
from webscout import exceptions
from webscout.litagent import LitAgent

# Model configurations
MODEL_CONFIGS = {
    "exaanswer": {
        "endpoint": "https://ayle.chat/api/exaanswer",
        "models": ["exaanswer"],
    },
    "gemini": {
        "endpoint": "https://ayle.chat/api/gemini",
        "models": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp-image-generation",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.5-pro-exp-03-25",
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.5-flash-preview-04-17",

        
        ],
    },
    "openrouter": {
        "endpoint": "https://ayle.chat/api/openrouter",
        "models": [
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "google/gemma-3-27b-it:free",
            "meta-llama/llama-4-maverick:free",           
        ],
    },
    "groq": {
        "endpoint": "https://ayle.chat/api/groq",
        "models": [
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-90b-vision-preview",
            "llama-3.3-70b-specdec",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "qwen-2.5-32b",
            "qwen-2.5-coder-32b",
            "qwen-qwq-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ],
    },
    "cerebras": {
        "endpoint": "https://ayle.chat/api/cerebras",
        "models": [
            "llama3.1-8b",
            "llama-3.3-70b"
        ],
    },
    "xai": {
        "endpoint": "https://ayle.chat/api/xai",
        "models": [
            "grok-3-mini-beta"
        ],
    },
}

class ExaChat(Provider):
    """
    A class to interact with multiple AI APIs through the Exa Chat interface.
    """
    AVAILABLE_MODELS = [
        # ExaAnswer Models
        "exaanswer",

        # XAI Models
        "grok-3-mini-beta",
        
        # Gemini Models
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.5-flash-preview-04-17",
        
        # OpenRouter Models
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-4-maverick:free",
        
        # Groq Models
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-90b-vision-preview",
        "llama-3.3-70b-specdec",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "qwen-2.5-32b",
        "qwen-2.5-coder-32b",
        "qwen-qwq-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",

        
        # Cerebras Models
        "llama3.1-8b",
        "llama-3.3-70b",

    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 4000,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "exaanswer",
        system_prompt: str = "You are a friendly, helpful AI assistant.",
        temperature: float = 0.5,
        presence_penalty: int = 0,
        frequency_penalty: int = 0,
        top_p: float = 1
    ):
        """Initializes the ExaChat client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
            
        self.session = Session() # Use curl_cffi Session
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.top_p = top_p
        
        # Initialize LitAgent for user agent generation
        self.agent = LitAgent()
        
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://ayle.chat/",
            "referer": "https://ayle.chat/",
            "user-agent": self.agent.random(),
        }
        
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly
        self.session.cookies.update({"session": uuid.uuid4().hex})

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

        self.provider = self._get_provider_from_model(self.model)
        self.model_name = self.model

    def _get_endpoint(self) -> str:
        """Get the API endpoint for the current provider."""
        return MODEL_CONFIGS[self.provider]["endpoint"]

    def _get_provider_from_model(self, model: str) -> str:
        """Determine the provider based on the model name."""
        for provider, config in MODEL_CONFIGS.items():
            if model in config["models"]:
                return provider
        
        available_models = []
        for provider, config in MODEL_CONFIGS.items():
            for model_name in config["models"]:
                available_models.append(f"{provider}/{model_name}")
        
        error_msg = f"Invalid model: {model}\nAvailable models: {', '.join(available_models)}"
        raise ValueError(error_msg)

    @staticmethod
    def _exachat_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from ExaChat stream JSON objects."""
        if isinstance(chunk, dict):
            return chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        return None

    def _make_request(self, payload: Dict[str, Any]) -> Response: # Change type hint to Response
        """Make the API request with proper error handling."""
        try:
            response = self.session.post(
                self._get_endpoint(),
                headers=self.headers,
                json=payload,
                timeout=self.timeout, # type: ignore
                stream=True, # Enable streaming for the request
                impersonate="chrome120" # Add impersonate
            )
            response.raise_for_status()
            return response
        except (CurlError, exceptions.FailedToGenerateResponseError, Exception) as e: # Catch CurlError and others
            raise exceptions.FailedToGenerateResponseError(f"API request failed: {e}") from e

    def _build_payload(self, conversation_prompt: str) -> Dict[str, Any]:
        """Build the appropriate payload based on the provider."""
        if self.provider == "exaanswer":
            return {
                "query": conversation_prompt,
                "messages": []
            }
        elif self.provider == "gemini":
            return {
                "query": conversation_prompt,
                "model": self.model,
                "messages": []
            }
        elif self.provider == "cerebras":
            return {
                "query": conversation_prompt,
                "model": self.model,
                "messages": []
            }
        else:  # openrouter or groq
            return {
                "query": conversation_prompt + "\n",  # Add newline for openrouter and groq models
                "model": self.model,
                "messages": []
            }

    def ask(
        self,
        prompt: str,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Dict[str, Any]:
        """Sends a prompt to the API and returns the response."""
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                error_msg = f"Optimizer is not one of {self.__available_optimizers}"
                raise exceptions.FailedToGenerateResponseError(error_msg)

        payload = self._build_payload(conversation_prompt)
        response = self._make_request(payload)
        
        try:
            full_response = ""
            # Use sanitize_stream to process the response
            processed_stream = sanitize_stream(
                data=response.iter_content(chunk_size=None), # Pass byte iterator
                intro_value=None, # API doesn't seem to use 'data:' prefix
                to_json=True,     # Stream sends JSON lines
                content_extractor=self._exachat_extractor, # Use the specific extractor
                yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
            )

            for content_chunk in processed_stream:
                # content_chunk is the string extracted by _exachat_extractor
                if content_chunk and isinstance(content_chunk, str):
                    full_response += content_chunk
                            
            self.last_response = {"text": full_response}
            self.conversation.update_chat_history(prompt, full_response)
            return self.last_response if not raw else full_response # Return dict or raw string
            
        except json.JSONDecodeError as e:
            raise exceptions.FailedToGenerateResponseError(f"Invalid JSON response: {e}") from e

    def chat(
        self,
        prompt: str,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> str:
        """Generate response."""
        response = self.ask(
            prompt, optimizer=optimizer, conversationally=conversationally
        )
        return self.get_message(response)

    def get_message(self, response: Union[Dict[str, Any], str]) -> str:
        """
        Retrieves message from response.
        
        Args:
            response (Union[Dict[str, Any], str]): The response to extract the message from
            
        Returns:
            str: The extracted message text
        """
        if isinstance(response, dict):
            return response.get("text", "")
        return str(response)

if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)
    
    # Test all available models
    working = 0
    total = len(ExaChat.AVAILABLE_MODELS)
    
    for model in ExaChat.AVAILABLE_MODELS:
        try:
            test_ai = ExaChat(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word")
            response_text = response
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model:<50} {'✗':<10} {str(e)}")
