from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import re
from typing import Any, Dict, Optional, Union, Generator
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent as Lit

class PIZZAGPT(Provider):
    """
    PIZZAGPT is a provider class for interacting with the PizzaGPT API.
    Supports web search integration and handles responses using regex.
    """
    AVAILABLE_MODELS = ["gpt-4o-mini"]
    
    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 600, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gpt-4o-mini"
    ) -> None:
        """Initialize PizzaGPT with enhanced configuration options."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
            
        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://www.pizzagpt.it/api/chatx-completion"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        
        self.headers = {
            "accept": "application/json",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://www.pizzagpt.it",
            "referer": "https://www.pizzagpt.it/en",
            "user-agent": Lit().random(),
            "x-secret": "Marinara",
        }

        self.__available_optimizers = (
            method for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly

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

    def _extract_content(self, text: str) -> Dict[str, Any]:
        """
        Extract content from response text using regex.
        """
        try:
            # Look for content pattern
            content_match = re.search(r'"content"\s*:\s*"(.*?)"(?=\s*[,}])', text, re.DOTALL)
            if not content_match:
                raise exceptions.FailedToGenerateResponseError("Content not found in response")
                
            content = content_match.group(1)
            # Unescape special characters
            content = content.encode().decode('unicode_escape')
            
            # Look for citations if present
            citations = []
            citations_match = re.search(r'"citations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if citations_match:
                citations_text = citations_match.group(1)
                citations = re.findall(r'"(.*?)"', citations_text)
            
            return {
                "content": content,
                "citations": citations
            }
            
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Failed to extract content: {str(e)}")

    def ask(
        self,
        prompt: str,
        stream: bool = False, # Note: API does not support streaming
        raw: bool = False, # Keep raw param for interface consistency
        optimizer: str = None,
        conversationally: bool = False,
        web_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a prompt to PizzaGPT API with optional web search capability.
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        payload = {
            "question": conversation_prompt,
            "model": self.model,
            "searchEnabled": web_search
        }

        try:
            # Use curl_cffi session post with impersonate
            response = self.session.post(
                self.api_endpoint,
                # headers are set on the session
                json=payload,
                timeout=self.timeout,
                # proxies are set on the session
                impersonate="chrome110" # Use a common impersonation profile
            )
            
            response.raise_for_status() # Check for HTTP errors

            response_text = response.text
            if not response_text:
                raise exceptions.FailedToGenerateResponseError("Empty response received from API")

            try:
                resp = self._extract_content(response_text)
                    
                self.last_response = {"text": resp['content']} # Store only text in last_response
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
                # Return the full extracted data (content + citations) or raw text
                return response_text if raw else resp 

            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Failed to parse response: {str(e)}")

        except CurlError as e: # Catch CurlError
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
        except Exception as e: # Catch other potential exceptions (like HTTPError)
            err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
            raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e} - {err_text}") from e

    def chat(
        self,
        prompt: str,
        stream: bool = False, # Keep stream param for interface consistency
        optimizer: str = None,
        conversationally: bool = False,
        web_search: bool = False,
        # Add raw parameter for consistency
        raw: bool = False
    ) -> str:
        """
        Chat with PizzaGPT with optional web search capability.
        """
        # API doesn't stream, call ask directly
        response_data = self.ask(
            prompt,
            stream=False, # Call ask in non-stream mode
            raw=raw, # Pass raw flag to ask
            optimizer=optimizer,
            conversationally=conversationally,
            web_search=web_search
        )
        # If raw=True, ask returns string, otherwise dict
        return response_data if raw else self.get_message(response_data)


    def get_message(self, response: dict) -> str:
        """Extract message from response dictionary."""
        # Handle case where raw response (string) might be passed mistakenly
        if isinstance(response, str):
             # Attempt to parse if it looks like the expected structure, otherwise return as is
             try:
                 extracted = self._extract_content(response)
                 return extracted.get("content", "")
             except:
                 return response # Return raw string if parsing fails
        elif isinstance(response, dict):
             # If it's already the extracted dict from ask(raw=False)
             if "content" in response:
                 return response.get("content", "")
             # If it's the last_response format
             elif "text" in response:
                 return response.get("text", "")
        return "" # Default empty string

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    from rich import print
    
    # Example usage with web search enabled
    ai = PIZZAGPT(timeout=60)
    try:
        print("[bold blue]Testing Chat (Web Search Disabled):[/bold blue]")
        response = ai.chat("hi", web_search=False)
        print(response)
        
        # print("\n[bold blue]Testing Chat (Web Search Enabled):[/bold blue]")
        # response_web = ai.chat("What's the weather in Rome?", web_search=True)
        # print(response_web)

    except Exception as e:
        print(f"[bold red]Error:[/bold red] {str(e)}")