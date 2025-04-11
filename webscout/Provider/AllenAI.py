import requests
import json
import os
from uuid import uuid4
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent

class AllenAI(Provider):
    """
    A class to interact with the AllenAI (Ai2 Playground) API.
    """

    AVAILABLE_MODELS = [
        'OLMo-2-1124-13B-Instruct',  
        'Llama-3-1-Tulu-3-8B',
        'olmo-2-0325-32b-instruct',
        'Llama-3-1-Tulu-3-70B',
        'OLMoE-1B-7B-0924-Instruct',
        'tulu3-405b',
        'olmo-2-0325-32b-instruct',
        'tulu-3-1-8b',
        'olmoe-0125'
    ]

    # Default model options from JS implementation
    DEFAULT_OPTIONS = {
        "max_tokens": 2048, 
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stop": None,
        "logprobs": None
    }
    
    # Host mapping for models - some models work best with specific hosts
    MODEL_HOST_MAP = {
        'tulu3-405b': 'inferd',
        'tulu2': 'inferd',
        'olmo-7b-instruct': 'inferd'
    }

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2048,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "OLMo-2-1124-13B-Instruct",
        host: str = None  # Now optional - will auto-detect if not provided
    ):
        """Initializes the AllenAI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
            
        self.url = "https://playground.allenai.org"
        # Updated API endpoint to v3 from v4
        self.api_endpoint = "https://olmo-api.allen.ai/v3/message/stream"
        self.whoami_endpoint = "https://olmo-api.allen.ai/v3/whoami"
        
        # Updated headers based on JS implementation
        self.headers = {
            'User-Agent': "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
            'Accept': '*/*',
            'Accept-Language': 'id-ID,id;q=0.9',
            'Origin': self.url,
            'Referer': f"{self.url}/",
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Priority': 'u=1, i',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'sec-ch-ua': '"Chromium";v="131", "Not_A Brand";v="24", "Microsoft Edge Simulate";v="131", "Lemur";v="131"',
            'sec-ch-ua-mobile': '?1',
            'sec-ch-ua-platform': '"Android"',
            'Content-Type': 'application/json'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(proxies)
        self.model = model
        
        # Auto-detect host if not provided
        if not host:
            # Use the preferred host from the model-host map, or default to modal
            self.host = self.MODEL_HOST_MAP.get(model, 'modal')
        else:
            self.host = host
            
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}       
        # Generate user ID if needed
        self.x_anonymous_user_id = None
        self.parent = None
        
        # Default options
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options["max_tokens"] = max_tokens

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

    def whoami(self):
        """Gets or creates a user ID for authentication with Allen AI API"""
        temp_id = str(uuid4())
        headers = self.session.headers.copy()
        headers.update({"x-anonymous-user-id": temp_id})
        
        try:
            response = self.session.get(
                self.whoami_endpoint,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.x_anonymous_user_id = data.get("client", temp_id)
                return data
            else:
                self.x_anonymous_user_id = temp_id
                return {"client": temp_id}
                
        except Exception as e:
            self.x_anonymous_user_id = temp_id
            return {"client": temp_id, "error": str(e)}
    

    def parse_stream(self, raw_data):
        """Parse the raw streaming data according to the JS implementation"""
        result = ""
        for line in raw_data.splitlines():
            try:
                parsed = json.loads(line)
                # Check if message starts with msg_ pattern
                if parsed.get("message", "").startswith("msg_"):
                    result += parsed.get("content", "")
            except:
                continue
        return result

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        host: str = None,
        private: bool = False,
        top_p: float = None,
        temperature: float = None,
        options: dict = None,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        # Ensure we have a user ID
        if not self.x_anonymous_user_id:
            self.whoami()
        
        # Prepare the API request
        self.session.headers.update({
            "x-anonymous-user-id": self.x_anonymous_user_id,
            "Content-Type": "application/json"
        })
        
        # Create options dictionary
        opts = self.options.copy()
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p
        if options:
            opts.update(options)
        
        # Use the host param or the default host
        use_host = host or self.host
        
        # List of hosts to try - start with provided host, then try alternative hosts
        hosts_to_try = [use_host]
        if use_host == 'modal':
            hosts_to_try.append('inferd')
        else:
            hosts_to_try.append('modal')
        
        last_error = None
        
        # Try each host until one works
        for current_host in hosts_to_try:
            # Create the JSON payload as per the JS implementation
            payload = {
                "content": conversation_prompt,
                "private": private,
                "model": self.model,
                "host": current_host,
                "opts": opts
            }
            
            # Add parent if exists
            if self.parent:
                payload["parent"] = self.parent
                
            try:
                if stream:
                    return self._stream_request(payload, prompt, raw)
                else:
                    return self._non_stream_request(payload, prompt)
            except exceptions.FailedToGenerateResponseError as e:
                last_error = e
                # Log the error but continue to try other hosts
                print(f"Host '{current_host}' failed for model '{self.model}', trying next host...")
                continue
        
        # If we've tried all hosts and none worked, raise the last error
        raise last_error or exceptions.FailedToGenerateResponseError("All hosts failed. Unable to complete request.")

    def _stream_request(self, payload, prompt, raw=False):
        """Handle streaming requests with the given payload"""
        try:
            response = self.session.post(
                self.api_endpoint,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise exceptions.FailedToGenerateResponseError(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            
            streaming_text = ""
            current_parent = None
            
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=False):
                if not chunk:
                    continue
                    
                decoded = chunk.decode(errors="ignore")
                for line in decoded.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    if isinstance(data, dict):
                        # Check for message pattern from JS implementation
                        if data.get("message", "").startswith("msg_") and "content" in data:
                            content = data.get("content", "")
                            if content:
                                streaming_text += content
                                resp = dict(text=content)
                                yield resp if raw else resp
                        
                        # Legacy handling for older API
                        elif "message" in data and data.get("content"):
                            content = data.get("content")
                            if content.strip():
                                streaming_text += content
                                resp = dict(text=content)
                                yield resp if raw else resp
                        
                        # Update parent ID if present
                        if data.get("id"):
                            current_parent = data.get("id")
                        elif data.get("children"):
                            for child in data["children"]:
                                if child.get("role") == "assistant":
                                    current_parent = child.get("id")
                                    break
                        
                        # Handle completion
                        if data.get("final") or data.get("finish_reason") == "stop":
                            if current_parent:
                                self.parent = current_parent
                            
                            # Update conversation history
                            self.conversation.update_chat_history(prompt, streaming_text)
                            self.last_response = {"text": streaming_text}
                            return
            
        except requests.RequestException as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed: {str(e)}")

    def _non_stream_request(self, payload, prompt):
        """Handle non-streaming requests with the given payload"""
        try:
            # For non-streaming requests, we can directly send without stream=True
            response = self.session.post(
                self.api_endpoint,
                json=payload,
                stream=False,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise exceptions.FailedToGenerateResponseError(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            
            # Parse the response as per JS implementation
            raw_response = response.text
            parsed_response = self.parse_stream(raw_response)
            self.conversation.update_chat_history(prompt, parsed_response)
            self.last_response = {"text": parsed_response}
            return self.last_response
            
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed: {str(e)}")

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        host: str = None,
        options: dict = None,
    ) -> str:
        def for_stream():
            for response in self.ask(
                prompt, 
                True, 
                optimizer=optimizer, 
                conversationally=conversationally,
                host=host,
                options=options
            ):
                yield self.get_message(response)
        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt, 
                    False, 
                    optimizer=optimizer, 
                    conversationally=conversationally,
                    host=host,
                    options=options
                )
            )
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]



if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in AllenAI.AVAILABLE_MODELS:
        try:
            # Auto-detect host
            test_ai = AllenAI(model=model, timeout=60)
            # Pass the host explicitly to display accurate error messages
            response = test_ai.chat("Say 'Hello' in one word")
            response_text = response
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
                print(f"{model:<50} {status:<10} {display_text} (host: {test_ai.host})")
            else:
                status = "✗"
                display_text = "Empty or invalid response"
                print(f"{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model:<50} {'✗':<10} {str(e)}")