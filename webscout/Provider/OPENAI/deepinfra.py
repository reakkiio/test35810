import requests
import json
import time
import uuid
from typing import List, Dict, Optional, Union, Generator, Any

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

# Attempt to import LitAgent, fallback if not available
try:
    from webscout.litagent import LitAgent
except ImportError:
    pass

# --- DeepInfra Client ---

class Completions(BaseCompletions):
    def __init__(self, client: 'DeepInfra'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 2049,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation.
        Mimics openai.chat.completions.create
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        payload.update(kwargs)

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload, timeout, proxies)
        else:
            return self._create_non_stream(request_id, created_time, model, payload, timeout, proxies)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any],
        timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.base_url,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies
            )
            response.raise_for_status()

            # Track token usage across chunks
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()

                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str == "[DONE]":
                            # Format the final [DONE] marker in OpenAI format
                            # print("data: [DONE]")
                            break

                        try:
                            data = json.loads(json_str)
                            choice_data = data.get('choices', [{}])[0]
                            delta_data = choice_data.get('delta', {})
                            finish_reason = choice_data.get('finish_reason')

                            # Update token counts if available
                            usage_data = data.get('usage', {})
                            if usage_data:
                                prompt_tokens = usage_data.get('prompt_tokens', prompt_tokens)
                                completion_tokens = usage_data.get('completion_tokens', completion_tokens)
                                total_tokens = usage_data.get('total_tokens', total_tokens)

                            # Create the delta object
                            delta = ChoiceDelta(
                                content=delta_data.get('content'),
                                role=delta_data.get('role'),
                                tool_calls=delta_data.get('tool_calls')
                            )

                            # Create the choice object
                            choice = Choice(
                                index=choice_data.get('index', 0),
                                delta=delta,
                                finish_reason=finish_reason,
                                logprobs=choice_data.get('logprobs')
                            )

                            # Create the chunk object
                            chunk = ChatCompletionChunk(
                                id=request_id,
                                choices=[choice],
                                created=created_time,
                                model=model,
                                system_fingerprint=data.get('system_fingerprint')
                            )

                            # Convert chunk to dict using Pydantic's API
                            if hasattr(chunk, "model_dump"):
                                chunk_dict = chunk.model_dump(exclude_none=True)
                            else:
                                chunk_dict = chunk.dict(exclude_none=True)

                            # Add usage information to match OpenAI format
                            # Even if we don't have real token counts, include estimated usage
                            # This matches the format in the examples
                            usage_dict = {
                                "prompt_tokens": prompt_tokens or 10,
                                "completion_tokens": completion_tokens or (len(delta_data.get('content', '')) if delta_data.get('content') else 0),
                                "total_tokens": total_tokens or (10 + (len(delta_data.get('content', '')) if delta_data.get('content') else 0)),
                                "estimated_cost": None
                            }

                            # Update completion_tokens and total_tokens as we receive more content
                            if delta_data.get('content'):
                                completion_tokens += 1
                                total_tokens = prompt_tokens + completion_tokens
                                usage_dict["completion_tokens"] = completion_tokens
                                usage_dict["total_tokens"] = total_tokens

                            chunk_dict["usage"] = usage_dict

                            # Format the response in OpenAI format exactly as requested
                            # We need to print the raw string and also yield the chunk object
                            # This ensures both the console output and the returned object are correct
                            # print(f"data: {json.dumps(chunk_dict)}")

                            # Return the chunk object for internal processing
                            yield chunk
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line: {json_str}")
                            continue
        except requests.exceptions.RequestException as e:
            print(f"Error during DeepInfra stream request: {e}")
            raise IOError(f"DeepInfra request failed: {e}") from e
        except Exception as e:
            print(f"Error processing DeepInfra stream: {e}")
            raise

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any],
        timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.base_url,
                headers=self._client.headers,
                json=payload,
                timeout=timeout or self._client.timeout,
                proxies=proxies
            )
            response.raise_for_status()
            data = response.json()

            choices_data = data.get('choices', [])
            usage_data = data.get('usage', {})

            choices = []
            for choice_d in choices_data:
                message_d = choice_d.get('message', {})
                message = ChatCompletionMessage(
                    role=message_d.get('role', 'assistant'),
                    content=message_d.get('content', '')
                )
                choice = Choice(
                    index=choice_d.get('index', 0),
                    message=message,
                    finish_reason=choice_d.get('finish_reason', 'stop')
                )
                choices.append(choice)

            usage = CompletionUsage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0)
            )

            completion = ChatCompletion(
                id=request_id,
                choices=choices,
                created=created_time,
                model=data.get('model', model),
                usage=usage,
            )
            return completion

        except requests.exceptions.RequestException as e:
            print(f"Error during DeepInfra non-stream request: {e}")
            raise IOError(f"DeepInfra request failed: {e}") from e
        except Exception as e:
            print(f"Error processing DeepInfra response: {e}")
            raise

class Chat(BaseChat):
    def __init__(self, client: 'DeepInfra'):
        self.completions = Completions(client)

class DeepInfra(OpenAICompatibleProvider):
    
    AVAILABLE_MODELS = [
        # "anthropic/claude-3-7-sonnet-latest",  # >>>> NOT WORKING
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Turbo",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-Prover-V2-671B",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-Guard-4-12B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "microsoft/Phi-4-multimodal-instruct",
        "microsoft/WizardLM-2-8x22B",
        "microsoft/phi-4",
        "microsoft/phi-4-reasoning-plus",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        "Qwen/QwQ-32B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
        # "google/gemini-1.5-flash",  # >>>> NOT WORKING
        # "google/gemini-1.5-flash-8b",  # >>>> NOT WORKING
        # "google/gemini-2.0-flash-001",  # >>>> NOT WORKING

        # "Gryphe/MythoMax-L2-13b",  # >>>> NOT WORKING

        # "meta-llama/Llama-3.2-1B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Llama-3.2-3B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Llama-3.2-90B-Vision-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Llama-3.2-11B-Vision-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3-70B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3-8B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3.1-70B-Instruct",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",  # >>>> NOT WORKING
        # "meta-llama/Meta-Llama-3.1-405B-Instruct",  # >>>> NOT WORKING
        # "mistralai/Mixtral-8x7B-Instruct-v0.1",  # >>>> NOT WORKING
        # "mistralai/Mistral-7B-Instruct-v0.3",  # >>>> NOT WORKING
        # "mistralai/Mistral-Nemo-Instruct-2407",  # >>>> NOT WORKING
        # "NousResearch/Hermes-3-Llama-3.1-405B",  # >>>> NOT WORKING
        # "NovaSky-AI/Sky-T1-32B-Preview",  # >>>> NOT WORKING
        # "Qwen/Qwen2.5-7B-Instruct",  # >>>> NOT WORKING
        # "Sao10K/L3.1-70B-Euryale-v2.2",  # >>>> NOT WORKING
        # "Sao10K/L3.3-70B-Euryale-v2.3",  # >>>> NOT WORKING
    ]

    def __init__(self, browser: str = "chrome"):
        self.timeout = None # Default timeout
        self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        self.session = requests.Session()

        agent = LitAgent()
        fingerprint = agent.generate_fingerprint(browser)

        self.headers = {
            "Accept": fingerprint["accept"],
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": fingerprint["accept_language"],
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Origin": "https://deepinfra.com",
            "Pragma": "no-cache",
            "Referer": "https://deepinfra.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "X-Deepinfra-Source": "web-embed",
            "Sec-CH-UA": fingerprint["sec_ch_ua"] or '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{fingerprint["platform"]}"',
            "User-Agent": fingerprint["user_agent"],
        }
        self.session.headers.update(self.headers)
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()