from typing import Any, Dict, Optional, Union, Generator
from uuid import uuid4
import re
from curl_cffi.requests import Session
from curl_cffi import CurlError
from webscout.AIbase import Provider
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream
from webscout.litagent import LitAgent
from webscout import exceptions

class lmarena(Provider):
    """
    Provider for the Arena API (battle mode).
    """
    AVAILABLE_MODELS = ["arena-battle"]

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
    ):
        self.url = "https://arena-api-stable.vercel.app/evaluation"
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.supabase_jwt = "base64-eyJhY2Nlc3NfdG9rZW4iOiJleUpoYkdjaU9pSklVekkxTmlJc0ltdHBaQ0k2SWtOVFQwNHhkM05uU0hkRlNFTkNNbGNpTENKMGVYQWlPaUpLVjFRaWZRLmV5SnBjM01pT2lKb2RIUndjem92TDJoMWIyZDZiMlZ4ZW1OeVpIWnJkM1IyYjJScExuTjFjR0ZpWVhObExtTnZMMkYxZEdndmRqRWlMQ0p6ZFdJaU9pSXhNV0kxT1ROaE9DMHpZak5sTFRSak1UQXRPRE13TUMwMk16QTBNMk15TW1VeU1qWWlMQ0poZFdRaU9pSmhkWFJvWlc1MGFXTmhkR1ZrSWl3aVpYaHdJam94TnpRM05UUXhPVGc1TENKcFlYUWlPakUzTkRjMU16Z3pPRGtzSW1WdFlXbHNJam9pSWl3aWNHaHZibVVpT2lJaUxDSmhjSEJmYldWMFlXUmhkR0VpT250OUxDSjFjMlZ5WDIxbGRHRmtZWFJoSWpwN0ltbGtJam9pTW1aaU1qVXhPRGN0T0RFMFppMDBNR0l3TFdJNE5UQXRZemswTnpOak1EVTVNakZtSW4wc0luSnZiR1VpT2lKaGRYUm9aVzUwYVdOaGRHVmtJaXdpWVdGc0lqb2lZV0ZzTVNJc0ltRnRjaUk2VzNzaWJXVjBhRzlrSWpvaVlXNXZibmx0YjNWeklpd2lkR2x0WlhOMFlXMXdJam94TnpRME9UWTJOak15ZlYwc0luTmxjM05wYjI1ZmFXUWlPaUl3TjJWaE5UZGtNeTFsT1RNMkxUUXpZVE10WW1Oa05pMW1aREZpTjJOa01ESmpaV0lpTENKcGMxOWhibTl1ZVcxdmRYTWlPblJ5ZFdWOS5jR3VsYlBRRmQ5MzZmcmpmdC1oWjBUQ0k1Rk1sdVU5ZUNac3h3VkNVTkhrIiwidG9rZW5fdHlwZSI6ImJlYXJlciIsImV4cGlyZXNfaW4iOjM2MDAsImV4cGlyZXNfYXQiOjE3NDc1NDE5ODksInJlZnJlc2hfdG9rZW4iOiJ0cG95ZTViN2s2N3kiLCJ1c2VyIjp7ImlkIjoiMTFiNTkzYTgtM2IzZS00YzEwLTgzMDAtNjMwNDNjMjJlMjI2IiwiYXVkIjoiYXV0aGVudGljYXRlZCIsInJvbGUiOiJhdXRoZW50aWNhdGVkIiwiZW1haWwiOiIiLCJwaG9uZSI6IiIsImxhc3Rfc2lnbl9pbl9hdCI6IjIwMjUtMDQtMThUMDg6NTc6MTIuNzMxMzg1WiIsImFwcF9tZXRhZGF0YSI6e30sInVzZXJfbWV0YWRhdGEiOnsiaWQiOiIyZmIyNTE4Ny04MTRmLTQwYjAtYjg1MC1jOTQ3M2MwNTkyMWYifSwiaWRlbnRpdGllcyI6W10sImNyZWF0ZWRfYXQiOiIyMDI1LTA0LTE4VDA4OjU3OjEyLjcyODk4NVoiLCJ1cGRhdGVkX2F0IjoiMjAyNS0wNS0xOFQwMzoxOTo0OS4yNzkxNzJaIiwiaXNfYW5vbnltb3VzIjp0cnVlfX0"
        self.agent = LitAgent()
        self.headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": "https://beta.lmarena.ai",
            "referer": "https://arena-api-stable.vercel.app/",
        }
        if self.supabase_jwt:
            self.headers["supabase-jwt"] = self.supabase_jwt
        self.session.headers.update(self.headers)
        self.session.proxies = proxies
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

    @staticmethod
    def _arena_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from the arena stream format 'b0:"..."'."""
        if isinstance(chunk, str):
            match = re.search(r'b0:"(.*?)"', chunk)
            if match:
                content = match.group(1).encode().decode('unicode_escape')
                return content.replace('\\', '\\').replace('\\"', '"')
        return None

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        session_id = str(uuid4())
        user_msg_id = str(uuid4())
        model_a_msg_id = str(uuid4())
        model_b_msg_id = str(uuid4())
        return {
            "id": session_id,
            "mode": "battle",
            "userMessageId": user_msg_id,
            "modelAMessageId": model_a_msg_id,
            "modelBMessageId": model_b_msg_id,
            "messages": [
                {
                    "id": user_msg_id,
                    "role": "user",
                    "content": prompt,
                    "experimental_attachments": [],
                    "parentMessageIds": [],
                    "participantPosition": "a",
                    "modelId": None,
                    "evaluationSessionId": session_id,
                    "status": "pending",
                    "failureReason": None
                }
            ],
            "modality": "chat"
        }

    def ask(
        self,
        prompt: str,
        stream: bool = False,
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
        payload = self._build_payload(conversation_prompt)
        def for_stream():
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )
                if not response.ok:
                    raise exceptions.FailedToGenerateResponseError(
                        f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                    )
                streaming_text = ""
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value=None,
                    to_json=False,
                    content_extractor=self._arena_extractor
                )
                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        yield content_chunk if raw else dict(text=content_chunk)
                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")
        def for_non_stream():
            full_text = ""
            for chunk_data in for_stream():
                if isinstance(chunk_data, dict) and "text" in chunk_data:
                    full_text += chunk_data["text"]
                elif isinstance(chunk_data, str):
                    full_text += chunk_data
            self.last_response = {"text": full_text}
            return self.last_response
        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator]:
        def for_stream():
            for response in self.ask(prompt, True, optimizer=optimizer, conversationally=conversationally):
                yield self.get_message(response)
        def for_non_stream():
            return self.get_message(
                self.ask(prompt, False, optimizer=optimizer, conversationally=conversationally)
            )
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<20} {'Status':<10} {'Response'}")
    print("-" * 80)
    try:
        test_ai = lmarena(timeout=60)
        response = test_ai.chat("Say 'Hello' in one word", stream=True)
        response_text = ""
        for chunk in response:
            response_text += chunk
            print(f"\r{'arena-battle':<20} {'Testing...':<10}", end="", flush=True)
        if response_text and len(response_text.strip()) > 0:
            status = "✓"
            display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
        else:
            status = "✗"
            display_text = "Empty or invalid response"
        print(f"\r{'arena-battle':<20} {status:<10} {display_text}")
    except Exception as e:
        print(f"\r{'arena-battle':<20} {'✗':<10} {str(e)}")
