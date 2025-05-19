import random
import re
import time
from typing import Dict, List, Optional, Any, Generator, Tuple, Union
import uuid
from curl_cffi.requests import Session
from curl_cffi import CurlError
from webscout.AIutel import Conversation as JsonConversation
from webscout import exceptions
from webscout.Provider.Blackboxai import to_data_uri

class RateLimitError(Exception):
    pass

class ModelNotFoundError(Exception):
    pass

Messages = List[Dict[str, Any]]
MediaListType = List[Tuple[Any, str]]
Result = Generator[Union[str, JsonConversation, Any], None, None]

class AuthData:
    def __init__(self):
        self.auth_token: Optional[str] = None
        self.app_token: Optional[str] = None
        self.created_at: float = time.time()
        self.tokens_valid: bool = False
        self.rate_limited_until: float = 0
    def is_valid(self, expiration_time: int) -> bool:
        return (self.auth_token and self.app_token and self.tokens_valid and time.time() - self.created_at < expiration_time)
    def invalidate(self):
        self.tokens_valid = False
    def set_rate_limit(self, seconds: int = 60):
        self.rate_limited_until = time.time() + seconds
    def is_rate_limited(self) -> bool:
        return time.time() < self.rate_limited_until

class Conversation(JsonConversation):
    message_history: Messages = []
    def __init__(self, model: str):
        self.model = model
        self.message_history = []
        self._auth_data: Dict[str, AuthData] = {}
    def get_auth_for_model(self, model: str, provider) -> AuthData:
        if model not in self._auth_data:
            self._auth_data[model] = AuthData()
        return self._auth_data[model]
    def get_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "message_history": self.message_history}
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        return state
    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

class PuterJS():
    label = "Puter.js"
    url = "https://docs.puter.com/playground"
    api_endpoint = "https://api.puter.com/drivers/call"
    working = True
    needs_auth = False
    supports_stream = False
    supports_system_message = True
    supports_message_history = True
    default_model = 'gpt-4o'
    default_vision_model = default_model
    openai_models = [default_vision_model]
    claude_models = []
    mistral_models = []
    xai_models = []
    deepseek_models = []
    gemini_models = []
    openrouter_models = []
    vision_models = [*openai_models]
    models = vision_models
    TOKEN_EXPIRATION = 30 * 60
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    RATE_LIMIT_DELAY = 60
    Conversation = Conversation
    AuthData = AuthData
    _shared_auth_data = {}

    @staticmethod
    def get_driver_for_model(model: str) -> str:
        if model in PuterJS.openai_models:
            return "openai-completion"
        elif model in PuterJS.claude_models:
            return "claude"
        elif model in PuterJS.mistral_models:
            return "mistral"
        elif model in PuterJS.xai_models:
            return "xai"
        elif model in PuterJS.deepseek_models:
            return "deepseek"
        elif model in PuterJS.gemini_models:
            return "gemini"
        elif model in PuterJS.openrouter_models:
            return "openrouter"
        else:
            return "openai-completion"

    @staticmethod
    def format_messages_with_images(messages: Messages, media: MediaListType = None) -> Messages:
        if not media:
            return messages

        formatted_messages = messages.copy()

        for i in range(len(formatted_messages) - 1, -1, -1):
            if formatted_messages[i]["role"] == "user":
                user_msg = formatted_messages[i]

                if isinstance(user_msg["content"], str):
                    text_content = user_msg["content"]
                    user_msg["content"] = [{"type": "text", "text": text_content}]
                elif not isinstance(user_msg["content"], list):
                    user_msg["content"] = []

                for image_data, image_name in media:
                    if isinstance(image_data, str) and (image_data.startswith("http://") or image_data.startswith("https://")):
                        user_msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        })
                    else:
                        image_uri = to_data_uri(image_data)
                        user_msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": image_uri}
                        })

                formatted_messages[i] = user_msg
                break

        return formatted_messages

    @classmethod
    def _create_temporary_account(cls, session: Session, proxy: str = None) -> Dict[str, str]:
        signup_headers = {
            "Content-Type": "application/json",
            "host": "puter.com",
            "connection": "keep-alive",
            "sec-ch-ua-platform": "macOS",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "accept": "*/*",
            "origin": "https://puter.com",
            "sec-fetch-site": "same-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://puter.com/",
            "accept-encoding": "gzip",
            "accept-language": "en-US,en;q=0.9"
        }

        signup_data = {
            "is_temp": True,
            "client_id": str(uuid.uuid4())
        }

        for attempt in range(cls.MAX_RETRIES):
            try:
                response = session.post(
                    "https://puter.com/signup",
                    headers=signup_headers,
                    json=signup_data,
                    proxies={"https": proxy} if proxy else None,
                    timeout=30
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', cls.RATE_LIMIT_DELAY))
                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(retry_after)
                        continue
                    else:
                        raise RateLimitError(f"Rate limited by Puter.js API. Try again after {retry_after} seconds.")

                if response.status_code != 200:
                    error_text = response.text
                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(cls.RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        raise Exception(f"Failed to create temporary account. Status: {response.status_code}, Details: {error_text}")

                try:
                    return response.json()
                except Exception as e:
                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(cls.RETRY_DELAY)
                        continue
                    else:
                        raise Exception(f"Failed to parse signup response as JSON: {e}")

            except Exception as e:
                if attempt < cls.MAX_RETRIES - 1:
                    time.sleep(cls.RETRY_DELAY * (2 ** attempt))
                    continue
                else:
                    raise e

        raise Exception("Failed to create temporary account after multiple retries")

    @classmethod
    def _get_app_token(cls, session: Session, auth_token: str, proxy: str = None) -> Dict[str, str]:
        app_token_headers = {
            "host": "api.puter.com",
            "connection": "keep-alive",
            "authorization": f"Bearer {auth_token}",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "accept": "*/*",
            "origin": "https://puter.com",
            "sec-fetch-site": "same-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://puter.com/",
            "accept-encoding": "gzip",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json"
        }

        origins = ["http://docs.puter.com", "https://docs.puter.com", "https://puter.com"]
        app_token_data = {"origin": random.choice(origins)}

        for attempt in range(cls.MAX_RETRIES):
            try:
                response = session.post(
                    "https://api.puter.com/auth/get-user-app-token",
                    headers=app_token_headers,
                    json=app_token_data,
                    proxies={"https": proxy} if proxy else None,
                    timeout=30
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', cls.RATE_LIMIT_DELAY))
                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(retry_after)
                        continue
                    else:
                        raise RateLimitError(f"Rate limited by Puter.js API. Try again after {retry_after} seconds.")

                if response.status_code != 200:
                    error_text = response.text
                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(cls.RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        raise Exception(f"Failed to get app token. Status: {response.status_code}, Details: {error_text}")

                try:
                    return response.json()
                except Exception as e:
                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(cls.RETRY_DELAY)
                        continue
                    else:
                        raise Exception(f"Failed to parse app token response as JSON: {e}")

            except Exception as e:
                if attempt < cls.MAX_RETRIES - 1:
                    time.sleep(cls.RETRY_DELAY * (2 ** attempt))
                    continue
                else:
                    raise e

        raise Exception("Failed to get app token after multiple retries")

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""

        if not model:
            return cls.default_model

        # Check if the model exists directly in our models list
        if model in cls.models:
            return model

        # Check if there's an alias for this model
        if model in cls.model_aliases:
            alias = cls.model_aliases[model]
            # If the alias is a list, randomly select one of the options
            if isinstance(alias, list):
                selected_model = random.choice(alias)
                return selected_model
            return alias

        raise ModelNotFoundError(f"Model {model} not found")

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = None,  # Ignored in synchronous version
        conversation: Optional[JsonConversation] = None,
        return_conversation: bool = False,
        media: MediaListType = None,  # Add media parameter for images
        **kwargs
    ) -> Result:
        model = cls.get_model(model)

        # Check if we need to use a vision model
        has_images = False
        if media is not None and len(media) > 0:
            has_images = True
            # If images are present and model doesn't support vision, switch to default vision model
            if model not in cls.vision_models:
                model = cls.default_vision_model

        # Check for image URLs in messages
        if not has_images:
            for msg in messages:
                if msg["role"] == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "image_url":
                                has_images = True
                                if model not in cls.vision_models:
                                    model = cls.default_vision_model
                                break
                    elif isinstance(content, str):
                        # Check for URLs in the text
                        urls = re.findall(r'https?://\S+\.(jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)
                        if urls:
                            has_images = True
                            if model not in cls.vision_models:
                                model = cls.default_vision_model
                            break

        # Check if the conversation is of the correct type
        if conversation is not None and not isinstance(conversation, cls.Conversation):
            # Convert generic JsonConversation to our specific Conversation class
            new_conversation = cls.Conversation(model)
            new_conversation.message_history = conversation.message_history.copy() if hasattr(conversation, 'message_history') else messages.copy()
            conversation = new_conversation

        # Initialize or update conversation
        if conversation is None:
            conversation = cls.Conversation(model)
            # Format messages with images if needed
            if has_images and media:
                conversation.message_history = cls.format_messages_with_images(messages, media)
            else:
                conversation.message_history = messages.copy()
        else:
            # Update message history with new messages
            if has_images and media:
                formatted_messages = cls.format_messages_with_images(messages, media)
                for msg in formatted_messages:
                    if msg not in conversation.message_history:
                        conversation.message_history.append(msg)
            else:
                for msg in messages:
                    if msg not in conversation.message_history:
                        conversation.message_history.append(msg)

        # Get the authentication data for this specific model
        auth_data = conversation.get_auth_for_model(model, cls)

        # Check if we can use shared auth data
        if model in cls._shared_auth_data and cls._shared_auth_data[model].is_valid(cls.TOKEN_EXPIRATION):
            # Copy shared auth data to conversation
            shared_auth = cls._shared_auth_data[model]
            auth_data.auth_token = shared_auth.auth_token
            auth_data.app_token = shared_auth.app_token
            auth_data.created_at = shared_auth.created_at
            auth_data.tokens_valid = shared_auth.tokens_valid

        # Check if rate limited
        if auth_data.is_rate_limited():
            wait_time = auth_data.rate_limited_until - time.time()
            if wait_time > 0:
                yield f"Rate limited. Please try again in {int(wait_time)} seconds."
                return

        with Session() as session:
            # Step 1: Create a temporary account (if needed)
            if not auth_data.is_valid(cls.TOKEN_EXPIRATION):
                try:
                    # Try to authenticate
                    signup_data = cls._create_temporary_account(session, proxy)
                    auth_data.auth_token = signup_data.get("token")

                    if not auth_data.auth_token:
                        yield f"Error: No auth token in response for model {model}"
                        return

                    # Get app token
                    app_token_data = cls._get_app_token(session, auth_data.auth_token, proxy)
                    auth_data.app_token = app_token_data.get("token")

                    if not auth_data.app_token:
                        yield f"Error: No app token in response for model {model}"
                        return

                    # Mark tokens as valid
                    auth_data.created_at = time.time()
                    auth_data.tokens_valid = True

                    # Update shared auth data
                    cls._shared_auth_data[model] = auth_data

                except RateLimitError as e:
                    # Set rate limit and inform user
                    auth_data.set_rate_limit(cls.RATE_LIMIT_DELAY)
                    yield str(e)
                    return
                except Exception as e:
                    yield f"Error during authentication for model {model}: {str(e)}"
                    return

            # Step 3: Make the chat request with proper image handling
            try:
                chat_headers = {
                    "host": "api.puter.com",
                    "connection": "keep-alive",
                    "authorization": f"Bearer {auth_data.app_token}",
                    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
                    "content-type": "application/json;charset=UTF-8",
                    "accept": "*/*",
                    "origin": "http://docs.puter.com",
                    "sec-fetch-site": "cross-site",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-dest": "empty",
                    "referer": "http://docs.puter.com/",
                    "accept-encoding": "gzip",
                    "accept-language": "en-US,en;q=0.9"
                }

                driver = cls.get_driver_for_model(model)

                processed_messages = conversation.message_history

                if has_images and not any(isinstance(msg.get("content"), list) for msg in processed_messages):
                    for i, msg in enumerate(processed_messages):
                        if msg["role"] == "user":
                            if media and len(media) > 0:
                                processed_messages = cls.format_messages_with_images([msg], media)
                            else:
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    urls = re.findall(r'https?://\S+\.(jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)
                                    if urls:
                                        text_parts = []
                                        image_urls = []

                                        words = content.split()
                                        for word in words:
                                            if re.match(r'https?://\S+\.(jpg|jpeg|png|gif|webp)', word, re.IGNORECASE):
                                                image_urls.append(word)
                                            else:
                                                text_parts.append(word)

                                        formatted_content = []
                                        if text_parts:
                                            formatted_content.append({
                                                "type": "text",
                                                "text": " ".join(text_parts)
                                            })

                                        for url in image_urls:
                                            formatted_content.append({
                                                "type": "image_url",
                                                "image_url": {"url": url}
                                            })

                                        processed_messages[i]["content"] = formatted_content
                            break

                chat_data = {
                    "interface": "puter-chat-completion",
                    "driver": driver,
                    "test_mode": False,
                    "method": "complete",
                    "args": {
                        "messages": processed_messages,
                        "model": model,
                        "stream": False,
                        "max_tokens": kwargs.get("max_tokens", 4096)
                    }
                }

                for key, value in kwargs.items():
                    if key not in ["messages", "model", "stream", "max_tokens"]:
                        chat_data["args"][key] = value

                for attempt in range(cls.MAX_RETRIES):
                    try:
                        response = session.post(
                            cls.api_endpoint,
                            headers=chat_headers,
                            json=chat_data,
                            proxies={"https": proxy} if proxy else None,
                            timeout=120
                        )

                        if response.status_code == 429:
                            retry_after = int(response.headers.get('Retry-After', cls.RATE_LIMIT_DELAY))
                            auth_data.set_rate_limit(retry_after)
                            if attempt < cls.MAX_RETRIES - 1:
                                time.sleep(min(retry_after, 10))
                                continue
                            else:
                                raise RateLimitError(f"Rate limited by Puter.js API. Try again after {retry_after} seconds.")

                        if response.status_code in [401, 403]:
                            error_text = response.text
                            auth_data.invalidate()
                            if attempt < cls.MAX_RETRIES - 1:
                                signup_data = cls._create_temporary_account(session, proxy)
                                auth_data.auth_token = signup_data.get("token")

                                app_token_data = cls._get_app_token(session, auth_data.auth_token, proxy)
                                auth_data.app_token = app_token_data.get("token")

                                auth_data.created_at = time.time()
                                auth_data.tokens_valid = True

                                cls._shared_auth_data[model] = auth_data

                                chat_headers["authorization"] = f"Bearer {auth_data.app_token}"

                                continue
                            else:
                                raise Exception(f"Authentication failed after {cls.MAX_RETRIES} attempts: {error_text}")

                        if response.status_code != 200:
                            error_text = response.text
                            if attempt < cls.MAX_RETRIES - 1:
                                time.sleep(cls.RETRY_DELAY * (attempt + 1))
                                continue
                            else:
                                raise Exception(f"Chat request failed. Status: {response.status_code}, Details: {error_text}")

                        try:
                            response_json = response.json()
                        except Exception as e:
                            error_text = response.text
                            if attempt < cls.MAX_RETRIES - 1:
                                time.sleep(cls.RETRY_DELAY)
                                continue
                            else:
                                raise Exception(f"Failed to parse chat response as JSON: {error_text}")

                        if response_json.get("success") is True:
                            content = response_json.get("result", {}).get("message", {}).get("content", "")

                            if content:
                                conversation.message_history.append({
                                    "role": "assistant",
                                    "content": content
                                })

                            yield content

                            if return_conversation:
                                yield conversation

                            return
                        else:
                            error_msg = response_json.get("error", {}).get("message", "Unknown error")

                            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                                auth_data.set_rate_limit()
                                if attempt < cls.MAX_RETRIES - 1:
                                    time.sleep(cls.RETRY_DELAY)
                                    continue
                                else:
                                    raise RateLimitError(f"Rate limited: {error_msg}")

                            if "auth" in error_msg.lower() or "token" in error_msg.lower():
                                auth_data.invalidate()
                                if attempt < cls.MAX_RETRIES - 1:
                                    signup_data = cls._create_temporary_account(session, proxy)
                                    auth_data.auth_token = signup_data.get("token")

                                    app_token_data = cls._get_app_token(session, auth_data.auth_token, proxy)
                                    auth_data.app_token = app_token_data.get("token")

                                    auth_data.created_at = time.time()
                                    auth_data.tokens_valid = True

                                    chat_headers["authorization"] = f"Bearer {auth_data.app_token}"

                                    cls._shared_auth_data[model] = auth_data

                                    continue

                            if attempt < cls.MAX_RETRIES - 1:
                                time.sleep(cls.RETRY_DELAY)
                                continue
                            else:
                                yield f"Error: {error_msg}"
                                return

                    except RateLimitError as e:
                        auth_data.set_rate_limit(cls.RATE_LIMIT_DELAY)
                        if attempt < cls.MAX_RETRIES - 1:
                            time.sleep(min(cls.RATE_LIMIT_DELAY, 10))
                            continue
                        else:
                            yield str(e)
                            return

                    except Exception as e:
                        if "token" in str(e).lower() or "auth" in str(e).lower():
                            auth_data.invalidate()

                        if attempt < cls.MAX_RETRIES - 1:
                            time.sleep(cls.RETRY_DELAY * (attempt + 1))
                            continue
                        else:
                            yield f"Error: {str(e)}"
                            return

            except Exception as e:
                if "token" in str(e).lower() or "auth" in str(e).lower():
                    auth_data.invalidate()

                yield f"Error: {str(e)}"
                return
if __name__ == "__main__":
    # Example usage
    puter = PuterJS()
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    result = puter.create_completion(model="gpt-4o", messages=messages)
    for res in result:
        print(res)