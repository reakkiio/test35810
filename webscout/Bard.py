# -*- coding: utf-8 -*-
#########################################
# Code Modified to use curl_cffi
#########################################
import asyncio
import json
import os
import random
import re
import string
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional

# Use curl_cffi for requests
from curl_cffi import CurlError
from curl_cffi.requests import AsyncSession
# Import common request exceptions (curl_cffi often wraps these)
from requests.exceptions import RequestException, Timeout, HTTPError

# For image models using validation. Adjust based on organization internal pydantic.
# Updated import for Pydantic V2
from pydantic import BaseModel, field_validator

# Rich is retained for logging within image methods.
from rich.console import Console
from rich.markdown import Markdown

console = Console()

#########################################
# New Enums and functions for endpoints,
# headers, models, file upload and images.
#########################################

class Endpoint(Enum):
    INIT = "https://gemini.google.com/app"
    GENERATE = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
    ROTATE_COOKIES = "https://accounts.google.com/RotateCookies"
    UPLOAD = "https://content-push.googleapis.com/upload"

class Headers(Enum):
    GEMINI = {
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "Host": "gemini.google.com",
        "Origin": "https://gemini.google.com",
        "Referer": "https://gemini.google.com/",
        # User-Agent will be handled by curl_cffi impersonate
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Same-Domain": "1",
    }
    ROTATE_COOKIES = {
        "Content-Type": "application/json",
    }
    UPLOAD = {"Push-ID": "feeds/mcudyrk2a4khkz"}

class Model(Enum):
    # Model definitions remain the same
    UNSPECIFIED = ("unspecified", {}, False)
    G_2_0_FLASH = (
        "gemini-2.0-flash",
        {"x-goog-ext-525001261-jspb": '[null,null,null,null,"f299729663a2343f"]'},
        False,
    )
    G_2_0_FLASH_THINKING = (
        "gemini-2.0-flash-thinking",
        {"x-goog-ext-525001261-jspb": '[null,null,null,null,"7ca48d02d802f20a"]'},
        False,
    )
    G_2_5_PRO = (
        "gemini-2.5-pro",
        {"x-goog-ext-525001261-jspb": '[null,null,null,null,"2525e3954d185b3c"]'},
        False,
    )
    G_2_0_EXP_ADVANCED = (
        "gemini-2.0-exp-advanced",
        {"x-goog-ext-525001261-jspb": '[null,null,null,null,"b1e46a6037e6aa9f"]'},
        True,
    )
    G_2_5_EXP_ADVANCED = (
        "gemini-2.5-exp-advanced",
        {"x-goog-ext-525001261-jspb": '[null,null,null,null,"203e6bb81620bcfe"]'},
        True,
    )
    G_2_5_FLASH = (
        "gemini-2.5-flash",
        {"x-goog-ext-525001261-jspb": '[1,null,null,null,"35609594dbe934d8"]'},
        False,
    )

    def __init__(self, name, header, advanced_only):
        self.model_name = name
        self.model_header = header
        self.advanced_only = advanced_only

    @classmethod
    def from_name(cls, name: str):
        for model in cls:
            if model.model_name == name:
                return model
        raise ValueError(
            f"Unknown model name: {name}. Available models: {', '.join([model.model_name for model in cls])}"
        )

async def upload_file(
    file: Union[bytes, str, Path],
    proxy: Optional[Union[str, Dict[str, str]]] = None,
    impersonate: str = "chrome110" # Added impersonate
) -> str:
    """
    Upload a file to Google's server and return its identifier using curl_cffi.

    Parameters:
        file: bytes | str | Path
            File data in bytes, or path to the file to be uploaded.
        proxy: str | Dict, optional
            Proxy URL or dictionary.
        impersonate: str, optional
            Browser profile for curl_cffi to impersonate. Defaults to "chrome110".

    Returns:
        str: Identifier of the uploaded file.
    Raises:
        HTTPError: If the upload request failed.
        RequestException: For other network-related errors.
    """
    if not isinstance(file, bytes):
        file_path = Path(file)
        if not file_path.is_file():
             raise FileNotFoundError(f"File not found at path: {file}")
        with open(file_path, "rb") as f:
            file_content = f.read()
    else:
        file_content = file

    # Prepare proxy dictionary for curl_cffi
    proxies_dict = None
    if isinstance(proxy, str):
        proxies_dict = {"http": proxy, "https": proxy} # curl_cffi uses http/https keys
    elif isinstance(proxy, dict):
        proxies_dict = proxy # Assume it's already in the correct format

    try:
        # Use AsyncSession from curl_cffi
        async with AsyncSession(
            proxies=proxies_dict,
            impersonate=impersonate,
            headers=Headers.UPLOAD.value, # Pass headers directly
            # follow_redirects=True is default in curl_cffi
        ) as client:
            response = await client.post(
                url=Endpoint.UPLOAD.value,
                # headers=Headers.UPLOAD.value, # Headers passed in session
                files={"file": file_content},
                # follow_redirects=True, # Default
            )
            response.raise_for_status() # Raises HTTPError for bad responses
            return response.text
    except HTTPError as e:
        console.log(f"[red]HTTP error during file upload: {e.response.status_code} {e}[/red]")
        raise # Re-raise HTTPError
    except (RequestException, CurlError) as e:
        console.log(f"[red]Network error during file upload: {e}[/red]")
        raise # Re-raise other request errors

#########################################
# Cookie loading and Chatbot classes
#########################################

def load_cookies(cookie_path: str) -> Tuple[str, str]:
    """Loads cookies from the provided JSON file."""
    try:
        with open(cookie_path, 'r', encoding='utf-8') as file: # Added encoding
            cookies = json.load(file)
        # Handle potential variations in cookie names (case-insensitivity)
        session_auth1 = next((item['value'] for item in cookies if item['name'].upper() == '__SECURE-1PSID'), None)
        session_auth2 = next((item['value'] for item in cookies if item['name'].upper() == '__SECURE-1PSIDTS'), None)

        if not session_auth1 or not session_auth2:
             raise StopIteration("Required cookies (__Secure-1PSID or __Secure-1PSIDTS) not found.")

        return session_auth1, session_auth2
    except FileNotFoundError:
        raise Exception(f"Cookie file not found at path: {cookie_path}")
    except json.JSONDecodeError:
        raise Exception("Invalid JSON format in the cookie file.")
    except StopIteration as e:
        raise Exception(f"{e} Check the cookie file format and content.")
    except Exception as e: # Catch other potential errors
        raise Exception(f"An unexpected error occurred while loading cookies: {e}")


class Chatbot:
    """
    Synchronous wrapper for the AsyncChatbot class.
    """
    def __init__(
        self,
        cookie_path: str,
        proxy: Optional[Union[str, Dict[str, str]]] = None, # Allow string or dict proxy
        timeout: int = 20,
        model: Model = Model.UNSPECIFIED,
        impersonate: str = "chrome110" # Added impersonate
    ):
        # Use asyncio.run() for cleaner async execution in sync context
        # Handle potential RuntimeError if an event loop is already running
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.secure_1psid, self.secure_1psidts = load_cookies(cookie_path)
        self.async_chatbot = self.loop.run_until_complete(
            AsyncChatbot.create(self.secure_1psid, self.secure_1psidts, proxy, timeout, model, impersonate) # Pass impersonate
        )

    def save_conversation(self, file_path: str, conversation_name: str):
        return self.loop.run_until_complete(
            self.async_chatbot.save_conversation(file_path, conversation_name)
        )

    def load_conversations(self, file_path: str) -> List[Dict]:
        return self.loop.run_until_complete(
            self.async_chatbot.load_conversations(file_path)
        )

    def load_conversation(self, file_path: str, conversation_name: str) -> bool:
        return self.loop.run_until_complete(
            self.async_chatbot.load_conversation(file_path, conversation_name)
        )

    def ask(self, message: str, image: Optional[Union[bytes, str, Path]] = None) -> dict: # Added image param
        # Pass image to async ask method
        return self.loop.run_until_complete(self.async_chatbot.ask(message, image=image))

class AsyncChatbot:
    """
    A class to interact with Google Gemini using curl_cffi.
    Parameters:
        secure_1psid: str
            The __Secure-1PSID cookie.
        secure_1psidts: str
            The __Secure-1PSIDTS cookie.
        proxy: Optional[Union[str, Dict[str, str]]]
            Proxy URL string or dictionary for curl_cffi.
        timeout: int
            Request timeout in seconds.
        model: Model
            Selected model for the session.
        impersonate: str
            Browser profile for curl_cffi to impersonate.
    """
    __slots__ = [
        "headers",
        "_reqid",
        "SNlM0e",
        "conversation_id",
        "response_id",
        "choice_id",
        "proxy", # Store the original proxy config
        "proxies_dict", # Store the curl_cffi-compatible proxy dict
        "secure_1psidts",
        "secure_1psid",
        "session",
        "timeout",
        "model",
        "impersonate", # Store impersonate setting
    ]

    def __init__(
        self,
        secure_1psid: str,
        secure_1psidts: str,
        proxy: Optional[Union[str, Dict[str, str]]] = None, # Allow string or dict proxy
        timeout: int = 20,
        model: Model = Model.UNSPECIFIED,
        impersonate: str = "chrome110", # Added impersonate
    ):
        headers = Headers.GEMINI.value.copy()
        if model != Model.UNSPECIFIED:
            headers.update(model.model_header)
        self._reqid = int("".join(random.choices(string.digits, k=7))) # Increased length for less collision chance
        self.proxy = proxy # Store original proxy setting
        self.impersonate = impersonate # Store impersonate setting

        # Prepare proxy dictionary for curl_cffi
        self.proxies_dict = None
        if isinstance(proxy, str):
            self.proxies_dict = {"http": proxy, "https": proxy} # curl_cffi uses http/https keys
        elif isinstance(proxy, dict):
            self.proxies_dict = proxy # Assume it's already in the correct format

        self.conversation_id = ""
        self.response_id = ""
        self.choice_id = ""
        self.secure_1psid = secure_1psid
        self.secure_1psidts = secure_1psidts

        # Initialize curl_cffi AsyncSession
        self.session = AsyncSession(
            headers=headers,
            cookies={"__Secure-1PSID": secure_1psid, "__Secure-1PSIDTS": secure_1psidts},
            proxies=self.proxies_dict,
            timeout=timeout,
            impersonate=self.impersonate,
            # verify=True, # Default in curl_cffi
            # http2=True, # Implicitly handled by curl_cffi if possible
        )
        # No need to set proxies/headers/cookies again, done in constructor

        self.timeout = timeout # Store timeout for potential direct use in requests
        self.model = model
        self.SNlM0e = None # Initialize SNlM0e

    @classmethod
    async def create(
        cls,
        secure_1psid: str,
        secure_1psidts: str,
        proxy: Optional[Union[str, Dict[str, str]]] = None, # Allow string or dict proxy
        timeout: int = 20,
        model: Model = Model.UNSPECIFIED,
        impersonate: str = "chrome110", # Added impersonate
    ) -> "AsyncChatbot":
        """
        Factory method to create and initialize an AsyncChatbot instance.
        Fetches the necessary SNlM0e value asynchronously.
        """
        instance = cls(secure_1psid, secure_1psidts, proxy, timeout, model, impersonate) # Pass impersonate
        try:
            instance.SNlM0e = await instance.__get_snlm0e()
        except Exception as e:
             # Log the error and re-raise or handle appropriately
             console.log(f"[red]Error during AsyncChatbot initialization (__get_snlm0e): {e}[/red]", style="bold red")
             # Optionally close the session if initialization fails critically
             await instance.session.close() # Use close() for AsyncSession
             raise # Re-raise the exception to signal failure
        return instance

    async def save_conversation(self, file_path: str, conversation_name: str) -> None:
        # Logic remains the same
        conversations = await self.load_conversations(file_path)
        conversation_data = {
            "conversation_name": conversation_name,
            "_reqid": self._reqid,
            "conversation_id": self.conversation_id,
            "response_id": self.response_id,
            "choice_id": self.choice_id,
            "SNlM0e": self.SNlM0e,
            "model_name": self.model.model_name, # Save the model used
            "timestamp": datetime.now().isoformat(), # Add timestamp
        }

        found = False
        for i, conv in enumerate(conversations):
            if conv.get("conversation_name") == conversation_name:
                conversations[i] = conversation_data # Update existing
                found = True
                break
        if not found:
            conversations.append(conversation_data) # Add new

        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, indent=4, ensure_ascii=False)
        except IOError as e:
            console.log(f"[red]Error saving conversation to {file_path}: {e}[/red]")
            raise

    async def load_conversations(self, file_path: str) -> List[Dict]:
        # Logic remains the same
        if not os.path.isfile(file_path):
            return []
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console.log(f"[red]Error loading conversations from {file_path}: {e}[/red]")
            return []

    async def load_conversation(self, file_path: str, conversation_name: str) -> bool:
        # Logic remains the same, but update headers on the session
        conversations = await self.load_conversations(file_path)
        for conversation in conversations:
            if conversation.get("conversation_name") == conversation_name:
                try:
                    self._reqid = conversation["_reqid"]
                    self.conversation_id = conversation["conversation_id"]
                    self.response_id = conversation["response_id"]
                    self.choice_id = conversation["choice_id"]
                    self.SNlM0e = conversation["SNlM0e"]
                    if "model_name" in conversation:
                         try:
                              self.model = Model.from_name(conversation["model_name"])
                              # Update headers in the session if model changed
                              self.session.headers.update(self.model.model_header)
                         except ValueError as e:
                              console.log(f"[yellow]Warning: Model '{conversation['model_name']}' from saved conversation not found. Using current model '{self.model.model_name}'. Error: {e}[/yellow]")

                    console.log(f"Loaded conversation '{conversation_name}'")
                    return True
                except KeyError as e:
                    console.log(f"[red]Error loading conversation '{conversation_name}': Missing key {e}[/red]")
                    return False
        console.log(f"[yellow]Conversation '{conversation_name}' not found in {file_path}[/yellow]")
        return False

    async def __get_snlm0e(self):
        """Fetches the SNlM0e value required for API requests using curl_cffi."""
        if not self.secure_1psid or not self.secure_1psidts:
             raise ValueError("Both __Secure-1PSID and __Secure-1PSIDTS cookies are required.")

        try:
            # Use the session's get method
            resp = await self.session.get(
                Endpoint.INIT.value,
                timeout=self.timeout, # Timeout is already set in session, but can override
                # follow_redirects=True # Default in curl_cffi
            )
            resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Regex logic remains the same
            snlm0e_match = re.search(r'["\']SNlM0e["\']\s*:\s*["\'](.*?)["\']', resp.text)
            if not snlm0e_match:
                error_message = "SNlM0e value not found in response."
                if "Sign in to continue" in resp.text or "accounts.google.com" in str(resp.url):
                     error_message += " Cookies might be invalid or expired. Please update them."
                elif resp.status_code == 429:
                     error_message += " Rate limit likely exceeded."
                else:
                     error_message += f" Response status: {resp.status_code}. Check cookie validity and network."
                raise ValueError(error_message)

            return snlm0e_match.group(1)

        except Timeout as e: # Catch requests.exceptions.Timeout
            raise TimeoutError(f"Request timed out while fetching SNlM0e: {e}") from e
        except (RequestException, CurlError) as e: # Catch general request errors and Curl specific errors
            raise ConnectionError(f"Network error while fetching SNlM0e: {e}") from e
        except HTTPError as e: # Catch requests.exceptions.HTTPError
             if e.response.status_code == 401 or e.response.status_code == 403:
                  raise PermissionError(f"Authentication failed (status {e.response.status_code}). Check cookies. {e}") from e
             else:
                  raise Exception(f"HTTP error {e.response.status_code} while fetching SNlM0e: {e}") from e


    async def ask(self, message: str, image: Optional[Union[bytes, str, Path]] = None) -> dict:
        """
        Sends a message to Google Gemini and returns the response using curl_cffi.

        Parameters:
            message: str
                The message to send.
            image: Optional[Union[bytes, str, Path]]
                Optional image data (bytes) or path to an image file to include.

        Returns:
            dict: A dictionary containing the response content and metadata.
        """
        if self.SNlM0e is None:
             raise RuntimeError("AsyncChatbot not properly initialized. Call AsyncChatbot.create()")

        params = {
            "bl": "boq_assistant-bard-web-server_20240625.13_p0", # Example, might need updates
            "_reqid": str(self._reqid),
            "rt": "c",
        }

        image_upload_id = None
        if image:
            try:
                # Pass proxy and impersonate settings to upload_file
                image_upload_id = await upload_file(image, proxy=self.proxies_dict, impersonate=self.impersonate)
                console.log(f"Image uploaded successfully. ID: {image_upload_id}")
            except Exception as e:
                console.log(f"[red]Error uploading image: {e}[/red]")
                return {"content": f"Error uploading image: {e}", "error": True}

        # Structure logic remains the same
        message_struct = [
            [message],
            None,
            [self.conversation_id, self.response_id, self.choice_id],
        ]
        if image_upload_id:
             message_struct = [
                 [message],
                 [[[image_upload_id, 1]]],
                 [self.conversation_id, self.response_id, self.choice_id],
             ]

        data = {
            "f.req": json.dumps([None, json.dumps(message_struct, ensure_ascii=False)], ensure_ascii=False),
            "at": self.SNlM0e,
        }

        try:
            # Use session.post
            resp = await self.session.post(
                Endpoint.GENERATE.value,
                params=params,
                data=data, # curl_cffi uses data for form-encoded
                timeout=self.timeout,
            )
            resp.raise_for_status() # Check for HTTP errors

            # Response processing logic remains the same
            lines = resp.text.splitlines()
            if len(lines) < 4:
                 raise ValueError(f"Unexpected response format from Gemini API. Status: {resp.status_code}. Content: {resp.text[:200]}...")

            chat_data_line = lines[3]
            if chat_data_line.startswith(")]}'"):
                 chat_data_line = chat_data_line.split('\n', 1)[-1].strip()

            chat_data = json.loads(chat_data_line)[0][2]

            if not chat_data:
                return {"content": f"Gemini returned an empty response structure. Status: {resp.status_code}."}

            json_chat_data = json.loads(chat_data)

            # Extraction logic remains the same
            content = json_chat_data[4][0][1][0] if len(json_chat_data) > 4 and len(json_chat_data[4]) > 0 and len(json_chat_data[4][0]) > 1 and len(json_chat_data[4][0][1]) > 0 else ""
            conversation_id = json_chat_data[1][0] if len(json_chat_data) > 1 and len(json_chat_data[1]) > 0 else self.conversation_id
            response_id = json_chat_data[1][1] if len(json_chat_data) > 1 and len(json_chat_data[1]) > 1 else self.response_id
            factualityQueries = json_chat_data[3] if len(json_chat_data) > 3 else None
            textQuery = json_chat_data[2][0] if len(json_chat_data) > 2 and json_chat_data[2] else ""
            choices = [{"id": i[0], "content": i[1]} for i in json_chat_data[4]] if len(json_chat_data) > 4 else []
            choice_id = choices[0]["id"] if choices else self.choice_id

            images = []
            if len(json_chat_data) > 4 and len(json_chat_data[4]) > 0 and len(json_chat_data[4][0]) > 4 and json_chat_data[4][0][4]:
                for img_data in json_chat_data[4][0][4]:
                    try:
                        img_url = img_data[0][0][0]
                        img_alt = img_data[2] if len(img_data) > 2 else ""
                        img_title = img_data[1] if len(img_data) > 1 else "[Image]"
                        images.append({"url": img_url, "alt": img_alt, "title": img_title})
                    except (IndexError, TypeError):
                        console.log("[yellow]Warning: Could not parse image data structure.[/yellow]")
                        continue

            results = {
                "content": content,
                "conversation_id": conversation_id,
                "response_id": response_id,
                "factualityQueries": factualityQueries,
                "textQuery": textQuery,
                "choices": choices,
                "images": images,
                "error": False,
            }

            # Update state
            self.conversation_id = conversation_id
            self.response_id = response_id
            self.choice_id = choice_id
            self._reqid += random.randint(1000, 9000)

            return results

        # Update exception handling
        except (IndexError, json.JSONDecodeError, TypeError) as e:
            console.log(f"[red]Error parsing Gemini response: {e}[/red]")
            return {"content": f"Error parsing Gemini response: {e}. Response: {resp.text[:200]}...", "error": True}
        except Timeout as e: # Catch requests.exceptions.Timeout
            console.log(f"[red]Request timed out: {e}[/red]")
            return {"content": f"Request timed out: {e}", "error": True}
        except (RequestException, CurlError) as e: # Catch general request/curl errors
            console.log(f"[red]Network error: {e}[/red]")
            return {"content": f"Network error: {e}", "error": True}
        except HTTPError as e: # Catch requests.exceptions.HTTPError
             console.log(f"[red]HTTP error {e.response.status_code}: {e}[/red]")
             return {"content": f"HTTP error {e.response.status_code}: {e}", "error": True}
        except Exception as e:
             console.log(f"[red]An unexpected error occurred during ask: {e}[/red]", style="bold red")
             return {"content": f"An unexpected error occurred: {e}", "error": True}


#########################################
# New Image classes
#########################################

class Image(BaseModel):
    """
    A single image object returned from Gemini.
    Parameters:
        url: str
            URL of the image.
        title: str, optional
            Title of the image (default: "[Image]").
        alt: str, optional
            Optional description.
        proxy: Optional[Union[str, Dict[str, str]]] = None # Allow string or dict proxy
            Proxy used when saving the image.
        impersonate: str = "chrome110" # Added impersonate for saving
            Browser profile for curl_cffi to impersonate.
    """
    url: str
    title: str = "[Image]"
    alt: str = ""
    proxy: Optional[Union[str, Dict[str, str]]] = None
    impersonate: str = "chrome110" # Default impersonation for saving

    def __str__(self):
        return f"{self.title}({self.url}) - {self.alt}"

    def __repr__(self):
        short_url = self.url if len(self.url) <= 50 else self.url[:20] + "..." + self.url[-20:]
        short_alt = self.alt[:30] + "..." if len(self.alt) > 30 else self.alt
        return f"Image(title='{self.title}', url='{short_url}', alt='{short_alt}')"

    async def save(
        self,
        path: str = "temp",
        filename: Optional[str] = None,
        cookies: Optional[dict] = None,
        verbose: bool = False,
        skip_invalid_filename: bool = False,
    ) -> Optional[str]:
        """
        Save the image to disk using curl_cffi.
        Parameters:
            path: str, optional
                Directory to save the image (default "./temp").
            filename: str, optional
                Filename to use; if not provided, inferred from URL.
            cookies: dict, optional
                Cookies used for the image request.
            verbose: bool, optional
                If True, outputs status messages (default False).
            skip_invalid_filename: bool, optional
                If True, skips saving if the filename is invalid.
        Returns:
            Absolute path of the saved image if successful; None if skipped.
        Raises:
            HTTPError if the network request fails.
            RequestException/CurlError for other network errors.
            IOError if file writing fails.
        """
        # Filename generation logic remains the same
        if not filename:
             try:
                  # Use httpx.URL temporarily just for parsing, or implement manually
                  # Let's use basic parsing to avoid httpx dependency here
                  from urllib.parse import urlparse, unquote
                  parsed_url = urlparse(self.url)
                  base_filename = os.path.basename(unquote(parsed_url.path))
                  safe_filename = re.sub(r'[<>:"/\\|?*]', '_', base_filename)
                  filename = safe_filename if safe_filename else f"image_{random.randint(1000, 9999)}.jpg"
             except Exception:
                  filename = f"image_{random.randint(1000, 9999)}.jpg"

        try:
            _ = Path(filename)
            max_len = 255
            if len(filename) > max_len:
                 name, ext = os.path.splitext(filename)
                 filename = name[:max_len - len(ext) -1] + ext
        except (OSError, ValueError):
            if verbose: console.log(f"[yellow]Invalid filename generated: {filename}[/yellow]")
            if skip_invalid_filename:
                if verbose: console.log("[yellow]Skipping save due to invalid filename.[/yellow]")
                return None
            filename = f"image_{random.randint(1000, 9999)}.jpg"
            if verbose: console.log(f"[yellow]Using fallback filename: {filename}[/yellow]")

        # Prepare proxy dictionary for curl_cffi
        proxies_dict = None
        if isinstance(self.proxy, str):
            proxies_dict = {"http": self.proxy, "https": self.proxy}
        elif isinstance(self.proxy, dict):
            proxies_dict = self.proxy

        try:
            # Use AsyncSession from curl_cffi
            async with AsyncSession(
                follow_redirects=True, # Default
                cookies=cookies,
                proxies=proxies_dict,
                impersonate=self.impersonate # Use stored impersonate setting
            ) as client:
                if verbose:
                    console.log(f"Attempting to download image from: {self.url}")
                response = await client.get(self.url)
                response.raise_for_status() # Raise HTTPError for bad responses

                content_type = response.headers.get("content-type", "").lower()
                if "image" not in content_type:
                    console.log(f"[yellow]Warning: Content type is '{content_type}', not an image. Saving anyway.[/yellow]")

                dest_path = Path(path)
                dest_path.mkdir(parents=True, exist_ok=True)
                dest = dest_path / filename

                # Use response.content which holds the bytes
                dest.write_bytes(response.content)
                if verbose:
                    console.log(f"Image saved successfully as {dest.resolve()}")
                return str(dest.resolve())

        # Update exception handling
        except HTTPError as e:
            console.log(f"[red]Error downloading image {self.url}: {e.response.status_code} {e}[/red]")
            raise
        except (RequestException, CurlError) as e:
            console.log(f"[red]Network error downloading image {self.url}: {e}[/red]")
            raise
        except IOError as e:
            console.log(f"[red]Error writing image file to {dest}: {e}[/red]")
            raise
        except Exception as e:
             console.log(f"[red]An unexpected error occurred during image save: {e}[/red]")
             raise


class WebImage(Image):
    """
    Image retrieved from web search results.
    Returned when asking Gemini to "SEND an image of [something]".
    """
    pass

class GeneratedImage(Image):
    """
    Image generated by Google's AI image generator (e.g., ImageFX).
    Parameters:
        cookies: dict[str, str]
            Cookies required for accessing the generated image URL, typically
            from the GeminiClient/Chatbot instance.
    """
    cookies: Dict[str, str]

    # Updated validator for Pydantic V2
    @field_validator("cookies")
    @classmethod
    def validate_cookies(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Ensures cookies are provided for generated images."""
        if not v or not isinstance(v, dict):
            raise ValueError("GeneratedImage requires a dictionary of cookies from the client.")
        return v

    async def save(self, **kwargs) -> Optional[str]:
        """
        Save the generated image to disk.
        Parameters:
            filename: str, optional
                Filename to use. If not provided, a default name including
                a timestamp and part of the URL is used. Generated images
                are often in .png or .jpg format.
            Additional arguments are passed to Image.save.
        Returns:
            Absolute path of the saved image if successful, None if skipped.
        """
        if "filename" not in kwargs:
             ext = ".jpg" if ".jpg" in self.url.lower() else ".png"
             url_part = self.url.split('/')[-1][:10]
             kwargs["filename"] = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{url_part}{ext}"

        # Pass the required cookies and other args (like impersonate) to the parent save method
        return await super().save(cookies=self.cookies, **kwargs)

#########################################
# Main usage demonstration
#########################################

async def main_async():
    """Asynchronous main function for demonstration."""
    cookies_file = "cookies.json"
    impersonate_profile = "chrome110" # Example browser profile

    bot = None
    try:
        bot = await AsyncChatbot.create(
            *load_cookies(cookies_file),
            model=Model.G_2_5_PRO,
            impersonate=impersonate_profile, # Pass impersonate setting
            # proxy="socks5://127.0.0.1:9050" # Example SOCKS proxy
        )
        console.log(f"[green]AsyncChatbot initialized successfully (impersonating {impersonate_profile}).[/green]")
    except FileNotFoundError:
        console.log(f"[bold red]Error: Cookie file '{cookies_file}' not found.[/bold red]")
        console.log("Please export cookies from your browser after logging into Google Gemini and save as cookies.json.")
        return
    except Exception as e:
        console.log(f"[bold red]Error initializing AsyncChatbot: {e}[/bold red]")
        return

    # --- Sample text query ---
    text_message = "Explain the concept of asynchronous programming in Python in simple terms."
    console.log(f"\n[cyan]Sending text query:[/cyan] '{text_message}'")
    try:
        response_text = await bot.ask(text_message)
        if response_text.get("error"):
             console.log(f"[red]Error in text response: {response_text.get('content')}[/red]")
        else:
             console.log("[blue]Text Response:[/blue]")
             console.print(Markdown(response_text.get("content", "No content received.")))
    except Exception as e:
        console.log(f"[red]Error during text query: {e}[/red]")

    # --- Image Generation Query ---
    image_prompt = "Generate an artistic image of a cat sitting on a crescent moon, starry night background."
    console.log(f"\n[cyan]Sending image generation query:[/cyan] '{image_prompt}'")
    try:
        response_image = await bot.ask(image_prompt)

        if response_image.get("error"):
             console.log(f"[red]Error in image response: {response_image.get('content')}[/red]")
        else:
            returned_images = response_image.get("images", [])
            if not returned_images:
                console.log("[yellow]No direct image data returned. Response content:[/yellow]")
                console.print(Markdown(response_image.get("content", "No content received.")))
            else:
                console.log(f"[green]Received {len(returned_images)} image(s).[/green]")
                for i, img_data in enumerate(returned_images):
                    console.log(f"Processing image {i+1}: URL: {img_data.get('url')}")
                    try:
                        # Pass impersonate setting when creating Image object
                        generated_img = GeneratedImage(
                            url=img_data.get('url'),
                            title=img_data.get('title', f"Generated Image {i+1}"),
                            alt=img_data.get('alt', ""),
                            cookies={"__Secure-1PSID": bot.secure_1psid, "__Secure-1PSIDTS": bot.secure_1psidts},
                            proxy=bot.proxy, # Pass proxy settings from bot
                            impersonate=bot.impersonate # Pass impersonate setting from bot
                        )
                        save_path = "downloaded_images"
                        saved_file = await generated_img.save(path=save_path, verbose=True, skip_invalid_filename=True)
                        if saved_file:
                            console.log(f"[blue]Image {i+1} saved to: {saved_file}[/blue]")
                        else:
                            console.log(f"[yellow]Image {i+1} skipped due to filename issue.[/yellow]")
                    except Exception as img_e:
                        console.log(f"[red]Error saving image {i+1}: {img_e}[/red]")

    except Exception as e:
        console.log(f"[red]Error during image generation query: {e}[/red]")

    # --- Image Understanding Query ---
    local_image_path = "path/to/your/local/image.jpg" # <--- CHANGE THIS PATH
    image_understanding_prompt = "Describe what you see in this image."

    if Path(local_image_path).is_file():
        console.log(f"\n[cyan]Sending image understanding query with image:[/cyan] '{local_image_path}'")
        console.log(f"[cyan]Prompt:[/cyan] '{image_understanding_prompt}'")
        try:
            response_understanding = await bot.ask(image_understanding_prompt, image=local_image_path)
            if response_understanding.get("error"):
                 console.log(f"[red]Error in image understanding response: {response_understanding.get('content')}[/red]")
            else:
                 console.log("[blue]Image Understanding Response:[/blue]")
                 console.print(Markdown(response_understanding.get("content", "No content received.")))
        except Exception as e:
            console.log(f"[red]Error during image understanding query: {e}[/red]")
    else:
        console.log(f"\n[yellow]Skipping image understanding query: File not found at '{local_image_path}'.[/yellow]")
        console.log("[yellow]Please update 'local_image_path' in the script to test this feature.[/yellow]")


    # --- Save/Load Conversation (logic remains the same) ---
    conversation_file = "conversations.json"
    conversation_name = f"Demo Conversation - {datetime.now().strftime('%Y%m%d_%H%M')}"
    console.log(f"\n[cyan]Saving conversation as:[/cyan] '{conversation_name}' to '{conversation_file}'")
    try:
        await bot.save_conversation(conversation_file, conversation_name)
        console.log(f"[green]Conversation saved successfully.[/green]")
    except Exception as e:
        console.log(f"[red]Error saving conversation: {e}[/red]")

    console.log(f"\n[cyan]Attempting to load conversation:[/cyan] '{conversation_name}' from '{conversation_file}'")
    try:
        loaded = await bot.load_conversation(conversation_file, conversation_name)
        if loaded:
            console.log("[green]Conversation loaded successfully. Sending a follow-up query.[/green]")
            follow_up_message = "What was the first question I asked in this session?"
            console.log(f"[cyan]Sending follow-up query:[/cyan] '{follow_up_message}'")
            response_follow_up = await bot.ask(follow_up_message)
            if response_follow_up.get("error"):
                 console.log(f"[red]Error in follow-up response: {response_follow_up.get('content')}[/red]")
            else:
                 console.log("[blue]Follow-up Response:[/blue]")
                 console.print(Markdown(response_follow_up.get("content", "No content received.")))
        else:
            console.log("[yellow]Could not load the conversation.[/yellow]")
    except Exception as e:
        console.log(f"[red]Error loading or using loaded conversation: {e}[/red]")

    # --- Cleanup ---
    if bot and bot.session:
        await bot.session.close() # Use close() for AsyncSession
        console.log("\n[grey]HTTP session closed.[/grey]")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.log("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as main_e:
        console.log(f"[bold red]An error occurred in the main execution: {main_e}[/bold red]")

