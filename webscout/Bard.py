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
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Use curl_cffi for requests
# Import trio before curl_cffi to prevent eventlet socket monkey-patching conflicts
# See: https://github.com/python-trio/trio/issues/3015
try:
    import trio  # noqa: F401
except ImportError:
    pass  # trio is optional, ignore if not available
from curl_cffi import CurlError
from curl_cffi.requests import AsyncSession

# For image models using validation. Adjust based on organization internal pydantic.
# Updated import for Pydantic V2
from pydantic import BaseModel, field_validator

# Import common request exceptions (curl_cffi often wraps these)
from requests.exceptions import HTTPError, RequestException, Timeout

# Rich is retained for logging within image methods.
from rich.console import Console

console = Console()

#########################################
# New Enums and functions for endpoints,
# headers, models, file upload and images.
#########################################

class Endpoint(Enum):
    """
    Enum for Google Gemini API endpoints.

    Attributes:
        INIT (str): URL for initializing the Gemini session.
        GENERATE (str): URL for generating chat responses.
        ROTATE_COOKIES (str): URL for rotating authentication cookies.
        UPLOAD (str): URL for uploading files/images.
    """
    INIT = "https://gemini.google.com/app"
    GENERATE = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
    ROTATE_COOKIES = "https://accounts.google.com/RotateCookies"
    UPLOAD = "https://content-push.googleapis.com/upload"

class Headers(Enum):
    """
    Enum for HTTP headers used in Gemini API requests.

    Attributes:
        GEMINI (dict): Headers for Gemini chat requests.
        ROTATE_COOKIES (dict): Headers for rotating cookies.
        UPLOAD (dict): Headers for file/image upload.
    """
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
    """
    Enum for available Gemini model configurations.

    Attributes:
        model_name (str): Name of the model.
        model_header (dict): Additional headers required for the model.
        advanced_only (bool): Whether the model is available only for advanced users.
    """
    # Only the specified models
    UNSPECIFIED = ("unspecified", {}, False)
    G_2_5_FLASH = (
        "gemini-2.5-flash",
        {"x-goog-ext-525001261-jspb": '[1,null,null,null,"71c2d248d3b102ff"]'},
        False,
    )
    G_2_5_PRO = (
        "gemini-2.5-pro",
        {"x-goog-ext-525001261-jspb": '[1,null,null,null,"2525e3954d185b3c"]'},
        False,
    )

    def __init__(self, name, header, advanced_only):
        """
        Initialize a Model enum member.

        Args:
            name (str): Model name.
            header (dict): Model-specific headers.
            advanced_only (bool): If True, model is for advanced users only.
        """
        self.model_name = name
        self.model_header = header
        self.advanced_only = advanced_only

    @classmethod
    def from_name(cls, name: str):
        """
        Get a Model enum member by its model name.

        Args:
            name (str): Name of the model.

        Returns:
            Model: Corresponding Model enum member.

        Raises:
            ValueError: If the model name is not found.
        """
        for model in cls:
            if model.model_name == name:
                return model
        raise ValueError(
            f"Unknown model name: {name}. Available models: {', '.join([model.model_name for model in cls])}"
        )

async def upload_file(
    file: Union[bytes, str, Path],
    proxy: Optional[Union[str, Dict[str, str]]] = None,
    impersonate: str = "chrome110"
) -> str:
    """
    Uploads a file to Google's Gemini server using curl_cffi and returns its identifier.

    Args:
        file (bytes | str | Path): File data in bytes or path to the file to be uploaded.
        proxy (str | dict, optional): Proxy URL or dictionary for the request.
        impersonate (str, optional): Browser profile for curl_cffi to impersonate. Defaults to "chrome110".

    Returns:
        str: Identifier of the uploaded file.

    Raises:
        HTTPError: If the upload request fails.
        RequestException: For other network-related errors.
        FileNotFoundError: If the file path does not exist.
    """
    # Handle file input
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
            headers=Headers.UPLOAD.value # Pass headers directly
            # follow_redirects is handled automatically by curl_cffi
        ) as client:
            response = await client.post(
                url=Endpoint.UPLOAD.value,
                files={"file": file_content},
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
    """
    Loads authentication cookies from a JSON file.

    Args:
        cookie_path (str): Path to the JSON file containing cookies.

    Returns:
        tuple[str, str]: Tuple containing __Secure-1PSID and __Secure-1PSIDTS cookie values.

    Raises:
        Exception: If the file is not found, invalid, or required cookies are missing.
    """
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

    This class provides a synchronous interface to interact with Google Gemini,
    handling authentication, conversation management, and message sending.

    Attributes:
        loop (asyncio.AbstractEventLoop): Event loop for running async tasks.
        secure_1psid (str): Authentication cookie.
        secure_1psidts (str): Authentication cookie.
        async_chatbot (AsyncChatbot): Underlying asynchronous chatbot instance.
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
    Asynchronous chatbot client for interacting with Google Gemini using curl_cffi.

    This class manages authentication, session state, conversation history,
    and sending/receiving messages (including images) asynchronously.

    Attributes:
        headers (dict): HTTP headers for requests.
        _reqid (int): Request identifier for Gemini API.
        SNlM0e (str): Session token required for API requests.
        conversation_id (str): Current conversation ID.
        response_id (str): Current response ID.
        choice_id (str): Current choice ID.
        proxy (str | dict | None): Proxy configuration.
        proxies_dict (dict | None): Proxy dictionary for curl_cffi.
        secure_1psid (str): Authentication cookie.
        secure_1psidts (str): Authentication cookie.
        session (AsyncSession): curl_cffi session for HTTP requests.
        timeout (int): Request timeout in seconds.
        model (Model): Selected Gemini model.
        impersonate (str): Browser profile for curl_cffi to impersonate.
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
            impersonate=self.impersonate
            # verify and http2 are handled automatically by curl_cffi
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
        if not self.secure_1psid:
            raise ValueError("__Secure-1PSID cookie is required.")

        try:
            # Use the session's get method
            resp = await self.session.get(
                Endpoint.INIT.value,
                timeout=self.timeout # Timeout is already set in session, but can override
                # follow_redirects is handled automatically by curl_cffi
            )
            resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Check for authentication issues
            if "Sign in to continue" in resp.text or "accounts.google.com" in str(resp.url):
                raise PermissionError("Authentication failed. Cookies might be invalid or expired. Please update them.")

            # Regex to find the SNlM0e value
            snlm0e_match = re.search(r'["\']SNlM0e["\']\s*:\s*["\'](.*?)["\']', resp.text)
            if not snlm0e_match:
                error_message = "SNlM0e value not found in response."
                if resp.status_code == 429:
                    error_message += " Rate limit likely exceeded."
                else:
                    error_message += f" Response status: {resp.status_code}. Check cookie validity and network."
                raise ValueError(error_message)

            # Try to refresh PSIDTS if needed
            if not self.secure_1psidts and "PSIDTS" not in self.session.cookies:
                try:
                    # Attempt to rotate cookies to get a fresh PSIDTS
                    await self.__rotate_cookies()
                except Exception as e:
                    console.log(f"[yellow]Warning: Could not refresh PSIDTS cookie: {e}[/yellow]")
                    # Continue anyway as some accounts don't need PSIDTS

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

    async def __rotate_cookies(self):
        """Rotates the __Secure-1PSIDTS cookie."""
        try:
            response = await self.session.post(
                Endpoint.ROTATE_COOKIES.value,
                headers=Headers.ROTATE_COOKIES.value,
                data='[000,"-0000000000000000000"]',
                timeout=self.timeout
            )
            response.raise_for_status()

            if new_1psidts := response.cookies.get("__Secure-1PSIDTS"):
                self.secure_1psidts = new_1psidts
                self.session.cookies.set("__Secure-1PSIDTS", new_1psidts)
                return new_1psidts
        except Exception as e:
            console.log(f"[yellow]Cookie rotation failed: {e}[/yellow]")
            raise


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
            "bl": "boq_assistant-bard-web-server_20240625.13_p0",
            "_reqid": str(self._reqid),
            "rt": "c",
        }

        # Handle image upload if provided
        image_upload_id = None
        if image:
            try:
                # Pass proxy and impersonate settings to upload_file
                image_upload_id = await upload_file(image, proxy=self.proxies_dict, impersonate=self.impersonate)
                console.log(f"Image uploaded successfully. ID: {image_upload_id}")
            except Exception as e:
                console.log(f"[red]Error uploading image: {e}[/red]")
                return {"content": f"Error uploading image: {e}", "error": True}

        # Prepare message structure
        if image_upload_id:
            message_struct = [
                [message],
                [[[image_upload_id, 1]]],
                [self.conversation_id, self.response_id, self.choice_id],
            ]
        else:
            message_struct = [
                [message],
                None,
                [self.conversation_id, self.response_id, self.choice_id],
            ]

        # Prepare request data
        data = {
            "f.req": json.dumps([None, json.dumps(message_struct, ensure_ascii=False)], ensure_ascii=False),
            "at": self.SNlM0e,
        }

        try:
            # Send request
            resp = await self.session.post(
                Endpoint.GENERATE.value,
                params=params,
                data=data,
                timeout=self.timeout,
            )
            resp.raise_for_status()

            # Process response
            lines = resp.text.splitlines()
            if len(lines) < 3:
                raise ValueError(f"Unexpected response format. Status: {resp.status_code}. Content: {resp.text[:200]}...")

            # Find the line with the response data
            chat_data_line = None
            for line in lines:
                if line.startswith(")]}'"):
                    chat_data_line = line[4:].strip()
                    break
                elif line.startswith("["):
                    chat_data_line = line
                    break

            if not chat_data_line:
                chat_data_line = lines[3] if len(lines) > 3 else lines[-1]
                if chat_data_line.startswith(")]}'"):
                    chat_data_line = chat_data_line[4:].strip()

            # Parse the response JSON
            response_json = json.loads(chat_data_line)

            # Find the main response body
            body = None
            body_index = 0

            for part_index, part in enumerate(response_json):
                try:
                    if isinstance(part, list) and len(part) > 2:
                        main_part = json.loads(part[2])
                        if main_part and len(main_part) > 4 and main_part[4]:
                            body = main_part
                            body_index = part_index
                            break
                except (IndexError, TypeError, json.JSONDecodeError):
                    continue

            if not body:
                return {"content": "Failed to parse response body. No valid data found.", "error": True}

            # Extract data from the response
            try:
                # Extract main content
                content = ""
                if len(body) > 4 and len(body[4]) > 0 and len(body[4][0]) > 1:
                    content = body[4][0][1][0] if len(body[4][0][1]) > 0 else ""

                # Extract conversation metadata
                conversation_id = body[1][0] if len(body) > 1 and len(body[1]) > 0 else self.conversation_id
                response_id = body[1][1] if len(body) > 1 and len(body[1]) > 1 else self.response_id

                # Extract additional data
                factualityQueries = body[3] if len(body) > 3 else None
                textQuery = body[2][0] if len(body) > 2 and body[2] else ""

                # Extract choices
                choices = []
                if len(body) > 4:
                    for candidate in body[4]:
                        if len(candidate) > 1 and isinstance(candidate[1], list) and len(candidate[1]) > 0:
                            choices.append({"id": candidate[0], "content": candidate[1][0]})

                choice_id = choices[0]["id"] if choices else self.choice_id

                # Extract images - multiple possible formats
                images = []

                # Format 1: Regular web images
                if len(body) > 4 and len(body[4]) > 0 and len(body[4][0]) > 4 and body[4][0][4]:
                    for img_data in body[4][0][4]:
                        try:
                            img_url = img_data[0][0][0]
                            img_alt = img_data[2] if len(img_data) > 2 else ""
                            img_title = img_data[1] if len(img_data) > 1 else "[Image]"
                            images.append({"url": img_url, "alt": img_alt, "title": img_title})
                        except (IndexError, TypeError):
                            console.log("[yellow]Warning: Could not parse image data structure (format 1).[/yellow]")
                            continue

                # Format 2: Generated images in standard location
                generated_images = []
                if len(body) > 4 and len(body[4]) > 0 and len(body[4][0]) > 12 and body[4][0][12]:
                    try:
                        # Path 1: Check for images in [12][7][0]
                        if body[4][0][12][7] and body[4][0][12][7][0]:
                            # This is the standard path for generated images
                            for img_index, img_data in enumerate(body[4][0][12][7][0]):
                                try:
                                    img_url = img_data[0][3][3]
                                    img_title = f"[Generated Image {img_index+1}]"
                                    img_alt = img_data[3][5][0] if len(img_data[3]) > 5 and len(img_data[3][5]) > 0 else ""
                                    generated_images.append({"url": img_url, "alt": img_alt, "title": img_title})
                                except (IndexError, TypeError):
                                    continue

                            # If we found images, but they might be in a different part of the response
                            if not generated_images:
                                # Look for image generation data in other response parts
                                for part_index, part in enumerate(response_json):
                                    if part_index <= body_index:
                                        continue
                                    try:
                                        img_part = json.loads(part[2])
                                        if img_part[4][0][12][7][0]:
                                            for img_index, img_data in enumerate(img_part[4][0][12][7][0]):
                                                try:
                                                    img_url = img_data[0][3][3]
                                                    img_title = f"[Generated Image {img_index+1}]"
                                                    img_alt = img_data[3][5][0] if len(img_data[3]) > 5 and len(img_data[3][5]) > 0 else ""
                                                    generated_images.append({"url": img_url, "alt": img_alt, "title": img_title})
                                                except (IndexError, TypeError):
                                                    continue
                                            break
                                    except (IndexError, TypeError, json.JSONDecodeError):
                                        continue
                    except (IndexError, TypeError):
                        pass

                # Format 3: Alternative location for generated images
                if len(generated_images) == 0 and len(body) > 4 and len(body[4]) > 0:
                    try:
                        # Try to find images in candidate[4] structure
                        candidate = body[4][0]
                        if len(candidate) > 22 and candidate[22]:
                            # Look for URLs in the candidate[22] field
                            import re
                            content = candidate[22][0] if isinstance(candidate[22], list) and len(candidate[22]) > 0 else str(candidate[22])
                            urls = re.findall(r'https?://[^\s]+', content)
                            for i, url in enumerate(urls):
                                # Clean up URL if it ends with punctuation
                                if url[-1] in ['.', ',', ')', ']', '}', '"', "'"]:
                                    url = url[:-1]
                                generated_images.append({
                                    "url": url,
                                    "title": f"[Generated Image {i+1}]",
                                    "alt": ""
                                })
                    except (IndexError, TypeError) as e:
                        console.log(f"[yellow]Warning: Could not parse alternative image structure: {e}[/yellow]")

                # Format 4: Look for image URLs in the text content
                if len(images) == 0 and len(generated_images) == 0 and content:
                    try:
                        import re
                        # Look for image URLs in the content - try multiple patterns

                        # Pattern 1: Standard image URLs
                        urls = re.findall(r'(https?://[^\s]+\.(jpg|jpeg|png|gif|webp))', content.lower())

                        # Pattern 2: Google image URLs (which might not have extensions)
                        google_urls = re.findall(r'(https?://lh\d+\.googleusercontent\.com/[^\s]+)', content)

                        # Pattern 3: General URLs that might be images
                        general_urls = re.findall(r'(https?://[^\s]+)', content)

                        # Combine all found URLs
                        all_urls = []
                        if urls:
                            all_urls.extend([url_tuple[0] for url_tuple in urls])
                        if google_urls:
                            all_urls.extend(google_urls)

                        # Add general URLs only if we didn't find any specific image URLs
                        if not all_urls and general_urls:
                            all_urls = general_urls

                        # Process all found URLs
                        if all_urls:
                            for i, url in enumerate(all_urls):
                                # Clean up URL if it ends with punctuation
                                if url[-1] in ['.', ',', ')', ']', '}', '"', "'"]:
                                    url = url[:-1]
                                images.append({
                                    "url": url,
                                    "title": f"[Image in Content {i+1}]",
                                    "alt": ""
                                })
                            console.log(f"[green]Found {len(all_urls)} potential image URLs in content.[/green]")
                    except Exception as e:
                        console.log(f"[yellow]Warning: Error extracting URLs from content: {e}[/yellow]")

                # Combine all images
                all_images = images + generated_images

                # Prepare results
                results = {
                    "content": content,
                    "conversation_id": conversation_id,
                    "response_id": response_id,
                    "factualityQueries": factualityQueries,
                    "textQuery": textQuery,
                    "choices": choices,
                    "images": all_images,
                    "error": False,
                }

                # Update state
                self.conversation_id = conversation_id
                self.response_id = response_id
                self.choice_id = choice_id
                self._reqid += random.randint(1000, 9000)

                return results

            except (IndexError, TypeError) as e:
                console.log(f"[red]Error extracting data from response: {e}[/red]")
                return {"content": f"Error extracting data from response: {e}", "error": True}

        except json.JSONDecodeError as e:
            console.log(f"[red]Error parsing JSON response: {e}[/red]")
            return {"content": f"Error parsing JSON response: {e}. Response: {resp.text[:200]}...", "error": True}
        except Timeout as e:
            console.log(f"[red]Request timed out: {e}[/red]")
            return {"content": f"Request timed out: {e}", "error": True}
        except (RequestException, CurlError) as e:
            console.log(f"[red]Network error: {e}[/red]")
            return {"content": f"Network error: {e}", "error": True}
        except HTTPError as e:
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
    Represents a single image object returned from Gemini.

    Attributes:
        url (str): URL of the image.
        title (str): Title of the image (default: "[Image]").
        alt (str): Optional description of the image.
        proxy (str | dict | None): Proxy used when saving the image.
        impersonate (str): Browser profile for curl_cffi to impersonate.
    """
    url: str
    title: str = "[Image]"
    alt: str = ""
    proxy: Optional[Union[str, Dict[str, str]]] = None
    impersonate: str = "chrome110"

    def __str__(self):
        return f"{self.title}({self.url}) - {self.alt}"

    def __repr__(self):
        short_url = self.url if len(self.url) <= 50 else self.url[:20] + "..." + self.url[-20:]
        short_alt = self.alt[:30] + "..." if len(self.alt) > 30 else self.alt
        return f"Image(title='{self.title}', url='{short_url}', alt='{short_alt}')"

    async def save(
        self,
        path: str = "downloaded_images",
        filename: Optional[str] = None,
        cookies: Optional[dict] = None,
        verbose: bool = False,
        skip_invalid_filename: bool = True,
    ) -> Optional[str]:
        """
        Save the image to disk using curl_cffi.
        Parameters:
            path: str, optional
                Directory to save the image (default "downloaded_images").
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
        # Generate filename from URL if not provided
        if not filename:
            try:
                from urllib.parse import unquote, urlparse
                parsed_url = urlparse(self.url)
                base_filename = os.path.basename(unquote(parsed_url.path))
                # Remove invalid characters for filenames
                safe_filename = re.sub(r'[<>:"/\\|?*]', '_', base_filename)
                if safe_filename and len(safe_filename) > 0:
                    filename = safe_filename
                else:
                    filename = f"image_{random.randint(1000, 9999)}.jpg"
            except Exception:
                filename = f"image_{random.randint(1000, 9999)}.jpg"

        # Validate filename length
        try:
            _ = Path(filename)
            max_len = 255
            if len(filename) > max_len:
                name, ext = os.path.splitext(filename)
                filename = name[:max_len - len(ext) - 1] + ext
        except (OSError, ValueError):
            if verbose:
                console.log(f"[yellow]Invalid filename generated: {filename}[/yellow]")
            if skip_invalid_filename:
                if verbose:
                    console.log("[yellow]Skipping save due to invalid filename.[/yellow]")
                return None
            filename = f"image_{random.randint(1000, 9999)}.jpg"
            if verbose:
                console.log(f"[yellow]Using fallback filename: {filename}[/yellow]")

        # Prepare proxy dictionary for curl_cffi
        proxies_dict = None
        if isinstance(self.proxy, str):
            proxies_dict = {"http": self.proxy, "https": self.proxy}
        elif isinstance(self.proxy, dict):
            proxies_dict = self.proxy

        try:
            # Use AsyncSession from curl_cffi
            async with AsyncSession(
                cookies=cookies,
                proxies=proxies_dict,
                impersonate=self.impersonate
                # follow_redirects is handled automatically by curl_cffi
            ) as client:
                if verbose:
                    console.log(f"Attempting to download image from: {self.url}")

                response = await client.get(self.url)
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if "image" not in content_type and verbose:
                    console.log(f"[yellow]Warning: Content type is '{content_type}', not an image. Saving anyway.[/yellow]")

                # Create directory and save file
                dest_path = Path(path)
                dest_path.mkdir(parents=True, exist_ok=True)
                dest = dest_path / filename

                # Write image data to file
                dest.write_bytes(response.content)

                if verbose:
                    console.log(f"Image saved successfully as {dest.resolve()}")

                return str(dest.resolve())

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
    Represents an image retrieved from web search results.

    Returned when asking Gemini to "SEND an image of [something]".
    """
    pass

class GeneratedImage(Image):
    """
    Represents an image generated by Google's AI image generator (e.g., ImageFX).

    Attributes:
        cookies (dict[str, str]): Cookies required for accessing the generated image URL,
            typically from the GeminiClient/Chatbot instance.
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
