"""
This module provides the AUTO provider, which automatically selects and uses
an available LLM provider from the webscout library that doesn't require
API keys or cookies. 
"""

from webscout.AIbase import Provider
from webscout.exceptions import AllProvidersFailure
from typing import Union, Any, Dict, Generator, Optional
import importlib
import pkgutil
import random
import inspect

def load_providers():
    """
    Dynamically loads all Provider classes from the `webscout.Provider` package.
    
    This function iterates through the modules in the `webscout.Provider` package,
    imports each module, and inspects its attributes to identify classes that
    inherit from the `Provider` base class. It also identifies providers that
    require special authentication parameters like 'api_key', 'cookie_file', or
    'cookie_path'.
    
    Returns:
        tuple: A tuple containing two elements:
            - provider_map (dict): A dictionary mapping uppercase provider names to their classes.
            - api_key_providers (set): A set of uppercase provider names requiring special authentication.
    """
    provider_map = {}
    api_key_providers = set()
    cookie_providers = set()
    provider_package = importlib.import_module("webscout.Provider")
    
    for _, module_name, _ in pkgutil.iter_modules(provider_package.__path__):
        try:
            module = importlib.import_module(f"webscout.Provider.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Provider) and attr != Provider:
                    provider_map[attr_name.upper()] = attr
                    # Check if the provider needs special parameters
                    sig = inspect.signature(attr.__init__).parameters
                    if 'api_key' in sig:
                        api_key_providers.add(attr_name.upper())
                    if 'cookie_file' in sig or 'cookie_path' in sig:
                        cookie_providers.add(attr_name.upper())
        except Exception as e:
            print(f"Error loading provider {module_name}: {e}")
    return provider_map, api_key_providers.union(cookie_providers)

provider_map, api_key_providers = load_providers()

class AUTO(Provider):
    """
    An automatic provider that intelligently selects and utilizes an available
    LLM provider from the webscout library.
    
    It cycles through available free providers
    until one successfully processes the request. Excludes providers
    requiring API keys or cookies by default.
    """
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
        exclude: Optional[list[str]] = None,
        print_provider_info: bool = False,
    ):
        """
        Initializes the AUTO provider, setting up the parameters for provider selection and request handling.
        
        This constructor initializes the AUTO provider with various configuration options,
        including conversation settings, request limits, and provider exclusions.

        Args:
            is_conversation (bool): Flag for conversational mode. Defaults to True.
            max_tokens (int): Maximum tokens for the response. Defaults to 600.
            timeout (int): Request timeout in seconds. Defaults to 30.
            intro (str, optional): Introductory prompt. Defaults to None.
            filepath (str, optional): Path for conversation history. Defaults to None.
            update_file (bool): Whether to update the history file. Defaults to True.
            proxies (dict): Proxies for requests. Defaults to {}.
            history_offset (int): History character offset limit. Defaults to 10250.
            act (str, optional): Awesome prompt key. Defaults to None.
            exclude (Optional[list[str]]): List of provider names (uppercase) to exclude. Defaults to None.
            print_provider_info (bool): Whether to print the name of the successful provider. Defaults to False.
        """
        self.provider = None  # type: Provider
        self.provider_name = None  # type: str
        self.is_conversation: bool = is_conversation
        self.max_tokens: int = max_tokens
        self.timeout: int = timeout
        self.intro: str = intro
        self.filepath: str = filepath
        self.update_file: bool = update_file
        self.proxies: dict = proxies
        self.history_offset: int = history_offset
        self.act: str = act
        self.exclude: list[str] = [e.upper() for e in exclude] if exclude else []
        self.print_provider_info: bool = print_provider_info


    @property
    def last_response(self) -> dict[str, Any]:
        """
        Retrieves the last response dictionary from the successfully used provider.
        
        Returns:
            dict[str, Any]: The last response dictionary, or an empty dictionary if no provider has been used yet.
        """
        return self.provider.last_response if self.provider else {}

    @property
    def conversation(self) -> object:
        """
        Retrieves the conversation object from the successfully used provider.
        
        Returns:
            object: The conversation object, or None if no provider has been used yet.
        """
        return self.provider.conversation if self.provider else None

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict, Generator]:
        """
        Sends the prompt to available providers, attempting to get a response from each until one succeeds.
        
        This method iterates through a shuffled list of available providers (excluding those requiring API keys or
        specified in the exclusion list) and attempts to send the prompt to each provider until a successful response
        is received.

        Args:
            prompt (str): The user's prompt.
            stream (bool): Whether to stream the response. Defaults to False.
            raw (bool): Whether to return the raw response format. Defaults to False.
            optimizer (str, optional): Name of the optimizer to use. Defaults to None.
            conversationally (bool): Whether to apply optimizer conversationally. Defaults to False.

        Returns:
            Union[Dict, Generator]: The response dictionary or generator from the successful provider.
        """
        ask_kwargs = {
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
            "optimizer": optimizer,
            "conversationally": conversationally,
        }

        # Filter out API key required providers and excluded providers
        available_providers = [
            (name, cls) for name, cls in provider_map.items()
            if name not in api_key_providers and name not in self.exclude
        ]

        # Shuffle the list of available providers
        random.shuffle(available_providers)

        # Try webscout-based providers
        for provider_name, provider_class in available_providers:
            try:
                self.provider_name = f"webscout-{provider_name}"
                
                self.provider = provider_class(
                    is_conversation=self.is_conversation,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    intro=self.intro,
                    filepath=self.filepath,
                    update_file=self.update_file,
                    proxies=self.proxies,
                    history_offset=self.history_offset,
                    act=self.act,
                )
                response = self.provider.ask(**ask_kwargs)
                # Print provider info if enabled
                if self.print_provider_info:
                    print(f"\033[1;34m{self.provider_name}\033[0m\n")
                return response
            except Exception:
                continue

        # If we get here, all providers failed
        raise AllProvidersFailure("All providers failed to process the request")

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]: 
        """
        Provides a simplified chat interface, returning the message string or a generator of message strings.

        Args:
            prompt (str): The user's prompt.
            stream (bool): Whether to stream the response. Defaults to False.
            optimizer (str, optional): Name of the optimizer to use. Defaults to None.
            conversationally (bool): Whether to apply optimizer conversationally. Defaults to False.

        Returns:
            Union[str, Generator[str, None, None]]: The response string or a generator yielding
                                                     response chunks.
        """
        # run_new_test is not used in the current implementation, but we keep it for API compatibility
        response = self.ask(
            prompt,
            stream,
            optimizer=optimizer,
            conversationally=conversationally,
        )
        
        if stream:
            for chunk in response:
                yield self.get_message(chunk)
        else:
            return self.get_message(response)

    def get_message(self, response: dict) -> str:
        """
        Extracts the message text from the provider's response dictionary.

        Args:
            response (dict): The response dictionary obtained from the `ask` method.

        Returns:
            str: The extracted message string.
        """
        assert self.provider is not None, "Chat with AI first"
        return self.provider.get_message(response)
    
if __name__ == "__main__":
    auto = AUTO(print_provider_info=True)
    response = auto.chat("Hello, how are you?", stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)