from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Generator, Any

# Re-define or import necessary response structure classes (like ChatCompletion, ChatCompletionChunk)
# For simplicity, we'll assume they are defined elsewhere or passed directly.
# You might want to define base versions of these classes here as well.

class BaseChatCompletionChunk: # Placeholder
    pass
class BaseChatCompletion: # Placeholder
    pass


class BaseCompletions(ABC):
    @abstractmethod
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Union[BaseChatCompletion, Generator[BaseChatCompletionChunk, None, None]]:
        """Abstract method to create chat completions."""
        raise NotImplementedError


class BaseChat(ABC):
    completions: BaseCompletions


class OpenAICompatibleProvider(ABC):
    """
    Abstract Base Class for providers mimicking the OpenAI Python client structure.
    Requires a nested 'chat.completions' structure.
    """
    chat: BaseChat

    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, **kwargs: Any):
        """Initialize the provider, potentially with an API key."""
        raise NotImplementedError

