from typing import List, Dict, Optional, Any, Union, Literal
from dataclasses import dataclass, asdict, is_dataclass, field
from enum import Enum
import time
import uuid

# --- OpenAI Response Structure Mimics ---
# Moved here for reusability across different OpenAI-compatible providers

class ToolCallType(str, Enum):
    """Type of tool call."""
    FUNCTION = "function"

@dataclass
class BaseModel:
    """Base class for all models."""
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        def _convert(obj: Any) -> Any:
            if is_dataclass(obj):
                return {k: _convert(v) for k, v in asdict(obj).items() if v is not None}
            elif isinstance(obj, list):
                return [_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        return _convert(self)

    def __getitem__(self, key):
        """Support dictionary-style access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} not found in {self.__class__.__name__}")

    def get(self, key, default=None):
        """Dictionary-style get method with default value."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        """Support 'in' operator."""
        return hasattr(self, key)

@dataclass
class FunctionCall(BaseModel):
    """Function call specification."""
    name: str
    arguments: str

@dataclass
class ToolFunction(BaseModel):
    """Function specification in a tool."""
    name: str
    arguments: str

@dataclass
class ToolCall(BaseModel):
    """Tool call specification."""
    id: str
    type: str
    function: ToolFunction

@dataclass
class CompletionUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, Any]] = None

@dataclass
class ChoiceDelta(BaseModel):
    """Delta content in streaming response."""
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    role: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

@dataclass
class ChatCompletionMessage(BaseModel):
    """Chat message in completion response."""
    role: str
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None

@dataclass
class Choice(BaseModel):
    """Choice in completion response."""
    index: int
    message: Optional[ChatCompletionMessage] = None
    delta: Optional[ChoiceDelta] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None

@dataclass
class ModelData(BaseModel):
    """OpenAI model info response."""
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "webscout"
    permission: Optional[List[Dict[str, Any]]] = None
    root: Optional[str] = None
    parent: Optional[str] = None

@dataclass
class ModelList(BaseModel):
    """OpenAI model list response."""
    data: List[ModelData] # Moved before 'object'
    object: str = "list"


# @dataclass
# class EmbeddingData(BaseModel):
#     """Single embedding data."""
#     embedding: List[float]
#     index: int
#     object: str = "embedding"

# @dataclass
# class EmbeddingResponse(BaseModel):
#     """OpenAI embeddings response."""
#     data: List[EmbeddingData]
#     model: str
#     usage: CompletionUsage
#     object: str = "list"

# @dataclass
# class FineTuningJob(BaseModel):
#     """OpenAI fine-tuning job."""
#     id: str
#     model: str
#     created_at: int
#     status: str
#     training_file: str
#     hyperparameters: Dict[str, Any]
#     object: str = "fine_tuning.job"
#     finished_at: Optional[int] = None
#     validation_file: Optional[str] = None
#     trained_tokens: Optional[int] = None
#     result_files: Optional[List[str]] = None
#     organization_id: Optional[str] = None

# @dataclass
# class FineTuningJobList(BaseModel):
#     """OpenAI fine-tuning job list response."""
#     data: List[FineTuningJob]
#     object: str = "list"
#     has_more: bool = False

# @dataclass
# class File(BaseModel):
#     """OpenAI file."""
#     id: str
#     bytes: int
#     created_at: int
#     filename: str
#     purpose: str
#     object: str = "file"
#     status: str = "uploaded"
#     status_details: Optional[str] = None

# @dataclass
# class FileList(BaseModel):
#     """OpenAI file list response."""
#     data: List[File]
#     object: str = "list"

# @dataclass
# class DeletedObject(BaseModel):
#     """OpenAI deleted object response."""
#     id: str
#     object: str = "deleted_object"
#     deleted: bool = True

# @dataclass
# class ImageData(BaseModel):
#     """OpenAI generated image."""
#     url: Optional[str] = None
#     b64_json: Optional[str] = None
#     revised_prompt: Optional[str] = None

# @dataclass
# class ImageResponse(BaseModel):
#     """OpenAI image generation response."""
#     data: List[ImageData]
#     created: int = int(time.time())

@dataclass
class ChatCompletion(BaseModel):
    """Chat completion response."""
    model: str
    choices: List[Choice]
    id: str = field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"
    system_fingerprint: Optional[str] = None
    usage: Optional[CompletionUsage] = None

@dataclass
class ChatCompletionChunk(BaseModel):
    """Streaming chat completion response chunk."""
    model: str
    choices: List[Choice]
    id: str = field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion.chunk"
    system_fingerprint: Optional[str] = None


# --- Helper Functions ---

def format_prompt(messages: List[Dict[str, Any]], add_special_tokens: bool = False,
                 do_continue: bool = False, include_system: bool = True) -> str:
    """
    Format a series of messages into a single string, optionally adding special tokens.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content'.
        add_special_tokens: Whether to add special formatting tokens.
        do_continue: If True, don't add the final "Assistant:" prompt.
        include_system: Whether to include system messages in the formatted output.

    Returns:
        A formatted string containing all messages.
    """
    # Helper function to convert content to string
    def to_string(value) -> str:
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            if "text" in value:
                return value.get("text", "")
            return ""
        elif isinstance(value, list):
            return "".join([to_string(v) for v in value])
        return str(value)

    # If there's only one message and no special tokens needed, just return its content
    if not add_special_tokens and len(messages) <= 1:
        return to_string(messages[0]["content"])

    # Filter and process messages
    processed_messages = [
        (message["role"], to_string(message["content"]))
        for message in messages
        if include_system or message.get("role") != "system"
    ]

    # Format each message as "Role: Content"
    formatted = "\n".join([
        f'{role.capitalize()}: {content}'
        for role, content in processed_messages
        if content.strip()
    ])

    # Add final prompt for assistant if needed
    if do_continue:
        return formatted

    return f"{formatted}\nAssistant:"


def get_system_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Extract and concatenate all system messages.

    Args:
        messages: A list of message dictionaries.

    Returns:
        A string containing all system messages concatenated with newlines.
    """
    return "\n".join([m["content"] for m in messages if m["role"] == "system"])


def get_last_user_message(messages: List[Dict[str, Any]]) -> str:
    """
    Get the content of the last user message in the conversation.

    Args:
        messages: A list of message dictionaries.

    Returns:
        The content of the last user message as a string.
    """
    for message in reversed(messages):
        if message["role"] == "user":
            if isinstance(message["content"], str):
                return message["content"]
            # Handle complex content structures
            if isinstance(message["content"], dict) and "text" in message["content"]:
                return message["content"]["text"]
            if isinstance(message["content"], list):
                text_parts = []
                for part in message["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "".join(text_parts)
    return ""
