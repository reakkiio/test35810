from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Generator, Any, TypedDict, Callable
import json
from dataclasses import dataclass

# Import WebScout Litlogger instead of standard logging
from webscout.Litlogger import Logger, LogLevel

logger = Logger(name="OpenAIBase", level=LogLevel.INFO)

# Import the LitMeta metaclass from Litproxy
try:
    from litproxy import LitMeta 
except ImportError:
    from .autoproxy import ProxyAutoMeta as LitMeta


# Import the utils for response structures
from webscout.Provider.OPENAI.utils import ChatCompletion, ChatCompletionChunk

# Define tool-related structures
class ToolDefinition(TypedDict):
    """Definition of a tool that can be called by the AI"""
    type: str
    function: Dict[str, Any]

class FunctionParameters(TypedDict):
    """Parameters for a function"""
    type: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str]

class FunctionDefinition(TypedDict):
    """Definition of a function that can be called by the AI"""
    name: str
    description: str
    parameters: FunctionParameters

@dataclass
class Tool:
    """Tool class that can be passed to the provider"""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    required_params: List[str] = None
    implementation: Optional[Callable] = None
    
    def to_dict(self) -> ToolDefinition:
        """Convert to OpenAI tool definition format"""
        function_def = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params or list(self.parameters.keys())
            }
        }
        
        return {
            "type": "function",
            "function": function_def
        }
    
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with the given arguments"""
        if not self.implementation:
            return f"Tool '{self.name}' does not have an implementation."
            
        try:
            return self.implementation(**arguments)
        except Exception as e:
            logger.error(f"Error executing tool '{self.name}': {str(e)}")
            return f"Error executing tool '{self.name}': {str(e)}"

class BaseCompletions(ABC):
    @abstractmethod
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],  # Changed to Any to support complex message structures
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None,  # Support for tool definitions
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,  # Support for tool_choice parameter
        timeout: Optional[int] = None,
        proxies: Optional[dict] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Abstract method to create chat completions with tool support.
        
        Args:
            model: The model to use for completion
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tools: List of tool definitions available for the model to use
            tool_choice: Control over which tool the model should use
            **kwargs: Additional model-specific parameters
            
        Returns:
            Either a completion object or a generator of completion chunks if streaming
        """
        raise NotImplementedError
    
    def format_tool_calls(self, tools: List[Union[Tool, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Convert tools to the format expected by the provider"""
        formatted_tools = []
        
        for tool in tools:
            if isinstance(tool, Tool):
                formatted_tools.append(tool.to_dict())
            elif isinstance(tool, dict):
                # Assume already formatted correctly
                formatted_tools.append(tool)
            else:
                logger.warning(f"Skipping invalid tool type: {type(tool)}")
        
        return formatted_tools
    
    def process_tool_calls(self, tool_calls: List[Dict[str, Any]], available_tools: Dict[str, Tool]) -> List[Dict[str, Any]]:
        """
        Process tool calls and execute the relevant tools.
        
        Args:
            tool_calls: List of tool calls from the model
            available_tools: Dictionary mapping tool names to their implementations
            
        Returns:
            List of results from executing the tools
        """
        results = []
        
        for call in tool_calls:
            try:
                function_call = call.get("function", {})
                tool_name = function_call.get("name")
                arguments_str = function_call.get("arguments", "{}")
                
                # Parse arguments 
                try:
                    if isinstance(arguments_str, str):
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = arguments_str
                except json.JSONDecodeError:
                    results.append({
                        "tool_call_id": call.get("id"),
                        "result": f"Error: Could not parse arguments JSON: {arguments_str}"
                    })
                    continue
                
                # Execute the tool if available
                if tool_name in available_tools:
                    tool_result = available_tools[tool_name].execute(arguments)
                    results.append({
                        "tool_call_id": call.get("id"),
                        "result": str(tool_result)
                    })
                else:
                    results.append({
                        "tool_call_id": call.get("id"),
                        "result": f"Error: Tool '{tool_name}' not found."
                    })
            except Exception as e:
                logger.error(f"Error processing tool call: {str(e)}")
                results.append({
                    "tool_call_id": call.get("id", "unknown"),
                    "result": f"Error processing tool call: {str(e)}"
                })
        
        return results


class BaseChat(ABC):
    completions: BaseCompletions


class OpenAICompatibleProvider(ABC, metaclass=LitMeta):
    """
    Abstract Base Class for providers mimicking the OpenAI Python client structure.
    Requires a nested 'chat.completions' structure with tool support.
    All subclasses automatically get proxy support via LitMeta.

    # Available proxy helpers:
    # - self.get_proxied_session() - returns a requests.Session with proxies
    # - self.get_proxied_curl_session() - returns a curl_cffi.Session with proxies
    # - self.get_proxied_curl_async_session() - returns a curl_cffi.AsyncSession with proxies

    # Proxy support is automatically injected into:
    # - requests.Session objects
    # - httpx.Client objects
    # - curl_cffi.requests.Session objects
    # - curl_cffi.requests.AsyncSession objects
    #
    # Inbuilt auto-retry is also enabled for all requests.Session and curl_cffi.Session objects.
    """
    chat: BaseChat
    available_tools: Dict[str, Tool] = {}  # Dictionary of available tools
    supports_tools: bool = False  # Whether the provider supports tools
    supports_tool_choice: bool = False  # Whether the provider supports tool_choice

    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, tools: Optional[List[Tool]] = None, proxies: Optional[dict] = None, disable_auto_proxy: bool = False, **kwargs: Any):
        self.available_tools = {}
        if tools:
            self.register_tools(tools)
        # self.proxies is set by ProxyAutoMeta
        # Subclasses should use self.proxies for all network requests
        # Optionally, use self.get_proxied_session() for a requests.Session with proxies
        # The disable_auto_proxy parameter is handled by ProxyAutoMeta
        # raise NotImplementedError  # <-- Commented out for metaclass test

    @property
    @abstractmethod
    def models(self):
        """
        Property that returns an object with a .list() method returning available models.
        Subclasses must implement this property.
        """
        pass
    
    def register_tools(self, tools: List[Tool]) -> None:
        """
        Register tools with the provider.
        
        Args:
            tools: List of Tool objects to register
        """
        for tool in tools:
            self.available_tools[tool.name] = tool
            
    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.available_tools.get(name)
    
    def format_tool_response(self, messages: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool results as messages to be sent back to the provider.
        
        Args:
            messages: The original messages
            tool_results: Results from executing tools
            
        Returns:
            Updated message list with tool results
        """
        updated_messages = messages.copy()
        
        # Find the assistant message with tool calls
        for i, msg in enumerate(reversed(updated_messages)):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                # For each tool result, add a tool message
                for result in tool_results:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["result"]
                    }
                    updated_messages.append(tool_message)
                break
        
        return updated_messages
