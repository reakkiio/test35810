"""
conversation.py

This module provides a modern conversation manager for handling chat-based interactions, message history, tool calls, and robust error handling. It defines the Conversation class and supporting types for managing conversational state, tool integration, and message validation.

Classes:
    ConversationError: Base exception for conversation-related errors.
    ToolCallError: Raised when there's an error with tool calls.
    MessageValidationError: Raised when message validation fails.
    Message: Represents a single message in the conversation.
    FunctionCall: TypedDict for a function call.
    ToolDefinition: TypedDict for a tool definition.
    FunctionCallData: TypedDict for function call data.
    Fn: Represents a function (tool) that the agent can call.
    Conversation: Main conversation manager class.

Functions:
    tools: Decorator to mark a function as a tool.
"""
import os
import json
from typing import Optional, Dict, List, Any, TypedDict, Callable, TypeVar, Union
from dataclasses import dataclass
from datetime import datetime

T = TypeVar('T')

class ConversationError(Exception):
    """Base exception for conversation-related errors."""
    pass

class ToolCallError(ConversationError):
    """Raised when there's an error with tool calls."""
    pass

class MessageValidationError(ConversationError):
    """Raised when message validation fails."""
    pass

@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str
    content: str
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FunctionCall(TypedDict):
    """Type for a function call."""
    name: str
    arguments: Dict[str, Any]

class ToolDefinition(TypedDict):
    """Type for a tool definition."""
    type: str
    function: Dict[str, Any]

class FunctionCallData(TypedDict, total=False):
    """Type for function call data"""
    tool_calls: List[FunctionCall]
    error: str

class Fn:
    """Represents a function (tool) that the agent can call."""
    def __init__(self, name: str, description: str, parameters: Dict[str, str]) -> None:
        self.name: str = name
        self.description: str = description
        self.parameters: Dict[str, str] = parameters

def tools(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to mark a function as a tool."""
    func._is_tool = True  # type: ignore
    return func

class Conversation:
    """
    Modern conversation manager with enhanced features.

    Key Features:
        - Robust message handling with metadata
        - Enhanced tool calling support
        - Efficient history management
        - Improved error handling
        - Memory optimization
    """

    intro = (
        "You're a helpful Large Language Model assistant. "
        "Respond directly to the user's questions or use tools when appropriate."
    )

    def __init__(
        self,
        status: bool = True,
        max_tokens: int = 600,
        filepath: Optional[str] = None,
        update_file: bool = True,
        tools: Optional[List[Fn]] = None,
        compression_threshold: int = 10000,
    ):
        """Initialize conversation manager with modern features."""
        self.status = status
        self.max_tokens_to_sample = max_tokens
        self.messages: List[Message] = []
        self.history_format = "\nUser: %(user)s\nAssistant: %(llm)s"
        self.tool_history_format = "\nUser: %(user)s\nAssistant: <tool_call>%(tool_json)s</tool_call>\nTool: %(result)s"
        self.file = filepath
        self.update_file = update_file
        self.history_offset = 10250
        self.prompt_allowance = 10
        self.tools = tools or []
        self.compression_threshold = compression_threshold
        if filepath:
            self.load_conversation(filepath, True)

    def load_conversation(self, filepath: str, exists: bool = True) -> None:
        """Load conversation with improved error handling."""
        try:
            if not isinstance(filepath, str):
                raise TypeError(f"Filepath must be str, not {type(filepath)}")
            
            if exists and not os.path.isfile(filepath):
                raise FileNotFoundError(f"File '{filepath}' does not exist")

            if not os.path.isfile(filepath):
                with open(filepath, "w", encoding="utf-8") as fh:
                    fh.write(self.intro)
            else:
                with open(filepath, encoding="utf-8") as fh:
                    file_contents = fh.readlines()
                    if file_contents:
                        self.intro = file_contents[0]
                        self._process_history_from_file(file_contents[1:])
        except Exception as e:
            raise ConversationError(f"Failed to load conversation: {str(e)}") from e

    def _process_history_from_file(self, lines: List[str]) -> None:
        """Process and structure conversation history from file."""
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(("User:", "Assistant:", "Tool:")):
                if current_role and current_content:
                    self.messages.append(Message(
                        role=current_role,
                        content="\n".join(current_content)
                    ))
                    current_content = []
                current_role = line.split(":")[0].lower()
                content = ":".join(line.split(":")[1:]).strip()
                current_content.append(content)
            elif line:
                current_content.append(line)
                
        if current_role and current_content:
            self.messages.append(Message(
                role=current_role,
                content="\n".join(current_content)
            ))

    def _compress_history(self) -> None:
        """Delete old history when it exceeds threshold."""
        if len(self.messages) > self.compression_threshold:
            # Remove oldest messages, keep only the most recent ones
            self.messages = self.messages[-self.compression_threshold:]

    # _summarize_messages removed

    def gen_complete_prompt(self, prompt: str, intro: Optional[str] = None) -> str:
        """Generate complete prompt with enhanced context management."""
        if not self.status:
            return prompt

        intro = intro or self.intro or ""
        
        # Add tool information if available
        tools_description = self.get_tools_description()
        if tools_description:
            try:
                date_str = f"Current date: {datetime.now().strftime('%d %b %Y')}"
            except:
                date_str = ""
                
            intro = self._generate_enhanced_intro(intro, tools_description, date_str)

        # Generate history string with proper formatting
        history = self._generate_history_string()
        
        # Combine and trim if needed
        complete_prompt = intro + self._trim_chat_history(
            history + "\nUser: " + prompt + "\nAssistant:",
            intro
        )
        
        return complete_prompt

    def _generate_enhanced_intro(self, intro: str, tools_description: str, date_str: str) -> str:
        """Generate enhanced introduction with tools and guidelines."""
        return f'''
{intro}

{date_str}

**CORE PROTOCOL:**

Your goal is to assist the user effectively. Analyze each query and choose one of two response modes:

**1. Tool Mode:**
   - **When:** If the query requires external data, calculations, or functions listed under AVAILABLE TOOLS.
   - **Action:** Output *ONLY* the complete JSON tool call within tags.
   - **Format:** Must start with `<tool_call>` and end with `</tool_call>`.

**2. Conversational Mode:**
   - **When:** For queries answerable with internal knowledge.
   - **Action:** Respond directly and concisely.

**AVAILABLE TOOLS:**
{tools_description}

**TOOL FORMAT:**
<tool_call>
{{
    "name": "tool_name",
    "arguments": {{
        "param": "value"
    }}
}}
</tool_call>
'''

    def _generate_history_string(self) -> str:
        """Generate formatted history string from messages."""
        history_parts = []
        for msg in self.messages:
            if msg.role == "system" and msg.metadata.get("summarized_count"):
                history_parts.append(f"[Previous messages summarized: {msg.metadata['summarized_count']}]")
            else:
                role_display = msg.role.capitalize()
                if "<tool_call>" in msg.content:
                    history_parts.append(f"{role_display}: {msg.content}")
                else:
                    history_parts.append(f"{role_display}: {msg.content}")
        return "\n".join(history_parts)

    def _trim_chat_history(self, chat_history: str, intro: str) -> str:
        """Trim chat history with improved token management."""
        intro = intro or ""
        total_length = len(intro) + len(chat_history)
        
        if total_length > self.history_offset:
            truncate_at = (total_length - self.history_offset) + self.prompt_allowance
            # Try to truncate at a message boundary
            lines = chat_history[truncate_at:].split('\n')
            for i, line in enumerate(lines):
                if line.startswith(("User:", "Assistant:", "Tool:")):
                    return "... " + "\n".join(lines[i:])
            return "... " + chat_history[truncate_at:]
        return chat_history

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message with enhanced validation and metadata support. Deletes oldest messages if total word count exceeds max_tokens_to_sample."""
        try:
            role = role.lower()  # Normalize role to lowercase
            if not self.validate_message(role, content):
                raise MessageValidationError("Invalid message role or content")

            # Calculate total word count in history
            def total_word_count(messages):
                return sum(len(msg.content.split()) for msg in messages)

            # Remove oldest messages until total word count is below limit
            temp_messages = self.messages.copy()
            while temp_messages and (total_word_count(temp_messages) + len(content.split()) > self.max_tokens_to_sample):
                temp_messages.pop(0)

            self.messages = temp_messages

            message = Message(role=role, content=content, metadata=metadata or {})
            self.messages.append(message)

            if self.file and self.update_file:
                self._append_to_file(message)

            self._compress_history()

        except Exception as e:
            raise ConversationError(f"Failed to add message: {str(e)}") from e

    def _append_to_file(self, message: Message) -> None:
        """Append message to file with error handling."""
        try:
            if not os.path.exists(self.file):
                with open(self.file, "w", encoding="utf-8") as fh:
                    fh.write(self.intro + "\n")

            with open(self.file, "a", encoding="utf-8") as fh:
                role_display = message.role.capitalize()
                fh.write(f"\n{role_display}: {message.content}")
                
        except Exception as e:
            raise ConversationError(f"Failed to write to file: {str(e)}") from e

    def validate_message(self, role: str, content: str) -> bool:
        """Validate message with enhanced role checking."""
        valid_roles = {'user', 'assistant', 'tool', 'system'}
        if role not in valid_roles:
            return False
        if not isinstance(content, str):
            return False
        # Allow empty content for assistant (needed for streaming)
        if not content and role != 'assistant':
            return False
        return True

    def handle_tool_response(self, response: str) -> Dict[str, Any]:
        """Process tool responses with enhanced error handling."""
        try:
            if "<tool_call>" in response:
                function_call_data = self._parse_function_call(response)
                
                if "error" in function_call_data:
                    return {
                        "is_tool_call": True,
                        "success": False,
                        "result": function_call_data["error"],
                        "original_response": response
                    }

                result = self.execute_function(function_call_data)
                self.add_message("tool", result)

                return {
                    "is_tool_call": True,
                    "success": True,
                    "result": result,
                    "tool_calls": function_call_data.get("tool_calls", []),
                    "original_response": response
                }

            return {
                "is_tool_call": False,
                "result": response,
                "original_response": response
            }
            
        except Exception as e:
            raise ToolCallError(f"Failed to handle tool response: {str(e)}") from e

    def _parse_function_call(self, response: str) -> FunctionCallData:
        """Parse function calls with improved JSON handling."""
        try:
            # Extract content between tool call tags
            start_tag = "<tool_call>"
            end_tag = "</tool_call>"
            start_idx = response.find(start_tag)
            end_idx = response.rfind(end_tag)

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No valid tool call tags found")

            json_str = response[start_idx + len(start_tag):end_idx].strip()
            
            # Handle both single and multiple tool calls
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return {"tool_calls": [parsed]}
                elif isinstance(parsed, list):
                    return {"tool_calls": parsed}
                else:
                    raise ValueError("Invalid tool call structure")
            except json.JSONDecodeError:
                # Try to extract valid JSON if embedded in other content
                import re
                json_pattern = re.search(r'\{[\s\S]*\}', json_str)
                if json_pattern:
                    parsed = json.loads(json_pattern.group(0))
                    return {"tool_calls": [parsed]}
                raise

        except Exception as e:
            return {"error": str(e)}

    def execute_function(self, function_call_data: FunctionCallData) -> str:
        """Execute functions with enhanced error handling."""
        try:
            tool_calls = function_call_data.get("tool_calls", [])
            if not tool_calls:
                raise ValueError("No tool calls provided")

            results = []
            for tool_call in tool_calls:
                name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                
                if not name or not isinstance(arguments, dict):
                    raise ValueError(f"Invalid tool call format: {tool_call}")
                
                # Execute the tool (implement actual logic here)
                results.append(f"Executed {name} with arguments {arguments}")

            return "; ".join(results)
            
        except Exception as e:
            raise ToolCallError(f"Failed to execute function: {str(e)}") from e

    def get_tools_description(self) -> str:
        """Get formatted tools description."""
        if not self.tools:
            return ""

        return "\n".join(
            f"- {fn.name}: {fn.description} (Parameters: {', '.join(f'{name}: {typ}' for name, typ in fn.parameters.items())})"
            for fn in self.tools
        )

    def update_chat_history(self, prompt: str, response: str) -> None:
        """Update chat history with a new prompt-response pair.
        
        Args:
            prompt: The user's prompt/question
            response: The assistant's response
            
        This method adds both the user's prompt and the assistant's response
        to the conversation history as separate messages.
        """
        # Add user's message (normalize role)
        self.add_message("user", prompt)
        
        # Add assistant's response (normalize role)
        self.add_message("assistant", response)

