"""Native OpenAI integration for Alfredo tools.

This module provides direct OpenAI API integration without LangChain dependency.
Converts Alfredo tools to OpenAI's native function calling format.
"""

import json
from typing import Any, Optional

from alfredo.tools.base import ToolResult
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolSpec

# Check if openai is available
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = object  # type: ignore[assignment,misc]


def check_openai_available() -> None:
    """Check if OpenAI is available and raise error if not."""
    if not OPENAI_AVAILABLE:
        msg = "OpenAI is not installed. Install it with: uv add openai"
        raise ImportError(msg)


def tool_spec_to_openai_format(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec to OpenAI's function calling format.

    Args:
        spec: The Alfredo tool specification

    Returns:
        Dictionary in OpenAI's tools format

    Example:
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read"
                        }
                    },
                    "required": ["path"]
                },
                "strict": True
            }
        }
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in spec.parameters:
        # Build property definition
        properties[param.name] = {
            "type": "string",  # Default to string for simplicity
            "description": param.instruction,
        }

        # Track required parameters
        if param.required:
            required.append(param.name)

    return {
        "type": "function",
        "function": {
            "name": spec.id,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "strict": True,
        },
    }


def get_all_tools_openai_format(
    cwd: Optional[str] = None,
    model_family: ModelFamily = ModelFamily.OPENAI,
    tool_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Get all registered tools in OpenAI format.

    Args:
        cwd: Working directory (not used for format conversion, but kept for API consistency)
        model_family: Model family for tool variant selection
        tool_ids: Optional list of specific tool IDs to include

    Returns:
        List of tool definitions in OpenAI format
    """
    # Import handlers to ensure they're registered
    from alfredo.tools.handlers import (  # noqa: F401
        code_analysis,
        command,
        discovery,
        file_ops,
        vision,
        web,
        workflow,
    )

    # Get specs
    if tool_ids:
        specs = [registry.get_spec(tool_id, model_family) for tool_id in tool_ids]
        # Filter out None values
        filtered_specs = [s for s in specs if s is not None]
    else:
        filtered_specs = registry.get_specs_for_variant(model_family)

    # Convert to OpenAI format
    return [tool_spec_to_openai_format(spec) for spec in filtered_specs]


class OpenAIAgent:
    """Agent for direct OpenAI API integration with Alfredo tools.

    This provides native OpenAI function calling without LangChain dependency.
    For most use cases, prefer the LangGraph agentic scaffold which provides
    more sophisticated planning and verification capabilities.
    """

    def __init__(
        self,
        cwd: Optional[str] = ".",
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        model_family: ModelFamily = ModelFamily.OPENAI,
    ) -> None:
        """Initialize the OpenAI agent.

        Args:
            cwd: Working directory for file operations
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model name to use
            model_family: Model family for tool variant selection
        """
        check_openai_available()

        self.cwd = cwd or "."
        self.model = model
        self.model_family = model_family
        self.client = OpenAI(api_key=api_key)

        # Import handlers to ensure registration
        from alfredo.tools.handlers import (  # noqa: F401
            code_analysis,
            command,
            discovery,
            file_ops,
            vision,
            web,
            workflow,
        )

    def get_tools_definition(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI format.

        Returns:
            List of tool definitions for OpenAI API
        """
        return get_all_tools_openai_format(cwd=self.cwd, model_family=self.model_family)

    def run(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """Run a conversation with tool calling support.

        Args:
            message: User message to process
            system_prompt: Optional system prompt (default provides basic instructions)
            max_iterations: Maximum number of tool calling iterations

        Returns:
            Dictionary with 'content' (final response) and 'tool_results' (list of tool executions)
        """
        # Build messages
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": (
                    f"You are a helpful AI assistant with access to tools for file operations, "
                    f"command execution, and more. Working directory: {self.cwd}"
                ),
            })

        messages.append({"role": "user", "content": message})

        # Get tools
        tools = self.get_tools_definition()

        # Track tool executions
        tool_results: list[dict[str, Any]] = []

        # Conversation loop
        for _iteration in range(max_iterations):
            # Call OpenAI API
            response = self.client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message

            # Add assistant message to history
            messages.append(response_message.model_dump())

            # Check if there are tool calls
            if not response_message.tool_calls:
                # No more tool calls, return final response
                return {
                    "content": response_message.content or "",
                    "tool_results": tool_results,
                }

            # Execute each tool call
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                result = self._execute_tool_call(tool_name, tool_args)

                # Track execution
                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                })

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result.output if result.success else f"Error: {result.error}",
                })

        # Max iterations reached
        return {
            "content": "Maximum iterations reached without completion",
            "tool_results": tool_results,
        }

    def _execute_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> ToolResult:
        """Execute a tool call using Alfredo's tool system.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            ToolResult with execution outcome
        """
        # Get handler
        handler_class = registry.get_handler(tool_name)

        if handler_class is None:
            return ToolResult.err(f"Unknown tool: {tool_name}")

        # Execute
        try:
            handler = handler_class(cwd=self.cwd)
            result: ToolResult = handler.execute(tool_args)
        except Exception as e:
            return ToolResult.err(f"Error executing {tool_name}: {e}")
        else:
            return result
