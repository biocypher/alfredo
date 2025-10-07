"""Agent interface for executing tools."""

import re
from pathlib import Path
from typing import Any, Optional

from alfredo.prompts.builder import PromptBuilder
from alfredo.tools.base import BaseToolHandler, ToolResult, ToolUse
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily


class Agent:
    """An AI agent that can execute tools.

    This class provides the core functionality for:
    - Building system prompts with tool definitions
    - Parsing tool invocations from model output
    - Executing tools and returning results
    - Managing working directory context
    """

    def __init__(
        self,
        cwd: Optional[str] = None,
        model_family: ModelFamily = ModelFamily.GENERIC,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            cwd: Working directory for file operations (defaults to current directory)
            model_family: Which model family to optimize prompts for
            context: Optional context dictionary for conditional tool availability
        """
        self.cwd = cwd or str(Path.cwd())
        self.model_family = model_family
        self.context = context or {}
        self.prompt_builder = PromptBuilder(self.cwd, model_family, context)

        # Import all handlers to ensure they're registered
        self._import_handlers()

    def _import_handlers(self) -> None:
        """Import all handler modules to ensure tools are registered."""
        # Import to trigger registration
        from alfredo.tools.handlers import (  # noqa: F401
            code_analysis,
            command,
            discovery,
            file_ops,
            web,
            workflow,
        )

    def get_system_prompt(self, tool_ids: Optional[list[str]] = None, include_examples: bool = False) -> str:
        """Get the system prompt for the model.

        Args:
            tool_ids: Optional list of specific tool IDs to include
            include_examples: Whether to include usage examples

        Returns:
            Formatted system prompt string
        """
        prompt = self.prompt_builder.build_system_prompt(tool_ids)

        if include_examples:
            prompt += "\n\n" + self.prompt_builder.build_tool_use_example()

        return prompt

    def get_available_tools(self) -> list[str]:
        """Get list of available tool IDs.

        Returns:
            List of tool IDs
        """
        return self.prompt_builder.get_available_tools()

    def parse_tool_use(self, text: str) -> Optional[ToolUse]:
        """Parse a tool invocation from model output.

        Expected format:
        <tool_name>
        <param1>value1</param1>
        <param2>value2</param2>
        </tool_name>

        Args:
            text: The model's output text

        Returns:
            ToolUse object if a valid tool invocation is found, None otherwise
        """
        # Find tool invocation pattern
        # Pattern: <tool_name>...</tool_name>
        tool_pattern = r"<(\w+)>(.*?)</\1>"
        match = re.search(tool_pattern, text, re.DOTALL)

        if not match:
            return None

        tool_name = match.group(1)
        content = match.group(2)

        # Parse parameters
        params = {}
        param_pattern = r"<(\w+)>(.*?)</\1>"

        for param_match in re.finditer(param_pattern, content, re.DOTALL):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            params[param_name] = param_value

        return ToolUse(name=tool_name, params=params)

    def execute_tool(self, tool_use: ToolUse) -> ToolResult:
        """Execute a tool with the given parameters.

        Args:
            tool_use: The tool invocation to execute

        Returns:
            ToolResult with execution outcome
        """
        # Get handler class
        handler_class = registry.get_handler(tool_use.name)

        if handler_class is None:
            return ToolResult.err(f"Unknown tool: {tool_use.name}")

        # Instantiate and execute
        try:
            handler: BaseToolHandler = handler_class(cwd=self.cwd)
            result: ToolResult = handler.execute(tool_use.params)
        except Exception as e:
            return ToolResult.err(f"Error executing {tool_use.name}: {e}")
        else:
            return result

    def execute_from_text(self, text: str) -> Optional[ToolResult]:
        """Parse and execute a tool from model output text.

        Args:
            text: The model's output text

        Returns:
            ToolResult if a tool was found and executed, None otherwise
        """
        tool_use = self.parse_tool_use(text)

        if tool_use is None:
            return None

        return self.execute_tool(tool_use)
