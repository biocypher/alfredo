"""Utilities for extracting tool handler source code for code-act framework.

This module provides functions to extract the implementation code of tool handlers
and format them for injection into system prompts, enabling a code-act framework
where the model can see and use the actual tool implementations.
"""

import inspect
from typing import Any, Optional

from alfredo.tools.registry import registry


def extract_tool_handler_code(tool_id: str) -> Optional[str]:
    """Extract the source code of a tool handler.

    Args:
        tool_id: The ID of the tool to extract code for

    Returns:
        The source code as a string, or None if handler not found
    """
    handler_class = registry.get_handler(tool_id)
    if handler_class is None:
        return None

    try:
        # Get the source code of the handler class
        source_code = inspect.getsource(handler_class)
        return source_code
    except (OSError, TypeError):
        # Can't get source for built-in or dynamically created classes
        return None


def extract_alfredo_tool_code(tools: list[Any], tool_ids: Optional[list[str]] = None) -> dict[str, str]:
    """Extract source code for specified Alfredo tools in the tool list.

    Args:
        tools: List of tools (can be AlfredoTool or regular StructuredTool)
        tool_ids: Optional list of specific tool IDs to extract code for.
                  If None, extracts code for all Alfredo tools.

    Returns:
        Dictionary mapping tool IDs to their source code
    """
    from alfredo.tools.alfredo_tool import AlfredoTool

    tool_code: dict[str, str] = {}

    # Get all registered Alfredo tool IDs
    alfredo_tool_ids = set(registry.get_all_tool_ids())

    # Create a set of requested tool IDs (if specified)
    # Note: Use 'is not None' to distinguish between None and empty list
    requested_ids = set(tool_ids) if tool_ids is not None else None

    for tool in tools:
        # Get the tool name
        if isinstance(tool, AlfredoTool):
            tool_name = tool.name
        else:
            tool_name = getattr(tool, "name", None)

        if tool_name is None:
            continue

        # Only extract code for Alfredo tools (not MCP or other external tools)
        if tool_name in alfredo_tool_ids:
            # If specific tool IDs requested, only extract those
            if requested_ids is not None and tool_name not in requested_ids:
                continue

            code = extract_tool_handler_code(tool_name)
            if code:
                tool_code[tool_name] = code

    return tool_code


def format_tool_code_section(tools: list[Any], tool_ids: Optional[list[str]] = None) -> str:
    """Format tool implementation code for injection into system prompts.

    Args:
        tools: List of tools to extract code from
        tool_ids: Optional list of specific tool IDs to extract code for.
                  If None, extracts code for all Alfredo tools.

    Returns:
        Formatted string containing tool implementations
    """
    tool_code = extract_alfredo_tool_code(tools, tool_ids=tool_ids)

    if not tool_code:
        return ""

    # Build the formatted section
    lines = [
        "# Tool Implementations",
        "",
        "Below are the full implementations of the available tools. You can use these",
        "functions directly in your scripts by importing them from the alfredo.tools.handlers module.",
        "",
        "**Code-Act Mode**: You can write Python scripts that call multiple tools by importing",
        "and using these handler classes. Each handler has an `execute(params)` method that",
        "takes a dictionary of parameters and returns a ToolResult object.",
        "",
        "**Example Usage**:",
        "```python",
        "from alfredo.tools.handlers.file_ops import ReadFileHandler",
        "from pathlib import Path",
        "",
        "# Create handler instance",
        "handler = ReadFileHandler(cwd='.')",
        "",
        "# Execute the tool",
        "result = handler.execute({'path': 'config.json'})",
        "if result.success:",
        "    print(result.output)",
        "else:",
        "    print(f'Error: {result.error}')",
        "```",
        "",
        "---",
        "",
    ]

    # Add each tool's source code
    for tool_id, code in sorted(tool_code.items()):
        lines.append(f"## Tool: {tool_id}")
        lines.append("")
        lines.append("```python")
        lines.append(code.rstrip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
