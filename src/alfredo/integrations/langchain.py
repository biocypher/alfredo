"""LangChain integration for Alfredo tools.

This module provides adapters to convert Alfredo tools into LangChain-compatible tools
that can be used with LangChain agents and LangGraph workflows.
"""

from typing import Any, Optional

try:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    StructuredTool = object  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment,misc]

from alfredo.tools.base import ToolResult
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolSpec


def check_langchain_available() -> None:
    """Check if LangChain is available and raise error if not."""
    if not LANGCHAIN_AVAILABLE:
        msg = "LangChain is not installed. Install it with: uv add langchain-core"
        raise ImportError(msg)


def create_pydantic_model_from_spec(spec: ToolSpec) -> type[BaseModel]:
    """Create a Pydantic model from a ToolSpec for LangChain args_schema.

    Args:
        spec: The tool specification

    Returns:
        Pydantic BaseModel class representing the tool's parameters
    """
    check_langchain_available()

    # Build field definitions for Pydantic model
    field_definitions: dict[str, Any] = {}

    for param in spec.parameters:
        # Determine the type (default to str)
        field_type = str

        # Create field with description
        if param.required:
            field_definitions[param.name] = (
                field_type,
                Field(..., description=param.instruction),
            )
        else:
            field_definitions[param.name] = (
                Optional[field_type],
                Field(None, description=param.instruction),
            )

    # Create the Pydantic model dynamically
    model = create_model(
        f"{spec.id}_input",
        **field_definitions,
    )

    return model


def create_langchain_tool(
    tool_id: str,
    cwd: Optional[str] = None,
    model_family: ModelFamily = ModelFamily.GENERIC,
) -> StructuredTool:
    """Create a LangChain StructuredTool from an Alfredo tool.

    Args:
        tool_id: The ID of the Alfredo tool to convert
        cwd: Working directory for file operations
        model_family: Model family for tool variant selection

    Returns:
        LangChain StructuredTool instance

    Raises:
        ImportError: If LangChain is not installed
        ValueError: If tool is not found in registry
    """
    check_langchain_available()

    # Get tool spec and handler
    spec = registry.get_spec(tool_id, model_family)
    if spec is None:
        msg = f"Tool '{tool_id}' not found in registry"
        raise ValueError(msg)

    handler_class = registry.get_handler(tool_id)
    if handler_class is None:
        msg = f"Handler for tool '{tool_id}' not found in registry"
        raise ValueError(msg)

    # Create Pydantic model for args
    args_schema = create_pydantic_model_from_spec(spec)

    # Create the function that LangChain will call
    def tool_func(**kwargs: Any) -> str:
        """Execute the Alfredo tool."""
        handler = handler_class(cwd=cwd)
        result: ToolResult = handler.execute(kwargs)

        if result.success:
            return result.output
        else:
            # LangChain expects exceptions for errors in some contexts
            # But we return error message for graceful handling
            return f"Error: {result.error}"

    # Create the LangChain tool
    lc_tool = StructuredTool.from_function(
        func=tool_func,
        name=spec.id,
        description=spec.description,
        args_schema=args_schema,
    )

    return lc_tool


def create_all_langchain_tools(
    cwd: Optional[str] = None,
    model_family: ModelFamily = ModelFamily.GENERIC,
    tool_ids: Optional[list[str]] = None,
) -> list[StructuredTool]:
    """Create LangChain tools for all registered Alfredo tools.

    Args:
        cwd: Working directory for file operations
        model_family: Model family for tool variant selection
        tool_ids: Optional list of specific tool IDs to convert. If None, converts all tools.

    Returns:
        List of LangChain StructuredTool instances

    Raises:
        ImportError: If LangChain is not installed
    """
    check_langchain_available()

    # Import handlers to ensure they're registered
    from alfredo.tools.handlers import command, discovery, file_ops, workflow  # noqa: F401

    if tool_ids is None:
        # Get all tools for the model family
        specs = registry.get_specs_for_variant(model_family)
        tool_ids = [spec.id for spec in specs]

    tools = []
    for tool_id in tool_ids:
        try:
            tool = create_langchain_tool(tool_id, cwd=cwd, model_family=model_family)
            tools.append(tool)
        except ValueError as e:
            # Skip tools that don't exist
            print(f"Warning: {e}")
            continue

    return tools


# Convenience function to create individual tools using decorator pattern
def as_langchain_tool(
    tool_id: str,
    cwd: Optional[str] = None,
    model_family: ModelFamily = ModelFamily.GENERIC,
) -> Any:
    """Decorator to convert an Alfredo tool handler into a LangChain tool function.

    This is useful when you want to use the @tool decorator pattern with Alfredo tools.

    Args:
        tool_id: The ID of the Alfredo tool
        cwd: Working directory for file operations
        model_family: Model family for tool variant selection

    Returns:
        Decorator function

    Example:
        ```python
        from alfredo.integrations.langchain import as_langchain_tool
        from langchain_core.tools import tool

        @as_langchain_tool("read_file")
        def read_file_tool(path: str) -> str:
            '''Read a file from the filesystem.'''
            pass
        ```
    """
    check_langchain_available()

    def decorator(func: Any) -> Any:
        # Get the handler
        handler_class = registry.get_handler(tool_id)
        if handler_class is None:
            msg = f"Handler for tool '{tool_id}' not found"
            raise ValueError(msg)

        def wrapper(**kwargs: Any) -> str:
            handler = handler_class(cwd=cwd)
            result: ToolResult = handler.execute(kwargs)

            if result.success:
                return result.output
            return f"Error: {result.error}"

        # Copy metadata from original function
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

    return decorator
