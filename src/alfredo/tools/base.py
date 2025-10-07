"""Base classes and utilities for tool handlers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolUse:
    """Represents a tool invocation from the model.

    Attributes:
        name: The tool name/ID
        params: Dictionary of parameter values
    """

    name: str
    params: dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the tool executed successfully
        output: The output/result text
        error: Error message if success is False
    """

    success: bool
    output: str
    error: Optional[str] = None

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, output=output, error=None)

    @classmethod
    def err(cls, error_msg: str) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, output="", error=error_msg)


class ToolValidationError(Exception):
    """Raised when tool parameter validation fails."""

    pass


class BaseToolHandler(ABC):
    """Base class for all tool handlers.

    Tool handlers implement the actual execution logic for tools.
    Subclasses must implement the execute() method.
    """

    def __init__(self, cwd: Optional[str] = None) -> None:
        """Initialize the tool handler.

        Args:
            cwd: Current working directory for relative path resolution
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

    @property
    @abstractmethod
    def tool_id(self) -> str:
        """Return the tool ID this handler implements."""
        pass

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            params: Dictionary of parameter values

        Returns:
            ToolResult with the execution result
        """
        pass

    def validate_required_param(self, params: dict[str, Any], param_name: str) -> None:
        """Validate that a required parameter is present.

        Args:
            params: Parameter dictionary
            param_name: Name of the required parameter

        Raises:
            ToolValidationError: If the parameter is missing or empty
        """
        if param_name not in params or not params[param_name]:
            msg = f"Missing required parameter: {param_name}"
            raise ToolValidationError(msg)

    def resolve_path(self, path: str) -> Path:
        """Resolve a potentially relative path to an absolute path.

        Args:
            path: Path string (relative or absolute)

        Returns:
            Absolute Path object
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.cwd / path_obj).resolve()

    def is_path_safe(self, path: Path) -> bool:
        """Check if a path is within the working directory (basic safety check).

        Args:
            path: Path to check

        Returns:
            True if the path is within cwd, False otherwise
        """
        try:
            resolved = path.resolve()
            cwd_resolved = self.cwd.resolve()
            return str(resolved).startswith(str(cwd_resolved))
        except (ValueError, OSError):
            return False

    def get_relative_path(self, path: Path) -> str:
        """Get a path relative to the working directory.

        Args:
            path: Absolute path

        Returns:
            Relative path string
        """
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)


class AsyncToolHandler(BaseToolHandler):
    """Base class for asynchronous tool handlers.

    For tools that need to perform async operations (e.g., network requests),
    inherit from this class instead and implement execute_async().
    """

    @abstractmethod
    async def execute_async(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool asynchronously.

        Args:
            params: Dictionary of parameter values

        Returns:
            ToolResult with the execution result
        """
        pass

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Synchronous wrapper that raises an error.

        Use execute_async() for async handlers.
        """
        msg = "This is an async handler. Use execute_async() instead."
        raise NotImplementedError(msg)
