"""Tool registry for managing available tools and their variants."""

import threading
from typing import Optional

from alfredo.tools.specs import ModelFamily, ToolSpec


class ToolRegistry:
    """Registry for managing tool specifications and handlers.

    This class maintains a mapping of tools by model family and provides
    methods to register, retrieve, and query available tools. It uses a
    singleton pattern to ensure a single global registry.
    """

    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ToolRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry if not already initialized."""
        if not getattr(self, "_initialized", False):
            self._tools: dict[ModelFamily, dict[str, ToolSpec]] = {}
            self._handlers: dict[str, type] = {}
            self._initialized: bool = True

    def register_spec(self, spec: ToolSpec) -> None:
        """Register a tool specification.

        Args:
            spec: The tool specification to register
        """
        if spec.variant not in self._tools:
            self._tools[spec.variant] = {}

        self._tools[spec.variant][spec.id] = spec

    def register_handler(self, tool_id: str, handler_class: type) -> None:
        """Register a handler class for a tool.

        Args:
            tool_id: The tool ID this handler implements
            handler_class: The handler class (should inherit from BaseToolHandler)
        """
        self._handlers[tool_id] = handler_class

    def get_spec(self, tool_id: str, variant: ModelFamily = ModelFamily.GENERIC) -> Optional[ToolSpec]:
        """Get a tool specification by ID and variant.

        Args:
            tool_id: The tool ID to look up
            variant: The model family variant (defaults to GENERIC)

        Returns:
            The tool specification if found, None otherwise
        """
        # Try exact variant first
        if variant in self._tools and tool_id in self._tools[variant]:
            return self._tools[variant][tool_id]

        # Fallback to GENERIC
        if ModelFamily.GENERIC in self._tools and tool_id in self._tools[ModelFamily.GENERIC]:
            return self._tools[ModelFamily.GENERIC][tool_id]

        return None

    def get_specs_for_variant(self, variant: ModelFamily = ModelFamily.GENERIC) -> list[ToolSpec]:
        """Get all tool specifications for a specific variant.

        If no tools are registered for the variant, returns GENERIC tools.

        Args:
            variant: The model family variant

        Returns:
            List of tool specifications
        """
        if variant in self._tools:
            return list(self._tools[variant].values())

        # Fallback to GENERIC
        if ModelFamily.GENERIC in self._tools:
            return list(self._tools[ModelFamily.GENERIC].values())

        return []

    def get_handler(self, tool_id: str) -> Optional[type]:
        """Get the handler class for a tool.

        Args:
            tool_id: The tool ID

        Returns:
            The handler class if registered, None otherwise
        """
        return self._handlers.get(tool_id)

    def get_all_tool_ids(self) -> list[str]:
        """Get a list of all registered tool IDs across all variants.

        Returns:
            List of unique tool IDs
        """
        tool_ids: set[str] = set()
        for tools in self._tools.values():
            tool_ids.update(tools.keys())
        return sorted(tool_ids)

    def clear(self) -> None:
        """Clear all registered tools and handlers (mainly for testing)."""
        self._tools.clear()
        self._handlers.clear()


# Global registry instance
registry = ToolRegistry()
