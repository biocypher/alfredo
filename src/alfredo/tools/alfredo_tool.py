"""AlfredoTool: Unified tool wrapper with node-specific system prompt instructions.

This module provides the AlfredoTool class, which wraps LangChain StructuredTools and adds:
- Node-specific system prompt instructions
- Metadata support
- Unified interface for Alfredo, MCP, and LangChain tools

Example:
    >>> # Create tool with different instructions for different nodes
    >>> tool = AlfredoTool.from_alfredo(
    ...     tool_id="write_todo_list",
    ...     cwd=".",
    ...     system_instructions={
    ...         "agent": "Use this to track sequential progress",
    ...         "planner": "Create checklist after making plan"
    ...     }
    ... )
    >>> tool.get_instruction_for_node("agent")
    'Use this to track sequential progress'
"""

from typing import Any, Optional

try:
    from langchain_core.tools import StructuredTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    StructuredTool = object  # type: ignore[assignment,misc]


class AlfredoTool:
    """Wrapper for tools with node-specific system prompt instructions.

    This class wraps a LangChain StructuredTool and adds the ability to define
    different system prompt instructions for different graph nodes.

    Attributes:
        langchain_tool: The underlying LangChain StructuredTool
        system_instructions: Mapping of node names to their specific instructions
        metadata: Optional additional metadata
    """

    def __init__(
        self,
        langchain_tool: StructuredTool,
        system_instructions: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize an AlfredoTool.

        Args:
            langchain_tool: The underlying LangChain StructuredTool
            system_instructions: Optional mapping of node names to instructions.
                Example: {
                    "agent": "Use this to track progress sequentially",
                    "planner": "Create checklist after making plan",
                    "verifier": "Check if all tasks completed"
                }
            metadata: Optional additional metadata

        Raises:
            ImportError: If LangChain is not available
            TypeError: If langchain_tool is not a StructuredTool
        """
        if not LANGCHAIN_AVAILABLE:
            msg = "LangChain is not installed. Install it with: uv add langchain-core"
            raise ImportError(msg)

        if not isinstance(langchain_tool, StructuredTool):
            msg = f"langchain_tool must be a StructuredTool, got {type(langchain_tool)}"
            raise TypeError(msg)

        self._langchain_tool = langchain_tool
        self._system_instructions = system_instructions or {}
        self._metadata = metadata or {}

    @property
    def langchain_tool(self) -> StructuredTool:
        """Get the underlying LangChain StructuredTool."""
        return self._langchain_tool

    @property
    def system_instructions(self) -> dict[str, str]:
        """Get the system instructions mapping."""
        return self._system_instructions

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the metadata dictionary."""
        return self._metadata

    @property
    def name(self) -> str:
        """Get the tool name (proxied from langchain_tool)."""
        return self._langchain_tool.name

    @property
    def description(self) -> str:
        """Get the tool description (proxied from langchain_tool)."""
        return self._langchain_tool.description

    def get_instruction_for_node(self, node_name: str) -> Optional[str]:
        """Get the system prompt instruction for a specific node.

        Args:
            node_name: Name of the graph node (e.g., "agent", "planner", "verifier")

        Returns:
            The instruction string for this node, or None if not targeted
        """
        return self._system_instructions.get(node_name)

    def is_available_for_node(self, node_name: str) -> bool:
        """Check if this tool has an instruction for a specific node.

        Args:
            node_name: Name of the graph node

        Returns:
            True if the tool has an instruction for this node, False otherwise
        """
        return node_name in self._system_instructions

    def get_target_nodes(self) -> list[str]:
        """Get all nodes this tool targets with specific instructions.

        Returns:
            List of node names that have specific instructions
        """
        return list(self._system_instructions.keys())

    def to_langchain_tool(self) -> StructuredTool:
        """Extract the underlying LangChain StructuredTool.

        This allows AlfredoTool to be used anywhere a LangChain tool is expected.

        Returns:
            The underlying StructuredTool
        """
        return self._langchain_tool

    @classmethod
    def from_alfredo(
        cls,
        tool_id: str,
        cwd: Optional[str] = None,
        system_instructions: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "AlfredoTool":
        """Create an AlfredoTool from an Alfredo tool ID.

        Args:
            tool_id: The ID of the Alfredo tool (e.g., "read_file", "write_todo_list")
            cwd: Working directory for file operations
            system_instructions: Optional node-specific instructions
            metadata: Optional additional metadata

        Returns:
            AlfredoTool instance wrapping the Alfredo tool

        Raises:
            ImportError: If LangChain is not available
            ValueError: If tool_id is not found in registry

        Example:
            >>> tool = AlfredoTool.from_alfredo(
            ...     tool_id="write_todo_list",
            ...     cwd=".",
            ...     system_instructions={
            ...         "agent": "Track your progress sequentially",
            ...         "planner": "Create initial checklist"
            ...     }
            ... )
        """
        from alfredo.integrations.langchain import create_langchain_tool

        # Create the underlying LangChain tool
        lc_tool = create_langchain_tool(tool_id=tool_id, cwd=cwd)

        # Wrap it as an AlfredoTool
        return cls(
            langchain_tool=lc_tool,
            system_instructions=system_instructions,
            metadata=metadata,
        )

    @classmethod
    def from_mcp(
        cls,
        mcp_tool: StructuredTool,
        system_instructions: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "AlfredoTool":
        """Create an AlfredoTool from an MCP tool.

        Args:
            mcp_tool: The MCP StructuredTool (from langchain-mcp-adapters)
            system_instructions: Optional node-specific instructions
            metadata: Optional additional metadata

        Returns:
            AlfredoTool instance wrapping the MCP tool

        Example:
            >>> from alfredo.integrations.mcp import load_mcp_tools_sync
            >>> mcp_tools = load_mcp_tools_sync(server_configs)
            >>> wrapped = AlfredoTool.from_mcp(
            ...     mcp_tools[0],
            ...     system_instructions={
            ...         "agent": "Use for external file operations"
            ...     }
            ... )
        """
        return cls(
            langchain_tool=mcp_tool,
            system_instructions=system_instructions,
            metadata=metadata,
        )

    @classmethod
    def from_langchain(
        cls,
        langchain_tool: StructuredTool,
        system_instructions: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "AlfredoTool":
        """Create an AlfredoTool from any LangChain StructuredTool.

        This allows you to wrap any LangChain tool (custom or third-party) as an
        AlfredoTool to add node-specific instructions.

        Args:
            langchain_tool: Any LangChain StructuredTool
            system_instructions: Optional node-specific instructions
            metadata: Optional additional metadata

        Returns:
            AlfredoTool instance wrapping the LangChain tool

        Example:
            >>> from langchain_core.tools import StructuredTool
            >>> custom_tool = StructuredTool.from_function(...)
            >>> wrapped = AlfredoTool.from_langchain(
            ...     custom_tool,
            ...     system_instructions={"agent": "Use carefully"}
            ... )
        """
        return cls(
            langchain_tool=langchain_tool,
            system_instructions=system_instructions,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """String representation of the AlfredoTool."""
        target_nodes = self.get_target_nodes()
        targets_str = f"targets={target_nodes}" if target_nodes else "no targets"
        return f"AlfredoTool(name={self.name!r}, {targets_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"AlfredoTool({self.name})"
