"""MCP (Model Context Protocol) integration for Alfredo tools.

This module provides utilities to load MCP tools and combine them with Alfredo tools
for use in the agentic scaffold.

MCP servers can be loaded from configuration dictionaries that specify the server
command, arguments, and transport mechanism.

Server Configuration Examples
-----------------------------

Local MCP Server (stdio):
    server_configs = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
            "transport": "stdio"
        }
    }

Remote MCP Server (HTTP):
    server_configs = {
        "api": {
            "transport": "streamable_http",
            "url": "https://api.example.com/mcp",
            "headers": {
                "Authorization": "Bearer your-token"
            }
        }
    }

Remote MCP Server (SSE):
    server_configs = {
        "weather": {
            "transport": "sse",
            "url": "https://weather.api.com/mcp",
            "headers": {
                "X-API-Key": "your-key"
            }
        }
    }
"""

from typing import Any, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MultiServerMCPClient = object

from alfredo.integrations.langchain import create_langchain_tools
from alfredo.tools.specs import ModelFamily


def check_mcp_available() -> None:
    """Check if langchain-mcp-adapters is available and raise error if not."""
    if not MCP_AVAILABLE:
        msg = "langchain-mcp-adapters is not installed. Install it with: uv add langchain-mcp-adapters"
        raise ImportError(msg)


def _make_sync_tool(async_tool: Any) -> Any:
    """Wrap an async MCP tool to make it synchronously callable.

    LangGraph's ToolNode tries to call tools synchronously, but langchain-mcp-adapters
    returns async-only tools that raise NotImplementedError. This wrapper makes them
    compatible by using the tool's ainvoke method in a thread pool.

    Args:
        async_tool: The async StructuredTool from MCP

    Returns:
        A sync-compatible StructuredTool
    """
    import asyncio
    import concurrent.futures

    from langchain_core.tools import StructuredTool

    # Use the tool's ainvoke method which properly handles async invocation
    async_invoke = async_tool.ainvoke

    # Create a sync wrapper that runs async code in a thread pool
    def sync_wrapper(**kwargs: Any) -> Any:
        """Synchronous wrapper that runs the async tool in a thread pool."""

        async def run_async() -> Any:
            """Async wrapper to invoke the tool."""
            return await async_invoke(kwargs)

        # Use a thread pool to run asyncio.run() to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, run_async())
            return future.result()

    # Create a new StructuredTool with the sync wrapper
    sync_tool = StructuredTool.from_function(
        func=sync_wrapper,
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
    )

    return sync_tool


async def load_mcp_tools(
    server_configs: dict[str, dict[str, Any]],
    make_sync: bool = True,
) -> list[Any]:
    """Load MCP tools from server configurations.

    Args:
        server_configs: Dictionary mapping server names to their configurations.
            Each configuration should have:
            - command: The command to run the MCP server
            - args: List of arguments for the command
            - transport: Transport type (e.g., "stdio")

            Example:
                {
                    "math": {
                        "command": "python",
                        "args": ["/path/to/math_server.py"],
                        "transport": "stdio"
                    }
                }
        make_sync: If True, wrap async tools to be sync-compatible with LangGraph (default: True)

    Returns:
        List of LangChain StructuredTool instances from the MCP servers

    Raises:
        ImportError: If langchain-mcp-adapters is not installed

    Example:
        ```python
        import asyncio
        from alfredo.integrations.mcp import load_mcp_tools

        async def main():
            server_configs = {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "transport": "stdio"
                }
            }

            mcp_tools = await load_mcp_tools(server_configs)
            print(f"Loaded {len(mcp_tools)} MCP tools")

        asyncio.run(main())
        ```
    """
    check_mcp_available()

    # Create MCP client with server configurations
    client = MultiServerMCPClient(server_configs)

    # Get tools from all configured servers
    tools = await client.get_tools()

    # Make tools sync-compatible for LangGraph if requested
    if make_sync:
        tools = [_make_sync_tool(tool) for tool in tools]

    return tools


async def load_combined_tools(
    cwd: Optional[str] = None,
    model_family: ModelFamily = ModelFamily.GENERIC,
    mcp_server_configs: Optional[dict[str, dict[str, Any]]] = None,
    alfredo_tool_ids: Optional[list[str]] = None,
    make_sync: bool = True,
) -> list[Any]:
    """Load both Alfredo and MCP tools in a single call.

    This is a convenience function that combines Alfredo tools with MCP tools,
    making it easy to create a unified toolset for the agentic scaffold.

    Args:
        cwd: Working directory for Alfredo file operations
        model_family: Model family for Alfredo tool variant selection
        mcp_server_configs: Optional MCP server configurations (same format as load_mcp_tools)
        alfredo_tool_ids: Optional list of specific Alfredo tool IDs to include.
            If None, includes all Alfredo tools.
        make_sync: If True, wrap MCP async tools to be sync-compatible with LangGraph (default: True)

    Returns:
        Combined list of LangChain StructuredTool instances

    Example:
        ```python
        import asyncio
        from alfredo.integrations.mcp import load_combined_tools

        async def main():
            # Load only Alfredo tools
            tools = await load_combined_tools(cwd=".")

            # Load Alfredo + MCP tools
            mcp_configs = {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "transport": "stdio"
                }
            }
            tools = await load_combined_tools(
                cwd=".",
                mcp_server_configs=mcp_configs
            )

        asyncio.run(main())
        ```
    """
    # Load Alfredo tools (synchronous)
    alfredo_tools = create_langchain_tools(
        cwd=cwd,
        model_family=model_family,
        tool_ids=alfredo_tool_ids,
    )

    # Load MCP tools if configured (asynchronous)
    if mcp_server_configs:
        mcp_tools = await load_mcp_tools(mcp_server_configs, make_sync=make_sync)
        return alfredo_tools + mcp_tools

    return alfredo_tools


def load_mcp_tools_sync(
    server_configs: dict[str, dict[str, Any]],
    make_sync: bool = True,
) -> list[Any]:
    """Synchronous wrapper for load_mcp_tools.

    This is a convenience function that creates an event loop and runs
    load_mcp_tools synchronously. Use this when you're not already in
    an async context.

    Args:
        server_configs: Dictionary mapping server names to configurations
            (same format as load_mcp_tools)
        make_sync: If True, wrap async tools to be sync-compatible with LangGraph (default: True)

    Returns:
        List of LangChain StructuredTool instances from the MCP servers

    Raises:
        ImportError: If langchain-mcp-adapters is not installed

    Example:
        ```python
        from alfredo.integrations.mcp import load_mcp_tools_sync

        server_configs = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "transport": "stdio"
            }
        }

        mcp_tools = load_mcp_tools_sync(server_configs)
        print(f"Loaded {len(mcp_tools)} MCP tools")
        ```
    """
    import asyncio

    return asyncio.run(load_mcp_tools(server_configs, make_sync=make_sync))


def load_combined_tools_sync(
    cwd: Optional[str] = None,
    model_family: ModelFamily = ModelFamily.GENERIC,
    mcp_server_configs: Optional[dict[str, dict[str, Any]]] = None,
    alfredo_tool_ids: Optional[list[str]] = None,
    make_sync: bool = True,
) -> list[Any]:
    """Synchronous wrapper for load_combined_tools.

    This is a convenience function that creates an event loop and runs
    load_combined_tools synchronously. Use this when you're not already in
    an async context.

    Args:
        cwd: Working directory for Alfredo file operations
        model_family: Model family for Alfredo tool variant selection
        mcp_server_configs: Optional MCP server configurations
        alfredo_tool_ids: Optional list of specific Alfredo tool IDs
        make_sync: If True, wrap MCP async tools to be sync-compatible with LangGraph (default: True)

    Returns:
        Combined list of LangChain StructuredTool instances

    Example:
        ```python
        from alfredo.integrations.mcp import load_combined_tools_sync

        mcp_configs = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "transport": "stdio"
            }
        }

        tools = load_combined_tools_sync(
            cwd=".",
            mcp_server_configs=mcp_configs
        )
        ```
    """
    import asyncio

    return asyncio.run(load_combined_tools(cwd, model_family, mcp_server_configs, alfredo_tool_ids, make_sync))


# AlfredoTool wrapping helpers for MCP tools


def wrap_mcp_tools(
    mcp_tools: list[Any],
    instruction_configs: Optional[dict[str, dict[str, str]]] = None,
) -> list[Any]:
    """Wrap MCP tools as AlfredoTools with optional system instructions.

    This function takes a list of MCP StructuredTools and wraps them as AlfredoTools,
    optionally adding node-specific system instructions.

    Args:
        mcp_tools: List of MCP StructuredTools (from load_mcp_tools or load_mcp_tools_sync)
        instruction_configs: Optional mapping of tool names to their system instructions.
            Example: {
                "mcp_tool_name": {
                    "agent": "Use this for external operations",
                    "verifier": "Check results with this"
                }
            }

    Returns:
        List of AlfredoTool instances

    Example:
        ```python
        from alfredo.integrations.mcp import load_mcp_tools_sync, wrap_mcp_tools

        # Load MCP tools
        server_configs = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "transport": "stdio"
            }
        }
        mcp_tools = load_mcp_tools_sync(server_configs)

        # Wrap with instructions
        wrapped_tools = wrap_mcp_tools(
            mcp_tools,
            instruction_configs={
                "read_file": {
                    "agent": "Use for external file reading"
                },
                "write_file": {
                    "agent": "Use for external file writing"
                }
            }
        )
        ```
    """
    from alfredo.tools.alfredo_tool import AlfredoTool

    instruction_configs = instruction_configs or {}
    wrapped = []

    for tool in mcp_tools:
        tool_name = getattr(tool, "name", "")
        system_instructions = instruction_configs.get(tool_name)

        wrapped_tool = AlfredoTool.from_mcp(
            mcp_tool=tool,
            system_instructions=system_instructions,
        )
        wrapped.append(wrapped_tool)

    return wrapped
