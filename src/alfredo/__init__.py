"""Alfredo: Python harness for AI agents with tool execution capabilities."""

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolUse
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec

__version__ = "0.0.1"

__all__ = [
    "BaseToolHandler",
    "ModelFamily",
    "ToolParameter",
    "ToolResult",
    "ToolSpec",
    "ToolUse",
    "registry",
]

# Agentic scaffold (recommended usage)
try:
    from alfredo.agentic.agent import Agent  # noqa: F401

    __all__.append("Agent")
except ImportError:
    pass  # LangGraph not installed

# Pre-built agents (requires agentic scaffold)
try:
    from alfredo.prebuilt import ExplorationAgent, ReflexionAgent  # noqa: F401

    __all__.extend(["ExplorationAgent", "ReflexionAgent"])
except ImportError:
    pass  # LangGraph not installed or prebuilt module not available

# Optional OpenAI integration (if openai is installed)
try:
    from alfredo.integrations.openai_native import (  # noqa: F401
        OpenAIAgent,
        get_all_tools_openai_format,
        tool_spec_to_openai_format,
    )

    __all__.extend(["OpenAIAgent", "get_all_tools_openai_format", "tool_spec_to_openai_format"])
except ImportError:
    pass  # OpenAI not installed, skip these exports
