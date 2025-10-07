"""Alfredo: Python harness for AI agents with tool execution capabilities."""

from alfredo.agent import Agent
from alfredo.tools.base import BaseToolHandler, ToolResult, ToolUse
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "BaseToolHandler",
    "ModelFamily",
    "ToolParameter",
    "ToolResult",
    "ToolSpec",
    "ToolUse",
    "registry",
]
