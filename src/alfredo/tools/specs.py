"""Tool specification system for defining tool metadata and parameters."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ModelFamily(str, Enum):
    """Model family identifiers for tool variants."""

    GENERIC = "generic"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    NEXT_GEN = "next_gen"


@dataclass
class ToolParameter:
    """Specification for a tool parameter.

    Attributes:
        name: Parameter name as used in tool invocation
        required: Whether this parameter must be provided
        instruction: Instructions for the model on how to use this parameter
        usage: Example usage text to show in the prompt
        description: Optional additional description
        dependencies: Other tool IDs that must be available for this parameter to be shown
        context_requirements: Optional callable to determine if parameter should be included based on context
    """

    name: str
    required: bool
    instruction: str
    usage: str = ""
    description: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    context_requirements: Optional[Callable[[dict[str, Any]], bool]] = None


@dataclass
class ToolSpec:
    """Specification for an AI agent tool.

    This defines the metadata, description, and parameters for a tool that can be
    invoked by an AI agent. Different variants can be defined for different model families.

    Attributes:
        id: Unique identifier for the tool
        name: Display name of the tool
        description: Detailed description of what the tool does
        variant: Model family this tool spec is designed for
        parameters: List of parameters the tool accepts
        instruction: Optional additional instructions for using the tool
        context_requirements: Optional callable to determine if tool should be available based on context
    """

    id: str
    name: str
    description: str
    variant: ModelFamily = ModelFamily.GENERIC
    parameters: list[ToolParameter] = field(default_factory=list)
    instruction: Optional[str] = None
    context_requirements: Optional[Callable[[dict[str, Any]], bool]] = None

    def format_for_prompt(self, context: Optional[dict[str, Any]] = None) -> str:
        """Format this tool spec as a prompt section.

        Args:
            context: Optional context dictionary for evaluating context requirements

        Returns:
            Formatted string suitable for inclusion in a system prompt
        """
        if context is None:
            context = {}

        # Filter parameters based on context requirements
        filtered_params = [
            p for p in self.parameters if p.context_requirements is None or p.context_requirements(context)
        ]

        # Build the prompt
        lines = [
            f"## {self.id}",
            f"Description: {self.description}",
        ]

        # Add parameters section
        if filtered_params:
            lines.append("Parameters:")
            for param in filtered_params:
                req_text = "required" if param.required else "optional"
                lines.append(f"- {param.name}: ({req_text}) {param.instruction}")
                if param.description:
                    lines.append(f"  {param.description}")
        else:
            lines.append("Parameters: None")

        # Add usage section
        lines.append("Usage:")
        lines.append(f"<{self.id}>")
        for param in filtered_params:
            usage_text = param.usage or f"{param.name} here"
            lines.append(f"<{param.name}>{usage_text}</{param.name}>")
        lines.append(f"</{self.id}>")

        return "\n".join(lines)
