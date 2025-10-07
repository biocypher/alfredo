"""Prompt builder for generating system prompts with tool definitions."""

from typing import Any, Optional

from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily


class PromptBuilder:
    """Builds system prompts that include tool definitions.

    This class generates formatted prompts that describe available tools
    to the AI agent in a format that can be easily understood and used.
    """

    def __init__(
        self,
        cwd: str,
        model_family: ModelFamily = ModelFamily.GENERIC,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            cwd: Current working directory to include in prompts
            model_family: Which model family to build prompts for
            context: Optional context dictionary for conditional tool inclusion
        """
        self.cwd = cwd
        self.model_family = model_family
        self.context = context or {}

    def build_system_prompt(self, tool_ids: Optional[list[str]] = None, include_base_instructions: bool = True) -> str:
        """Build a complete system prompt with tool definitions.

        Args:
            tool_ids: Optional list of specific tool IDs to include. If None, includes all available tools.
            include_base_instructions: Whether to include base agent instructions

        Returns:
            Formatted system prompt string
        """
        sections = []

        if include_base_instructions:
            sections.append(self._get_base_instructions())

        sections.append(self._get_tool_section(tool_ids))

        return "\n\n====\n\n".join(sections)

    def _get_base_instructions(self) -> str:
        """Get base instructions for the agent."""
        return f"""You are an AI assistant with access to tools for file operations, command execution, and more.

IMPORTANT RULES:
- You can use ONE tool per message
- Always use the exact XML format specified for each tool
- Relative paths are resolved from the current working directory: {self.cwd}
- After using a tool, you will receive the result and can decide on the next action
- If you need information to proceed, use the ask_followup_question tool
- When you believe the task is complete, use the attempt_completion tool
"""

    def _get_tool_section(self, tool_ids: Optional[list[str]] = None) -> str:
        """Build the tools section of the prompt.

        Args:
            tool_ids: Optional list of specific tool IDs to include

        Returns:
            Formatted tools section
        """
        # Get tool specs
        if tool_ids:
            all_specs = [registry.get_spec(tool_id, self.model_family) for tool_id in tool_ids]
            specs = [s for s in all_specs if s is not None]  # Filter out None values
        else:
            specs = registry.get_specs_for_variant(self.model_family)

        if not specs:
            return "# Tools\n\nNo tools available."

        # Format each tool
        tool_descriptions = []
        for spec in specs:
            # Check context requirements
            if spec.context_requirements and not spec.context_requirements(self.context):
                continue

            tool_descriptions.append(spec.format_for_prompt(self.context))

        if not tool_descriptions:
            return "# Tools\n\nNo tools available for the current context."

        return "# Tools\n\n" + "\n\n".join(tool_descriptions)

    def build_tool_use_example(self) -> str:
        """Build an example of tool usage for the prompt.

        Returns:
            Formatted example string
        """
        return """# Tool Usage Example

To read a file:
<read_file>
<path>src/main.py</path>
</read_file>

To write a file:
<write_to_file>
<path>output.txt</path>
<content>
This is the content of the file.
It can span multiple lines.
</content>
</write_to_file>

To execute a command:
<execute_command>
<command>ls -la</command>
</execute_command>
"""

    def get_available_tools(self) -> list[str]:
        """Get a list of available tool IDs for the current configuration.

        Returns:
            List of tool IDs
        """
        specs = registry.get_specs_for_variant(self.model_family)
        return [
            spec.id for spec in specs if spec.context_requirements is None or spec.context_requirements(self.context)
        ]
