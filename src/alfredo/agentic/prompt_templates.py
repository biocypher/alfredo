"""Dataclass for storing custom prompt templates."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplates:
    """Storage for custom prompt templates.

    Templates can be provided in two formats:

    1. **Plain text (auto-wrapping mode)**:
       - System automatically prepends dynamic variables (task, plan, etc.)
       - System automatically appends tool_instructions
       - User provides the core prompt content

    2. **Explicit placeholders (validation mode)**:
       - User includes {variable} placeholders in the template
       - System validates all required placeholders are present
       - System formats the template with actual values

    Example (plain text):
        >>> templates = PromptTemplates(
        ...     planner="Create a detailed step-by-step implementation plan."
        ... )

    Example (explicit placeholders):
        >>> templates = PromptTemplates(
        ...     agent='''
        ... # Task: {task}
        ... # Plan: {plan}
        ...
        ... Execute the plan carefully.
        ...
        ... {tool_instructions}
        ... '''
        ... )

    Attributes:
        planner: Custom template for the planner node
            Required vars: task, tool_instructions
        agent: Custom template for the agent node
            Required vars: task, plan, tool_instructions
        verifier: Custom template for the verifier node
            Required vars: task, answer, trace_section, tool_instructions
        replan: Custom template for the replan node
            Required vars: task, previous_plan, verification_feedback, tool_instructions
    """

    planner: Optional[str] = None
    agent: Optional[str] = None
    verifier: Optional[str] = None
    replan: Optional[str] = None
