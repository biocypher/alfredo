"""Prompts for planning, execution, and verification in the agentic scaffold."""

from typing import Any, Optional


def _extract_instructions_for_node(tools: list[Any], node_name: str) -> str:
    """Extract all system instructions for a specific node from the tool list.

    Args:
        tools: List of tools (can be AlfredoTool or regular StructuredTool)
        node_name: Name of the node to extract instructions for

    Returns:
        Concatenated instruction string, or empty string if no instructions
    """
    from alfredo.tools.alfredo_tool import AlfredoTool

    instructions = []
    for tool in tools:
        if isinstance(tool, AlfredoTool):
            instruction = tool.get_instruction_for_node(node_name)
            if instruction:
                instructions.append(instruction)

    return "\n\n".join(instructions) if instructions else ""


def _process_custom_template(
    custom_template: str,
    required_vars: dict[str, Any],
    var_order: list[str],
) -> str:
    """Process custom template with auto-wrapping or validation.

    Supports two modes:
    1. Auto-wrap mode: If template has NO placeholders, automatically prepends
       dynamic variables and appends tool_instructions
    2. Validation mode: If template HAS placeholders, validates all required
       variables are present and formats the template

    Args:
        custom_template: User's template string
        required_vars: Dict of all required variables (e.g., {"task": "...", "plan": "...", "tool_instructions": "..."})
        var_order: Order of variables for auto-prepending (e.g., ["task", "plan"])
            Note: tool_instructions is always appended, never prepended

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If template has placeholders but is missing required ones

    Examples:
        >>> # Auto-wrap mode (no placeholders)
        >>> result = _process_custom_template(
        ...     "Create a detailed plan.",
        ...     {"task": "Build app", "tool_instructions": "Use tools"},
        ...     ["task"]
        ... )
        >>> # Result: "# Task\\nBuild app\\n\\nCreate a detailed plan.\\nUse tools"

        >>> # Validation mode (has placeholders)
        >>> result = _process_custom_template(
        ...     "Task: {task}\\n\\n{tool_instructions}",
        ...     {"task": "Build app", "tool_instructions": "Use tools"},
        ...     ["task"]
        ... )
        >>> # Result: "Task: Build app\\n\\nUse tools"
    """
    import re

    # Check if template has placeholders
    has_placeholders = "{" in custom_template and "}" in custom_template

    if not has_placeholders:
        # AUTO-WRAP MODE
        # Build prepend section (task, plan, etc.) - everything EXCEPT tool_instructions
        prepend_parts = []
        for var_name in var_order:
            if var_name in required_vars and var_name != "tool_instructions":
                value = required_vars[var_name]
                # Format header nicely (e.g., "previous_plan" -> "Previous Plan")
                header = var_name.replace("_", " ").title()
                prepend_parts.append(f"# {header}\n{value}")

        prepend_section = "\n\n".join(prepend_parts)

        # Get tool_instructions for appending
        # Note: tool_instructions already includes the "# Tool-Specific Instructions" header
        tool_instructions = required_vars.get("tool_instructions", "")

        # Build final prompt: PREPEND + USER CONTENT + APPEND
        if prepend_section:
            return f"{prepend_section}\n\n{custom_template}{tool_instructions}"
        else:
            return f"{custom_template}{tool_instructions}"

    else:
        # VALIDATION MODE
        # Extract placeholder names from template
        found_placeholders = set(re.findall(r"\{(\w+)\}", custom_template))
        required_keys = set(required_vars.keys())

        # Check if all required placeholders present (including tool_instructions!)
        missing = required_keys - found_placeholders
        if missing:
            error_msg = (
                f"Template missing required placeholders: {missing}. "
                f"Required: {required_keys}. "
                f"Found in template: {found_placeholders}"
            )
            raise ValueError(error_msg)

        # Format with values
        return custom_template.format(**required_vars)


def get_planning_prompt(
    task: str,
    tools: Optional[list[Any]] = None,
    custom_template: Optional[str] = None,
    code_act_tools: Optional[list[str]] = None,
) -> str:
    """Get the system prompt for creating an implementation plan.

    Args:
        task: The task to create a plan for
        tools: Optional list of tools (for extracting node-specific instructions)
        custom_template: Optional custom template string. Can be:
            - Plain text (auto-prepended with {task}, auto-appended with {tool_instructions})
            - Template with {task} and {tool_instructions} placeholders
        code_act_tools: Optional list of tool IDs to expose as code (e.g., ["read_file", "write_to_file"])

    Returns:
        Formatted planning prompt
    """
    # Extract tool-specific instructions for the planner node
    tool_instructions = ""
    if tools:
        extracted = _extract_instructions_for_node(tools, "planner")
        if extracted:
            tool_instructions = f"\n\n# Tool-Specific Instructions\n\n{extracted}"

        # Inject tool code if code_act_tools is specified
        if code_act_tools:
            from alfredo.tools.code_extraction import format_tool_code_section

            tool_code = format_tool_code_section(tools, tool_ids=code_act_tools)
            if tool_code:
                tool_instructions += f"\n\n{tool_code}"

    # If custom template provided, process it
    if custom_template:
        return _process_custom_template(
            custom_template=custom_template,
            required_vars={"task": task, "tool_instructions": tool_instructions},
            var_order=["task"],
        )

    # Default built-in prompt
    return f"""You are a meticulous AI agent tasked with creating a comprehensive implementation plan.

# Your Task
{task}

# Instructions

Your job is to create a detailed, actionable plan for completing this task. The plan will be used by
an execution agent to implement the solution.

## Planning Process

1. **Analyze the Task**: Understand what needs to be done, why, and the expected outcome
2. **Break Down Steps**: Decompose the task into logical, sequential steps
3. **Identify Requirements**: Note any files, tools, or information needed
4. **Consider Edge Cases**: Think about potential issues and how to handle them

## Plan Format

Your plan should be structured as follows:

# Implementation Plan

## Overview
[2-3 sentences describing the overall approach and goal]

## Steps
1. [First action to take]
   - Rationale: [Why this step is necessary]
   - Expected outcome: [What this achieves]

2. [Second action to take]
   - Rationale: [Why this step is necessary]
   - Expected outcome: [What this achieves]

[Continue for all steps...]

N. Verify the task is complete and call attempt_completion tool
   - Rationale: Signal completion and provide final summary
   - Expected outcome: Task ends successfully

## Success Criteria
[How to verify the task is complete]

IMPORTANT: The final step must ALWAYS be calling the attempt_completion tool to complete the task.

## Potential Issues
[Any risks or edge cases to watch for]

# Important Notes

- Be specific about file paths, commands, and tool usage
- Consider dependencies between steps
- Think about validation and testing
- Keep the plan focused and actionable
{tool_instructions}
Now create the implementation plan for the task above."""


def get_agent_system_prompt(
    task: str,
    plan: str,
    tools: Optional[list[Any]] = None,
    custom_template: Optional[str] = None,
    code_act_tools: Optional[list[str]] = None,
) -> str:
    """Get the system prompt for the agent node.

    Args:
        task: The original task
        plan: The current implementation plan
        tools: Optional list of tools (for extracting node-specific instructions)
        custom_template: Optional custom template string. Can be:
            - Plain text (auto-prepended with {task} and {plan}, auto-appended with {tool_instructions})
            - Template with {task}, {plan}, and {tool_instructions} placeholders
        code_act_tools: Optional list of tool IDs to expose as code (e.g., ["read_file", "write_to_file"])

    Returns:
        Formatted agent system prompt
    """
    # Extract tool-specific instructions for the agent node
    tool_instructions = ""
    if tools:
        extracted = _extract_instructions_for_node(tools, "agent")
        if extracted:
            tool_instructions = f"\n\n# Tool-Specific Instructions\n\n{extracted}"

        # Inject tool code if code_act_tools is specified
        if code_act_tools:
            from alfredo.tools.code_extraction import format_tool_code_section

            tool_code = format_tool_code_section(tools, tool_ids=code_act_tools)
            if tool_code:
                tool_instructions += f"\n\n{tool_code}"

    # If custom template provided, process it
    if custom_template:
        return _process_custom_template(
            custom_template=custom_template,
            required_vars={"task": task, "plan": plan, "tool_instructions": tool_instructions},
            var_order=["task", "plan"],
        )

    # Default built-in prompt
    # Conditionally include plan section if plan is provided
    plan_section = ""
    if plan and plan.strip():
        plan_section = f"""
# Implementation Plan
{plan}
"""

    return f"""You are an autonomous AI agent executing a task using a ReAct (Reasoning-Action-Observation) approach.

# Original Task
{task}
{plan_section}
# Your Role

You will iterate through a think-act-observe loop:
1. **Think**: Reason about the current state and decide the next action
2. **Act**: Use tools to take actions or gather information
3. **Observe**: Examine tool results and update your understanding

# Important Rules
{"- Follow the implementation plan, but adapt if you discover new information" if plan and plan.strip() else "- Break down the task into logical steps and work through them systematically"}
- Use ONE tool call per message - you'll see the result before proceeding
- Think step-by-step and be methodical
- Be specific in your reasoning and actions

# Tools

You have access to various tools for:
- Reading and writing files
- Executing commands
- Searching code
- Web fetching
- And more

Use tools strategically to accomplish the task efficiently.
{tool_instructions}
# ⚠️ CRITICAL: How to Complete the Task

**YOU MUST call the `attempt_completion` tool when you finish the task.**

This is the ONLY way to complete execution. If you don't call it, the task will never end.

**When to call `attempt_completion`:**
- ✅ All task requirements are satisfied
- ✅ You have verified your work (ran tests, checked files, etc.)
- ✅ Everything works as expected

**Do NOT call `attempt_completion` if:**
- ❌ You haven't completed all steps
- ❌ There are errors or failures
- ❌ You're still working on something

**How to call it:**
```
attempt_completion(result="I completed the task by: 1. Created X, 2. Tested Y, 3. Result: Success")
```

Your answer will be verified before final completion."""


def get_verification_prompt(
    task: str,
    answer: str,
    execution_trace: str = "",
    tools: Optional[list[Any]] = None,
    custom_template: Optional[str] = None,
    code_act_tools: Optional[list[str]] = None,
) -> str:
    """Get the prompt for verifying if an answer satisfies the task.

    Args:
        task: The original task
        answer: The agent's proposed answer
        execution_trace: Optional trace of actions taken during execution
        tools: Optional list of tools (for extracting node-specific instructions)
        custom_template: Optional custom template string. Can be:
            - Plain text (auto-prepended with {task}, {answer}, {trace_section}, auto-appended with {tool_instructions})
            - Template with {task}, {answer}, {trace_section}, and {tool_instructions} placeholders
        code_act_tools: Optional list of tool IDs to expose as code (e.g., ["read_file", "write_to_file"])

    Returns:
        Formatted verification prompt
    """
    # Extract tool-specific instructions for the verifier node
    tool_instructions = ""
    if tools:
        extracted = _extract_instructions_for_node(tools, "verifier")
        if extracted:
            tool_instructions = f"\n\n# Tool-Specific Instructions\n\n{extracted}"

        # Inject tool code if code_act_tools is specified
        if code_act_tools:
            from alfredo.tools.code_extraction import format_tool_code_section

            tool_code = format_tool_code_section(tools, tool_ids=code_act_tools)
            if tool_code:
                tool_instructions += f"\n\n{tool_code}"

    trace_section = ""
    if execution_trace:
        trace_section = f"""# Execution Trace

Below is the complete trace of actions taken to complete the task:

{execution_trace}
"""

    # If custom template provided, process it
    if custom_template:
        return _process_custom_template(
            custom_template=custom_template,
            required_vars={
                "task": task,
                "answer": answer,
                "trace_section": trace_section,
                "tool_instructions": tool_instructions,
            },
            var_order=["task", "answer", "trace_section"],
        )

    # Default built-in prompt
    return f"""You are a verification agent. Your job is to determine if the task was actually completed by examining both the claimed answer and the execution trace.

# Original Task
{task}

# Proposed Answer
{answer}
{trace_section}
# Verification Criteria

Evaluate whether the task was completed by checking:
1. **Actions taken**: Review the execution trace - were the right tools used?
2. **Evidence of completion**: Do the tool results show the work was actually done?
3. **Addresses the requirement**: Does the answer accurately reflect what was accomplished?
4. **Completeness**: Are all aspects of the task covered?
5. **Accuracy**: Is the claimed answer consistent with what the trace shows?

**IMPORTANT**: Don't just trust the answer - verify it against the actual actions taken.
If the answer claims something was done, check the execution trace to confirm it actually happened.

# Your Response

Respond with ONLY ONE of the following:

**If the task is verified as complete:**
VERIFIED: [Brief explanation citing specific evidence from the execution trace]

**If the task is NOT complete:**
NOT_VERIFIED: [Specific explanation of what's missing, incorrect, or not evidenced in the trace]

Be objective and thorough in your evaluation. Base your decision on concrete evidence from the execution trace.
{tool_instructions}"""


def get_replan_prompt(
    task: str,
    previous_plan: str,
    verification_feedback: str,
    tools: Optional[list[Any]] = None,
    custom_template: Optional[str] = None,
    code_act_tools: Optional[list[str]] = None,
) -> str:
    """Get the prompt for creating a new plan after verification failure.

    Args:
        task: The original task
        previous_plan: The plan that was just attempted
        verification_feedback: Feedback from the verification step
        tools: Optional list of tools (for extracting node-specific instructions)
        custom_template: Optional custom template string. Can be:
            - Plain text (auto-prepended with {task}, {previous_plan}, {verification_feedback}, auto-appended with {tool_instructions})
            - Template with {task}, {previous_plan}, {verification_feedback}, and {tool_instructions} placeholders
        code_act_tools: Optional list of tool IDs to expose as code (e.g., ["read_file", "write_to_file"])

    Returns:
        Formatted replanning prompt
    """
    # Extract tool-specific instructions for the replan node
    tool_instructions = ""
    if tools:
        extracted = _extract_instructions_for_node(tools, "replan")
        if extracted:
            tool_instructions = f"\n\n# Tool-Specific Instructions\n\n{extracted}"

        # Inject tool code if code_act_tools is specified
        if code_act_tools:
            from alfredo.tools.code_extraction import format_tool_code_section

            tool_code = format_tool_code_section(tools, tool_ids=code_act_tools)
            if tool_code:
                tool_instructions += f"\n\n{tool_code}"

    # If custom template provided, process it
    if custom_template:
        return _process_custom_template(
            custom_template=custom_template,
            required_vars={
                "task": task,
                "previous_plan": previous_plan,
                "verification_feedback": verification_feedback,
                "tool_instructions": tool_instructions,
            },
            var_order=["task", "previous_plan", "verification_feedback"],
        )

    # Default built-in prompt
    return f"""You are a meticulous AI agent tasked with creating a NEW implementation plan.

# Original Task
{task}

# Previous Plan (that didn't fully succeed)
{previous_plan}

# Verification Feedback
{verification_feedback}

# Instructions

The previous attempt didn't fully satisfy the task requirements. Create a NEW plan that:
1. Addresses the issues identified in the verification feedback
2. Learns from what was attempted before
3. Takes a different or improved approach

Use the same plan format as before:

# Implementation Plan

## Overview
[What went wrong and how this plan fixes it]

## Steps
1. [Action]
   - Rationale: [Why]
   - Expected outcome: [What this achieves]

[Continue...]

## Success Criteria
[How to verify the task is complete]
{tool_instructions}
Now create the improved implementation plan."""


def get_context_summary_prompt(messages_summary: str, task: str, plan: str) -> str:
    """Get the prompt for summarizing context when approaching token limits.

    Args:
        messages_summary: Summary of recent message history
        task: The original task
        plan: The current plan

    Returns:
        Formatted summarization prompt
    """
    return f"""You are compacting the conversation context to stay within token limits.

# Task
{task}

# Current Plan
{plan}

# Recent Activity
{messages_summary}

# Instructions

Create a concise summary that preserves:
1. **Key Actions Taken**: What tools were used and major steps completed
2. **Important Findings**: Critical information discovered
3. **Current State**: Where we are in the plan execution
4. **Next Steps**: What needs to be done next

Keep the summary focused on information essential for continuing the task.

Format:
## Actions Completed
[List of key actions]

## Key Findings
[Important information]

## Current Progress
[Where we are in the plan]

## Next Steps
[What to do next]"""
