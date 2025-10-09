"""Tests for custom prompt template functionality."""

import os

import pytest


def test_plain_text_auto_wrapping() -> None:
    """Test that plain text gets task prepended and tool_instructions appended."""
    from alfredo.agentic.prompts import get_planning_prompt

    custom_template = "Create a detailed step-by-step plan."
    prompt = get_planning_prompt(task="Build an app", tools=None, custom_template=custom_template)

    # Should have task at beginning
    assert "Build an app" in prompt
    # Should have user's content
    assert "Create a detailed step-by-step plan." in prompt
    # Task should come before user content
    assert prompt.index("Build an app") < prompt.index("Create a detailed step-by-step plan.")


def test_explicit_placeholders_validation_pass() -> None:
    """Test that templates with all required placeholders work."""
    from alfredo.agentic.prompts import get_planning_prompt

    custom_template = """Task: {task}

Create a detailed plan.

{tool_instructions}"""
    prompt = get_planning_prompt(task="Build an app", tools=None, custom_template=custom_template)

    # Should have task and user content
    assert "Build an app" in prompt
    assert "Create a detailed plan." in prompt


def test_explicit_placeholders_validation_fail() -> None:
    """Test that missing placeholders raise ValueError."""
    from alfredo.agentic.prompts import get_agent_system_prompt

    # Missing {plan} and {tool_instructions}
    custom_template = "Just do {task}"

    with pytest.raises(ValueError, match="missing required placeholders"):
        get_agent_system_prompt(task="Test", plan="Plan", tools=None, custom_template=custom_template)


def test_tool_instructions_auto_appended() -> None:
    """Test that tool_instructions are always appended in plain text mode."""
    from alfredo.agentic.prompts import get_agent_system_prompt
    from alfredo.integrations.langchain import create_alfredo_tools
    from alfredo.tools.handlers.todo import TODO_SYSTEM_INSTRUCTIONS

    # Create tools with custom instructions
    tools = create_alfredo_tools(
        cwd=".",
        tool_configs={
            "write_todo_list": TODO_SYSTEM_INSTRUCTIONS,
            "read_todo_list": TODO_SYSTEM_INSTRUCTIONS,
        },
    )

    custom_template = "Execute the plan carefully."
    prompt = get_agent_system_prompt(task="Test", plan="Plan", tools=tools, custom_template=custom_template)

    # Should have user content
    assert "Execute the plan carefully." in prompt
    # Should have todo tool instructions appended
    assert "Todo List Management" in prompt or "write_todo_list" in prompt


def test_agent_setter_methods() -> None:
    """Test that agent setter methods work and rebuild graph."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    # Set custom planner prompt
    agent.set_planner_prompt("Create a simple 3-step plan.")

    # Verify it was set
    template = agent.get_prompt_template("planner")
    assert template == "Create a simple 3-step plan."

    # Set custom agent prompt
    agent.set_agent_prompt("Work methodically.")
    assert agent.get_prompt_template("agent") == "Work methodically."

    # Reset prompts
    agent.reset_prompts()
    assert agent.get_prompt_template("planner") is None
    assert agent.get_prompt_template("agent") is None


def test_get_system_prompts_with_custom_templates() -> None:
    """Test that get_system_prompts uses custom templates."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    # Set custom planner prompt
    agent.set_planner_prompt("Make a concise plan.")

    # Get system prompts
    prompts = agent.get_system_prompts(task="Test task")

    # Planner should use custom template
    assert "Make a concise plan." in prompts["planner"]
    assert "Test task" in prompts["planner"]


def test_verifier_prompt_with_trace_section() -> None:
    """Test verifier prompt handles trace_section placeholder correctly."""
    from alfredo.agentic.prompts import get_verification_prompt

    # Plain text mode (no placeholders)
    custom_template = "Verify the task was completed."
    prompt = get_verification_prompt(
        task="Test",
        answer="Done",
        execution_trace="Step 1: Did X",
        tools=None,
        custom_template=custom_template,
    )

    # Should have all parts
    assert "Test" in prompt
    assert "Done" in prompt
    assert "Step 1: Did X" in prompt
    assert "Verify the task was completed." in prompt


def test_replan_prompt_with_feedback() -> None:
    """Test replan prompt handles all required variables."""
    from alfredo.agentic.prompts import get_replan_prompt

    # Explicit placeholders mode
    custom_template = """Task: {task}
Previous: {previous_plan}
Feedback: {verification_feedback}

Create improved plan.

{tool_instructions}"""

    prompt = get_replan_prompt(
        task="Test task",
        previous_plan="Old plan",
        verification_feedback="Needs improvement",
        tools=None,
        custom_template=custom_template,
    )

    assert "Test task" in prompt
    assert "Old plan" in prompt
    assert "Needs improvement" in prompt
    assert "Create improved plan." in prompt


def test_multiple_dynamic_vars_prepended() -> None:
    """Test that multiple variables are prepended in correct order."""
    from alfredo.agentic.prompts import get_agent_system_prompt

    custom_template = "Your custom instructions here."
    prompt = get_agent_system_prompt(
        task="Build app",
        plan="Step 1: Code\nStep 2: Test",
        tools=None,
        custom_template=custom_template,
    )

    # Task should come first
    task_pos = prompt.index("Build app")
    # Plan should come after task
    plan_pos = prompt.index("Step 1: Code")
    # User content should come after both
    content_pos = prompt.index("Your custom instructions here.")

    assert task_pos < plan_pos < content_pos


def test_empty_tool_instructions() -> None:
    """Test that empty tool instructions don't add extra content."""
    from alfredo.agentic.prompts import get_planning_prompt

    custom_template = "Simple plan."
    # No tools provided, so tool_instructions will be empty
    prompt = get_planning_prompt(task="Test", tools=None, custom_template=custom_template)

    # Should have task and content
    assert "Test" in prompt
    assert "Simple plan." in prompt
    # Should not have tool instructions header when empty
    assert "Tool-Specific Instructions" not in prompt


def test_placeholder_detection() -> None:
    """Test that placeholder detection works correctly."""
    from alfredo.agentic.prompts import _process_custom_template

    # No placeholders
    result = _process_custom_template(
        custom_template="Plain text",
        required_vars={"task": "Test", "tool_instructions": ""},
        var_order=["task"],
    )
    # Should have prepended task
    assert "Test" in result
    assert "Plain text" in result

    # With placeholders
    result = _process_custom_template(
        custom_template="Task: {task}\n{tool_instructions}",
        required_vars={"task": "Test", "tool_instructions": "Tools"},
        var_order=["task"],
    )
    # Should format with values
    assert "Task: Test" in result
    assert "Tools" in result


def test_missing_placeholder_error_message() -> None:
    """Test that missing placeholder errors are clear and helpful."""
    from alfredo.agentic.prompts import get_agent_system_prompt

    # Only has {task}, missing {plan} and {tool_instructions}
    custom_template = "Do {task} now."

    try:
        get_agent_system_prompt(task="Test", plan="Plan", tools=None, custom_template=custom_template)
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        # Should mention what's missing
        assert "missing required placeholders" in error_msg.lower()
        # Should show required keys
        assert "plan" in error_msg.lower() or "'plan'" in error_msg
        assert "tool_instructions" in error_msg.lower() or "'tool_instructions'" in error_msg
