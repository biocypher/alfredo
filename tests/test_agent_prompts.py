"""Tests for Agent prompt and tool description methods."""

import os
from typing import Any

import pytest


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agent_get_system_prompts() -> None:
    """Test that get_system_prompts returns all expected prompts."""
    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    prompts = agent.get_system_prompts(
        task="Create a hello world script",
        plan="1. Create file\n2. Write code",
    )

    # Check that all expected node prompts are present
    assert "planner" in prompts
    assert "agent" in prompts
    assert "verifier" in prompts
    assert "replan" in prompts

    # Check that prompts contain expected content
    assert "Create a hello world script" in prompts["planner"]
    assert "Create a hello world script" in prompts["agent"]
    assert "1. Create file" in prompts["agent"]

    # Check that each prompt is a non-empty string
    for node_name, prompt_text in prompts.items():
        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0, f"Prompt for {node_name} is empty"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agent_get_tool_descriptions() -> None:
    """Test that get_tool_descriptions returns tool information."""
    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    tool_descriptions = agent.get_tool_descriptions()

    # Should have multiple tools
    assert len(tool_descriptions) > 0

    # Check structure of each tool description
    for tool in tool_descriptions:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert "tool_type" in tool

        # Check types
        assert isinstance(tool["name"], str)
        assert isinstance(tool["description"], str)
        assert isinstance(tool["parameters"], list)
        assert tool["tool_type"] in ["alfredo", "mcp"]

        # Check parameter structure
        for param in tool["parameters"]:
            assert "name" in param
            assert "required" in param
            assert "description" in param

    # Check that key tools are present
    tool_names = [t["name"] for t in tool_descriptions]
    assert "read_file" in tool_names
    assert "write_to_file" in tool_names
    assert "attempt_completion" in tool_names


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agent_has_todo_tools() -> None:
    """Test that _has_todo_tools detects todo tools correctly."""
    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    # By default, all tools should be loaded, including todo tools
    has_todo = agent._has_todo_tools()
    assert isinstance(has_todo, bool)

    # Check that todo tools are actually present
    tool_descriptions = agent.get_tool_descriptions()
    tool_names = [t["name"] for t in tool_descriptions]
    has_write_todo = "write_todo_list" in tool_names
    has_read_todo = "read_todo_list" in tool_names

    assert has_todo == (has_write_todo or has_read_todo)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agent_display_tool_descriptions(capsys: Any) -> None:
    """Test that display_tool_descriptions prints formatted output."""
    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    # Call display method (should print to stdout)
    agent.display_tool_descriptions()

    # Capture output
    captured = capsys.readouterr()

    # Check that output contains expected sections
    assert "TOOL DESCRIPTIONS" in captured.out
    assert "ALFREDO TOOLS" in captured.out
    assert "Total Tools:" in captured.out

    # Check that specific tools are shown
    assert "read_file" in captured.out
    assert "write_to_file" in captured.out
    assert "attempt_completion" in captured.out


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agent_display_system_prompts(capsys: Any) -> None:
    """Test that display_system_prompts prints formatted output."""
    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    # Call display method with example task
    agent.display_system_prompts(
        task="Create a test script",
        plan="1. Write test\n2. Run test",
    )

    # Capture output
    captured = capsys.readouterr()

    # Check that output contains expected sections
    assert "AGENT SYSTEM PROMPTS" in captured.out
    assert "Configuration" in captured.out
    assert "PLANNER NODE PROMPT" in captured.out
    assert "AGENT NODE PROMPT" in captured.out
    assert "VERIFIER NODE PROMPT" in captured.out
    assert "REPLAN NODE PROMPT" in captured.out

    # Check configuration details
    assert "Model: gpt-4.1-mini" in captured.out
    assert "Tools Available:" in captured.out

    # Check that task appears in prompts
    assert "Create a test script" in captured.out


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_tool_type_detection() -> None:
    """Test that tool type detection works correctly."""
    from alfredo import Agent

    agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=False)

    tool_descriptions = agent.get_tool_descriptions()

    # All default tools should be Alfredo tools
    for tool in tool_descriptions:
        # Alfredo tools start with specific prefixes
        if tool["name"] in ["read_file", "write_to_file", "list_files", "execute_command", "attempt_completion"]:
            assert tool["tool_type"] == "alfredo"


def test_get_system_prompts_without_api_key() -> None:
    """Test that get_system_prompts works even without creating a full agent."""
    # This test doesn't require an API key because we're not creating the graph
    # We'll test the method logic directly

    # Import the prompt functions
    from alfredo.agentic.prompts import (
        get_agent_system_prompt,
        get_planning_prompt,
        get_replan_prompt,
        get_verification_prompt,
    )

    # Test that prompts can be generated
    planner = get_planning_prompt(task="Test task", has_todo_tools=False)
    agent = get_agent_system_prompt(task="Test task", plan="Test plan", has_todo_tools=False)
    verifier = get_verification_prompt(task="Test task", answer="Test answer")
    replan = get_replan_prompt(task="Test task", previous_plan="Old plan", verification_feedback="Feedback")

    # Check that all prompts are non-empty strings
    assert isinstance(planner, str) and len(planner) > 0
    assert isinstance(agent, str) and len(agent) > 0
    assert isinstance(verifier, str) and len(verifier) > 0
    assert isinstance(replan, str) and len(replan) > 0

    # Check that prompts contain expected content
    assert "Test task" in planner
    assert "Test task" in agent
    assert "Test plan" in agent
    assert "Test task" in verifier
    assert "Test answer" in verifier
    assert "Old plan" in replan
    assert "Feedback" in replan


def test_is_mcp_tool_method() -> None:
    """Test that _is_mcp_tool correctly identifies Alfredo vs MCP tools."""
    from langchain_core.tools import StructuredTool

    from alfredo.agentic.agent import Agent
    from alfredo.integrations.langchain import create_langchain_tools

    # Get Alfredo tools
    alfredo_tools = create_langchain_tools(cwd=".")

    # Create mock MCP tools
    mock_mcp_tools = [
        StructuredTool.from_function(
            name="fs_read_file",
            description="Mock MCP filesystem tool",
            func=lambda path: f"Reading {path}",
        ),
        StructuredTool.from_function(
            name="bc_get_balance",
            description="Mock MCP blockchain tool",
            func=lambda address: f"Balance of {address}",
        ),
    ]

    # Create minimal agent instance (without initializing the graph)
    agent = Agent.__new__(Agent)
    agent.tools = alfredo_tools + mock_mcp_tools

    # Test Alfredo tools are correctly identified
    alfredo_tool_names = ["read_file", "write_to_file", "replace_in_file", "list_files", "execute_command"]
    for tool_name in alfredo_tool_names:
        assert not agent._is_mcp_tool(tool_name), f"{tool_name} should be identified as Alfredo tool"

    # Test MCP tools are correctly identified
    mcp_tool_names = ["fs_read_file", "bc_get_balance", "github_create_issue"]
    for tool_name in mcp_tool_names:
        assert agent._is_mcp_tool(tool_name), f"{tool_name} should be identified as MCP tool"


def test_tool_classification_with_registry() -> None:
    """Test that tool classification uses registry as authoritative source."""
    from alfredo.agentic.agent import Agent
    from alfredo.integrations.langchain import create_langchain_tools
    from alfredo.tools.registry import registry

    # Get all registered Alfredo tool IDs
    registered_tool_ids = set(registry.get_all_tool_ids())

    # Create tools and agent
    alfredo_tools = create_langchain_tools(cwd=".")
    agent = Agent.__new__(Agent)
    agent.tools = alfredo_tools

    # Get tool descriptions
    tool_descriptions = agent.get_tool_descriptions()

    # Verify all tools are correctly classified
    for tool_desc in tool_descriptions:
        tool_name = tool_desc["name"]
        tool_type = tool_desc["tool_type"]

        # If tool is in registry, it should be classified as alfredo
        if tool_name in registered_tool_ids:
            assert tool_type == "alfredo", f"{tool_name} is in registry but classified as {tool_type}"
        else:
            assert tool_type == "mcp", f"{tool_name} is not in registry but classified as {tool_type}"

    # Verify key tools are present and correctly classified
    key_tools = ["read_file", "write_to_file", "replace_in_file", "attempt_completion"]
    classified_names = {t["name"]: t["tool_type"] for t in tool_descriptions}

    for tool_name in key_tools:
        assert tool_name in classified_names, f"{tool_name} not found in tool descriptions"
        assert classified_names[tool_name] == "alfredo", (
            f"{tool_name} should be classified as alfredo, got {classified_names[tool_name]}"
        )
