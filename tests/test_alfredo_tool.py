"""Tests for AlfredoTool class and node-specific system instructions."""

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from alfredo.tools.alfredo_tool import AlfredoTool

# Test fixtures


class MockInputSchema(BaseModel):
    """Mock input schema for testing."""

    text: str = Field(description="Input text")


def mock_tool_function(text: str) -> str:
    """Mock tool function for testing."""
    return f"Processed: {text}"


@pytest.fixture
def mock_langchain_tool() -> StructuredTool:
    """Create a mock LangChain StructuredTool for testing."""
    return StructuredTool.from_function(
        func=mock_tool_function,
        name="mock_tool",
        description="A mock tool for testing",
        args_schema=MockInputSchema,
    )


# Test AlfredoTool creation and basic functionality


def test_alfredo_tool_creation_from_langchain(mock_langchain_tool: StructuredTool) -> None:
    """Test creating AlfredoTool from LangChain tool."""
    tool = AlfredoTool.from_langchain(mock_langchain_tool)

    assert tool.name == "mock_tool"
    assert tool.description == "A mock tool for testing"
    assert isinstance(tool.langchain_tool, StructuredTool)


def test_alfredo_tool_with_system_instructions(mock_langchain_tool: StructuredTool) -> None:
    """Test AlfredoTool with node-specific instructions."""
    instructions = {
        "agent": "Agent instruction",
        "planner": "Planner instruction",
    }

    tool = AlfredoTool.from_langchain(
        mock_langchain_tool,
        system_instructions=instructions,
    )

    assert tool.get_instruction_for_node("agent") == "Agent instruction"
    assert tool.get_instruction_for_node("planner") == "Planner instruction"
    assert tool.get_instruction_for_node("verifier") is None


def test_alfredo_tool_target_nodes(mock_langchain_tool: StructuredTool) -> None:
    """Test getting target nodes from AlfredoTool."""
    instructions = {
        "agent": "Agent instruction",
        "verifier": "Verifier instruction",
    }

    tool = AlfredoTool.from_langchain(
        mock_langchain_tool,
        system_instructions=instructions,
    )

    target_nodes = tool.get_target_nodes()
    assert set(target_nodes) == {"agent", "verifier"}


def test_alfredo_tool_is_available_for_node(mock_langchain_tool: StructuredTool) -> None:
    """Test checking if tool is available for specific nodes."""
    instructions = {"agent": "Agent instruction"}

    tool = AlfredoTool.from_langchain(
        mock_langchain_tool,
        system_instructions=instructions,
    )

    assert tool.is_available_for_node("agent") is True
    assert tool.is_available_for_node("planner") is False
    assert tool.is_available_for_node("verifier") is False


def test_alfredo_tool_to_langchain_tool(mock_langchain_tool: StructuredTool) -> None:
    """Test extracting underlying LangChain tool."""
    tool = AlfredoTool.from_langchain(mock_langchain_tool)
    extracted = tool.to_langchain_tool()

    assert extracted == mock_langchain_tool
    assert extracted.name == "mock_tool"


def test_alfredo_tool_without_instructions(mock_langchain_tool: StructuredTool) -> None:
    """Test AlfredoTool without any system instructions."""
    tool = AlfredoTool.from_langchain(mock_langchain_tool)

    assert tool.get_target_nodes() == []
    assert tool.get_instruction_for_node("agent") is None
    assert tool.is_available_for_node("agent") is False


# Test AlfredoTool factory methods


def test_alfredo_tool_from_alfredo() -> None:
    """Test creating AlfredoTool from Alfredo tool ID."""
    # Import handlers to register tools
    from alfredo.tools.handlers import todo  # noqa: F401

    tool = AlfredoTool.from_alfredo(
        tool_id="write_todo_list",
        cwd=".",
        system_instructions={
            "agent": "Track your progress",
        },
    )

    assert tool.name == "write_todo_list"
    assert tool.get_instruction_for_node("agent") == "Track your progress"


def test_alfredo_tool_from_mcp(mock_langchain_tool: StructuredTool) -> None:
    """Test creating AlfredoTool from MCP tool."""
    tool = AlfredoTool.from_mcp(
        mock_langchain_tool,
        system_instructions={
            "agent": "Use for external operations",
        },
    )

    assert tool.name == "mock_tool"
    assert tool.get_instruction_for_node("agent") == "Use for external operations"


# Test metadata support


def test_alfredo_tool_with_metadata(mock_langchain_tool: StructuredTool) -> None:
    """Test AlfredoTool with metadata."""
    metadata = {
        "category": "filesystem",
        "version": "1.0",
    }

    tool = AlfredoTool.from_langchain(
        mock_langchain_tool,
        metadata=metadata,
    )

    assert tool.metadata == metadata
    assert tool.metadata["category"] == "filesystem"


# Test integration with prompt building


def test_extract_instructions_for_node() -> None:
    """Test extracting instructions for specific node from tool list."""
    from alfredo.agentic.prompts import _extract_instructions_for_node
    from alfredo.tools.handlers import todo, workflow  # noqa: F401

    tool1 = AlfredoTool.from_alfredo(
        "write_todo_list",
        system_instructions={
            "agent": "Instruction 1",
        },
    )

    tool2 = AlfredoTool.from_alfredo(
        "attempt_completion",
        system_instructions={
            "agent": "Instruction 2",
            "planner": "Planner instruction",
        },
    )

    tools = [tool1, tool2]

    # Extract for agent node
    agent_instructions = _extract_instructions_for_node(tools, "agent")
    assert "Instruction 1" in agent_instructions
    assert "Instruction 2" in agent_instructions

    # Extract for planner node
    planner_instructions = _extract_instructions_for_node(tools, "planner")
    assert "Planner instruction" in planner_instructions
    assert "Instruction 1" not in planner_instructions

    # Extract for node with no instructions
    verifier_instructions = _extract_instructions_for_node(tools, "verifier")
    assert verifier_instructions == ""


# Test backward compatibility


def test_mixed_tool_list(mock_langchain_tool: StructuredTool) -> None:
    """Test working with mixed list of AlfredoTools and plain StructuredTools."""
    from alfredo.agentic.prompts import _extract_instructions_for_node

    alfredo_tool = AlfredoTool.from_langchain(
        mock_langchain_tool,
        system_instructions={"agent": "Alfredo instruction"},
    )

    # Plain LangChain tool (no instructions)
    plain_tool = StructuredTool.from_function(
        func=lambda x: x,
        name="plain_tool",
        description="Plain tool",
    )

    tools = [alfredo_tool, plain_tool]

    # Should only extract from AlfredoTools
    instructions = _extract_instructions_for_node(tools, "agent")
    assert "Alfredo instruction" in instructions


# Test tool normalization


def test_normalize_tools() -> None:
    """Test normalizing mixed tool list to AlfredoTools."""
    from alfredo.agentic.graph import _normalize_tools
    from alfredo.tools.handlers import todo  # noqa: F401

    # Create mixed list
    alfredo_tool = AlfredoTool.from_alfredo("write_todo_list")
    plain_tool = StructuredTool.from_function(
        func=lambda x: x,
        name="plain_tool",
        description="Plain tool",
    )

    mixed_tools = [alfredo_tool, plain_tool]
    normalized = _normalize_tools(mixed_tools)

    # All should be AlfredoTools
    assert all(isinstance(t, AlfredoTool) for t in normalized)
    assert len(normalized) == 2

    # Original AlfredoTool should be unchanged
    assert normalized[0] == alfredo_tool

    # Plain tool should be wrapped
    assert normalized[1].name == "plain_tool"


# Test todo tool instructions


def test_todo_tool_instructions() -> None:
    """Test that todo tools have proper system instructions defined."""
    from alfredo.tools.handlers.todo import TODO_SYSTEM_INSTRUCTIONS

    assert "planner" in TODO_SYSTEM_INSTRUCTIONS
    assert "agent" in TODO_SYSTEM_INSTRUCTIONS
    assert "write_todo_list" in TODO_SYSTEM_INSTRUCTIONS["planner"]
    assert "sequential" in TODO_SYSTEM_INSTRUCTIONS["agent"].lower()


def test_create_todo_alfredo_tools() -> None:
    """Test creating todo tools as AlfredoTools with instructions."""
    from alfredo.integrations.langchain import create_alfredo_tools
    from alfredo.tools.handlers.todo import TODO_SYSTEM_INSTRUCTIONS

    tools = create_alfredo_tools(
        tool_ids=["write_todo_list", "read_todo_list"],
        tool_configs={
            "write_todo_list": TODO_SYSTEM_INSTRUCTIONS,
            "read_todo_list": TODO_SYSTEM_INSTRUCTIONS,
        },
    )

    assert len(tools) == 3  # write_todo_list, read_todo_list, attempt_completion
    write_todo = next(t for t in tools if t.name == "write_todo_list")

    assert write_todo.is_available_for_node("agent")
    assert write_todo.is_available_for_node("planner")
    assert write_todo.get_instruction_for_node("agent") is not None


# Test LangChain integration helpers


def test_create_alfredo_tool_helper() -> None:
    """Test create_alfredo_tool helper function."""
    from alfredo.integrations.langchain import create_alfredo_tool

    tool = create_alfredo_tool(
        tool_id="read_file",
        cwd=".",
        system_instructions={
            "agent": "Test instruction",
        },
    )

    assert isinstance(tool, AlfredoTool)
    assert tool.name == "read_file"
    assert tool.get_instruction_for_node("agent") == "Test instruction"


def test_wrap_langchain_tool_helper(mock_langchain_tool: StructuredTool) -> None:
    """Test wrap_langchain_tool helper function."""
    from alfredo.integrations.langchain import wrap_langchain_tool

    tool = wrap_langchain_tool(
        mock_langchain_tool,
        system_instructions={
            "agent": "Wrapped instruction",
        },
    )

    assert isinstance(tool, AlfredoTool)
    assert tool.name == "mock_tool"
    assert tool.get_instruction_for_node("agent") == "Wrapped instruction"


# Test string representations


def test_alfredo_tool_repr(mock_langchain_tool: StructuredTool) -> None:
    """Test AlfredoTool string representation."""
    tool = AlfredoTool.from_langchain(
        mock_langchain_tool,
        system_instructions={
            "agent": "Test",
            "planner": "Test",
        },
    )

    repr_str = repr(tool)
    assert "mock_tool" in repr_str
    assert "targets=" in repr_str


def test_alfredo_tool_str(mock_langchain_tool: StructuredTool) -> None:
    """Test AlfredoTool __str__ method."""
    tool = AlfredoTool.from_langchain(mock_langchain_tool)
    str_repr = str(tool)
    assert "mock_tool" in str_repr


# Test prompt formatting


def test_prompt_formatting_with_tool_instructions() -> None:
    """Test that tool instructions are added as a proper section in prompts."""
    from alfredo.agentic.prompts import get_agent_system_prompt, get_planning_prompt
    from alfredo.tools.handlers import todo  # noqa: F401

    # Create tools with instructions
    tools = [
        AlfredoTool.from_alfredo(
            "write_todo_list",
            system_instructions={
                "agent": "Agent instruction here",
                "planner": "Planner instruction here",
            },
        )
    ]

    # Test agent prompt
    agent_prompt = get_agent_system_prompt("Test task", "Test plan", tools=tools)
    assert "# Tool-Specific Instructions" in agent_prompt
    assert "Agent instruction here" in agent_prompt

    # Test planner prompt
    planner_prompt = get_planning_prompt("Test task", tools=tools)
    assert "# Tool-Specific Instructions" in planner_prompt
    assert "Planner instruction here" in planner_prompt


def test_prompt_without_tool_instructions() -> None:
    """Test that prompts work correctly without any tool instructions."""
    from alfredo.agentic.prompts import get_agent_system_prompt

    # No tools or tools without instructions for this node
    agent_prompt = get_agent_system_prompt("Test task", "Test plan", tools=None)
    assert "# Tool-Specific Instructions" not in agent_prompt

    # Tool with instructions for different node
    tools = [
        AlfredoTool.from_alfredo(
            "write_todo_list",
            system_instructions={
                "planner": "Planner only",
            },
        )
    ]
    agent_prompt = get_agent_system_prompt("Test task", "Test plan", tools=tools)
    assert "# Tool-Specific Instructions" not in agent_prompt
