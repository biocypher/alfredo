"""Tests for OpenAI native integration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Try to import OpenAI integration
try:
    from alfredo.integrations.openai_native import (
        OPENAI_AVAILABLE,
        OpenAIAgent,
        get_all_tools_openai_format,
        tool_spec_to_openai_format,
    )
except ImportError:
    OPENAI_AVAILABLE = False


# Skip all tests if OpenAI is not installed
pytestmark = pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not installed")


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for tests."""
    return tmp_path


def test_tool_spec_to_openai_format() -> None:
    """Test converting ToolSpec to OpenAI format."""
    # Import to register tools
    from alfredo.tools.handlers import file_ops  # noqa: F401
    from alfredo.tools.registry import registry
    from alfredo.tools.specs import ModelFamily

    # Get read_file spec
    spec = registry.get_spec("read_file", ModelFamily.GENERIC)
    assert spec is not None

    # Convert to OpenAI format
    openai_tool = tool_spec_to_openai_format(spec)

    # Verify structure
    assert openai_tool["type"] == "function"
    assert "function" in openai_tool

    function = openai_tool["function"]
    assert function["name"] == "read_file"
    assert "description" in function
    assert function["strict"] is True

    # Check parameters
    params = function["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "required" in params

    # Check path parameter
    assert "path" in params["properties"]
    assert params["properties"]["path"]["type"] == "string"
    assert "description" in params["properties"]["path"]
    assert "path" in params["required"]


def test_get_all_tools_openai_format() -> None:
    """Test getting all tools in OpenAI format."""
    tools = get_all_tools_openai_format()

    # Should have multiple tools
    assert len(tools) > 0

    # Each tool should have correct structure
    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_get_specific_tools_openai_format() -> None:
    """Test getting specific tools in OpenAI format."""
    tools = get_all_tools_openai_format(tool_ids=["read_file", "write_to_file"])

    assert len(tools) == 2
    tool_names = [tool["function"]["name"] for tool in tools]
    assert "read_file" in tool_names
    assert "write_to_file" in tool_names


def test_openai_agent_initialization(temp_dir: Path) -> None:
    """Test OpenAI agent initialization."""
    with patch("alfredo.integrations.openai_native.OpenAI") as mock_openai:
        agent = OpenAIAgent(cwd=str(temp_dir), api_key="test-key", model="gpt-4o-mini")

        assert agent.cwd == str(temp_dir)
        assert agent.model == "gpt-4o-mini"
        mock_openai.assert_called_once()


def test_openai_agent_get_tools_definition(temp_dir: Path) -> None:
    """Test getting tools definition from agent."""
    with patch("alfredo.integrations.openai_native.OpenAI"):
        agent = OpenAIAgent(cwd=str(temp_dir))
        tools = agent.get_tools_definition()

        assert len(tools) > 0
        assert all(tool["type"] == "function" for tool in tools)


def test_openai_agent_tool_execution(temp_dir: Path) -> None:
    """Test tool execution through OpenAI agent."""
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello, world!")

    with patch("alfredo.integrations.openai_native.OpenAI") as mock_openai:
        # Mock the OpenAI API response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First call: Model requests to use read_file tool
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "read_file"
        mock_tool_call.function.arguments = json.dumps({"path": str(test_file)})

        mock_response_1 = MagicMock()
        mock_response_1.choices = [MagicMock()]
        mock_response_1.choices[0].message.content = None
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
        mock_response_1.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_123", "type": "function"}],
        }

        # Second call: Model provides final response
        mock_response_2 = MagicMock()
        mock_response_2.choices = [MagicMock()]
        mock_response_2.choices[0].message.content = "The file contains: Hello, world!"
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "The file contains: Hello, world!",
        }

        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]

        # Run the agent
        agent = OpenAIAgent(cwd=str(temp_dir))
        result = agent.run("Read the test.txt file")

        # Verify results
        assert result["content"] == "The file contains: Hello, world!"
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "read_file"
        assert result["tool_results"][0]["success"] is True
        assert "Hello, world!" in result["tool_results"][0]["output"]


def test_openai_agent_unknown_tool(temp_dir: Path) -> None:
    """Test handling of unknown tool."""
    with patch("alfredo.integrations.openai_native.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock tool call for unknown tool
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "unknown_tool"
        mock_tool_call.function.arguments = json.dumps({})

        mock_response_1 = MagicMock()
        mock_response_1.choices = [MagicMock()]
        mock_response_1.choices[0].message.content = None
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
        mock_response_1.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_123"}],
        }

        mock_response_2 = MagicMock()
        mock_response_2.choices = [MagicMock()]
        mock_response_2.choices[0].message.content = "I encountered an error"
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "I encountered an error",
        }

        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]

        agent = OpenAIAgent(cwd=str(temp_dir))
        result = agent.run("Use unknown tool")

        # Should have error in tool results
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["success"] is False
        assert "Unknown tool" in result["tool_results"][0]["error"]


def test_openai_agent_max_iterations(temp_dir: Path) -> None:
    """Test max iterations limit."""
    with patch("alfredo.integrations.openai_native.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Always return tool calls (infinite loop scenario)
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "read_file"
        mock_tool_call.function.arguments = json.dumps({"path": "test.txt"})

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_123"}],
        }

        mock_client.chat.completions.create.return_value = mock_response

        agent = OpenAIAgent(cwd=str(temp_dir))
        result = agent.run("Test", max_iterations=3)

        # Should hit max iterations
        assert "Maximum iterations" in result["content"]
        assert len(result["tool_results"]) == 3  # Should execute 3 times


def test_openai_agent_custom_system_prompt(temp_dir: Path) -> None:
    """Test using custom system prompt."""
    with patch("alfredo.integrations.openai_native.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Custom response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": "Custom response"}

        mock_client.chat.completions.create.return_value = mock_response

        agent = OpenAIAgent(cwd=str(temp_dir))
        custom_prompt = "You are a specialized code assistant."
        agent.run("Test", system_prompt=custom_prompt)

        # Verify custom prompt was used
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom_prompt
