"""Tests for LangChain integration."""

import pytest

# Check if LangChain is available
try:
    from langchain_core.tools import StructuredTool  # noqa: F401

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from pathlib import Path
from tempfile import TemporaryDirectory

pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")

if LANGCHAIN_AVAILABLE:
    from alfredo.integrations.langchain import (
        create_langchain_tool,
        create_langchain_tools,
        create_pydantic_model_from_spec,
    )
    from alfredo.tools.registry import registry
    from alfredo.tools.specs import ModelFamily


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_create_pydantic_model_from_spec() -> None:
    """Test creating Pydantic model from ToolSpec."""
    # Import to register tools
    from alfredo.tools.handlers import file_ops  # noqa: F401

    spec = registry.get_spec("read_file", ModelFamily.GENERIC)
    assert spec is not None

    model = create_pydantic_model_from_spec(spec)

    # Check model has the expected field
    assert "path" in model.model_fields
    assert model.model_fields["path"].is_required()


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_create_langchain_tool() -> None:
    """Test creating a LangChain tool from Alfredo tool."""
    tool = create_langchain_tool("read_file")

    assert tool.name == "read_file"
    assert "read" in tool.description.lower()
    assert hasattr(tool, "args_schema")
    assert "path" in tool.args_schema.model_fields  # type: ignore[union-attr]


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_create_langchain_tools() -> None:
    """Test creating all LangChain tools."""
    tools = create_langchain_tools()

    assert len(tools) > 0
    assert all(hasattr(tool, "name") for tool in tools)
    assert all(hasattr(tool, "description") for tool in tools)

    # Check expected tools are present
    tool_names = [tool.name for tool in tools]
    assert "read_file" in tool_names
    assert "write_to_file" in tool_names
    assert "list_files" in tool_names


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_langchain_tool_invoke() -> None:
    """Test invoking a LangChain tool."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a test file
        test_file = tmppath / "test.txt"
        test_file.write_text("Test content")

        # Create tool with temp directory
        tool = create_langchain_tool("read_file", cwd=str(tmppath))

        # Invoke the tool
        result = tool.invoke({"path": "test.txt"})

        assert "Test content" in result


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_langchain_tool_write_and_read() -> None:
    """Test write and read operations through LangChain tools."""
    with TemporaryDirectory() as tmpdir:
        # Create tools
        write_tool = create_langchain_tool("write_to_file", cwd=tmpdir)
        read_tool = create_langchain_tool("read_file", cwd=tmpdir)

        # Write a file
        write_result = write_tool.invoke({"path": "output.txt", "content": "LangChain test"})
        assert "output.txt" in write_result

        # Read it back
        read_result = read_tool.invoke({"path": "output.txt"})
        assert "LangChain test" in read_result


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_langchain_tool_list_files() -> None:
    """Test list_files through LangChain."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create some files
        (tmppath / "file1.txt").write_text("content1")
        (tmppath / "file2.txt").write_text("content2")

        # Create tool
        tool = create_langchain_tool("list_files", cwd=str(tmppath))

        # Invoke
        result = tool.invoke({"path": ".", "recursive": "false"})

        assert "file1.txt" in result
        assert "file2.txt" in result


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_langchain_tool_error_handling() -> None:
    """Test that errors are handled gracefully."""
    tool = create_langchain_tool("read_file")

    # Try to read a non-existent file
    result = tool.invoke({"path": "nonexistent.txt"})

    assert "Error" in result or "not found" in result.lower()


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_specific_tools_conversion() -> None:
    """Test converting specific tools only.

    Note: attempt_completion is always auto-added to ensure agent can exit react loop.
    """
    tool_ids = ["read_file", "write_to_file"]
    tools = create_langchain_tools(tool_ids=tool_ids)

    # Now expects 3 tools: read_file, write_to_file, and auto-added attempt_completion
    assert len(tools) == 3
    tool_names = [tool.name for tool in tools]
    assert "read_file" in tool_names
    assert "write_to_file" in tool_names
    assert "attempt_completion" in tool_names  # Auto-added


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_langchain_list_code_definitions() -> None:
    """Test list_code_definition_names through LangChain."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a sample Python file
        (tmppath / "sample.py").write_text(
            """
def my_function():
    pass

class MyClass:
    pass
"""
        )

        # Create tool
        tool = create_langchain_tool("list_code_definition_names", cwd=str(tmppath))

        # Invoke
        result = tool.invoke({"path": "."})

        assert "sample.py" in result
        assert "my_function" in result
        assert "MyClass" in result


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_langchain_web_fetch() -> None:
    """Test web_fetch tool conversion to LangChain."""
    from unittest.mock import Mock, patch

    # Create mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = "<html><body><h1>Test</h1></body></html>"
    mock_response.raise_for_status = Mock()

    with patch("alfredo.tools.handlers.web.requests.get", return_value=mock_response):
        # Create tool
        tool = create_langchain_tool("web_fetch")

        # Invoke
        result = tool.invoke({"url": "https://example.com"})

        assert "Test" in result
        assert "example.com" in result


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_new_tools_in_all_tools_list() -> None:
    """Test that new tools are included in create_langchain_tools."""
    tools = create_langchain_tools()

    tool_names = [tool.name for tool in tools]

    # Check new tools are present
    assert "list_code_definition_names" in tool_names
    assert "web_fetch" in tool_names


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_attempt_completion_always_included_with_all_tools() -> None:
    """Test that attempt_completion is always included when getting all tools."""
    tools = create_langchain_tools()

    tool_names = [tool.name for tool in tools]

    # attempt_completion must be present
    assert "attempt_completion" in tool_names, "attempt_completion must be included in default tools"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_attempt_completion_always_included_with_custom_tools() -> None:
    """Test that attempt_completion is always included even with custom tool list."""
    # Request only specific tools (excluding attempt_completion)
    tool_ids = ["read_file", "write_to_file"]
    tools = create_langchain_tools(tool_ids=tool_ids)

    tool_names = [tool.name for tool in tools]

    # attempt_completion must be auto-added
    assert "attempt_completion" in tool_names, "attempt_completion must be auto-added to custom tool lists"
    assert "read_file" in tool_names
    assert "write_to_file" in tool_names


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_attempt_completion_not_duplicated() -> None:
    """Test that attempt_completion is not duplicated if explicitly included."""
    # Request tools including attempt_completion
    tool_ids = ["read_file", "attempt_completion"]
    tools = create_langchain_tools(tool_ids=tool_ids)

    tool_names = [tool.name for tool in tools]

    # Should appear exactly once
    assert tool_names.count("attempt_completion") == 1, "attempt_completion should appear exactly once"
