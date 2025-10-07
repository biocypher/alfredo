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
        create_all_langchain_tools,
        create_langchain_tool,
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
    assert "path" in tool.args_schema.model_fields


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_create_all_langchain_tools() -> None:
    """Test creating all LangChain tools."""
    tools = create_all_langchain_tools()

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
    """Test converting specific tools only."""
    tool_ids = ["read_file", "write_to_file"]
    tools = create_all_langchain_tools(tool_ids=tool_ids)

    assert len(tools) == 2
    tool_names = [tool.name for tool in tools]
    assert "read_file" in tool_names
    assert "write_to_file" in tool_names
