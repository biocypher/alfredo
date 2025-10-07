"""Tests for the Agent class."""

from pathlib import Path
from tempfile import TemporaryDirectory

from alfredo import Agent


def test_agent_initialization() -> None:
    """Test that agent initializes correctly."""
    agent = Agent()
    assert agent.cwd is not None
    assert agent.model_family is not None


def test_get_system_prompt() -> None:
    """Test system prompt generation."""
    agent = Agent()
    prompt = agent.get_system_prompt()

    assert "Tools" in prompt or "tools" in prompt.lower()
    assert "read_file" in prompt
    assert "write_to_file" in prompt
    assert "execute_command" in prompt


def test_get_available_tools() -> None:
    """Test getting available tools list."""
    agent = Agent()
    tools = agent.get_available_tools()

    assert len(tools) > 0
    assert "read_file" in tools
    assert "write_to_file" in tools
    assert "list_files" in tools


def test_parse_tool_use() -> None:
    """Test parsing tool invocations."""
    agent = Agent()

    # Test valid tool use
    text = """
    <read_file>
    <path>test.txt</path>
    </read_file>
    """
    tool_use = agent.parse_tool_use(text)

    assert tool_use is not None
    assert tool_use.name == "read_file"
    assert tool_use.params["path"] == "test.txt"


def test_execute_read_file() -> None:
    """Test reading a file through the agent."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a test file
        test_file = tmppath / "test.txt"
        test_file.write_text("Hello, World!")

        # Create agent with temp directory
        agent = Agent(cwd=str(tmppath))

        # Execute read_file
        text = """
        <read_file>
        <path>test.txt</path>
        </read_file>
        """
        result = agent.execute_from_text(text)

        assert result is not None
        assert result.success is True
        assert "Hello, World!" in result.output


def test_execute_write_file() -> None:
    """Test writing a file through the agent."""
    with TemporaryDirectory() as tmpdir:
        agent = Agent(cwd=tmpdir)

        # Execute write_to_file
        text = """
        <write_to_file>
        <path>output.txt</path>
        <content>Test content</content>
        </write_to_file>
        """
        result = agent.execute_from_text(text)

        assert result is not None
        assert result.success is True

        # Verify file was created
        output_file = Path(tmpdir) / "output.txt"
        assert output_file.exists()
        assert output_file.read_text() == "Test content"


def test_execute_list_files() -> None:
    """Test listing files through the agent."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create some test files
        (tmppath / "file1.txt").write_text("content1")
        (tmppath / "file2.txt").write_text("content2")

        agent = Agent(cwd=str(tmppath))

        # Execute list_files
        text = """
        <list_files>
        <path>.</path>
        </list_files>
        """
        result = agent.execute_from_text(text)

        assert result is not None
        assert result.success is True
        assert "file1.txt" in result.output
        assert "file2.txt" in result.output
