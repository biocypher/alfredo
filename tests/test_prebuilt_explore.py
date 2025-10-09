"""Tests for prebuilt ExplorationAgent and extended read_file tool."""

from pathlib import Path

import pytest

from alfredo.tools.handlers.file_ops import ReadFileHandler
from alfredo.tools.registry import registry


class TestReadFileWithOffsetLimit:
    """Test the extended read_file tool with offset and limit parameters."""

    def test_read_full_file(self, tmp_path: Path) -> None:
        """Test reading a full file without offset or limit."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        test_file.write_text(content)

        # Read the file
        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt"})

        assert result.success
        assert result.output == content

    def test_read_with_limit(self, tmp_path: Path) -> None:
        """Test reading a file with limit parameter."""
        # Create a test file with 10 lines
        test_file = tmp_path / "test.txt"
        lines = [f"Line {i}\n" for i in range(1, 11)]
        test_file.write_text("".join(lines))

        # Read only first 3 lines
        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt", "limit": "3"})

        assert result.success
        assert "[Showing lines 1-3 of 10 total lines]" in result.output
        assert "Line 1\n" in result.output
        assert "Line 2\n" in result.output
        assert "Line 3\n" in result.output
        assert "Line 4" not in result.output

    def test_read_with_offset(self, tmp_path: Path) -> None:
        """Test reading a file with offset parameter."""
        # Create a test file with 10 lines
        test_file = tmp_path / "test.txt"
        lines = [f"Line {i}\n" for i in range(1, 11)]
        test_file.write_text("".join(lines))

        # Read from line 5 (0-indexed, so offset=4)
        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt", "offset": "4"})

        assert result.success
        assert "[Showing lines 5-10 of 10 total lines]" in result.output
        assert "Line 1\n" not in result.output  # More specific - with newline
        assert "Line 5\n" in result.output
        assert "Line 10\n" in result.output

    def test_read_with_offset_and_limit(self, tmp_path: Path) -> None:
        """Test reading a file with both offset and limit."""
        # Create a test file with 10 lines
        test_file = tmp_path / "test.txt"
        lines = [f"Line {i}\n" for i in range(1, 11)]
        test_file.write_text("".join(lines))

        # Read 3 lines starting from line 5 (offset=4, limit=3)
        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt", "offset": "4", "limit": "3"})

        assert result.success
        assert "[Showing lines 5-7 of 10 total lines]" in result.output
        assert "Line 5\n" in result.output
        assert "Line 6\n" in result.output
        assert "Line 7\n" in result.output
        assert "Line 8" not in result.output

    def test_offset_exceeds_file_length(self, tmp_path: Path) -> None:
        """Test error handling when offset exceeds file length."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\n")

        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt", "offset": "10"})

        assert not result.success
        assert result.error is not None
        assert "exceeds total lines" in result.error

    def test_invalid_offset(self, tmp_path: Path) -> None:
        """Test error handling for invalid offset value."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\n")

        handler = ReadFileHandler(cwd=tmp_path)

        # Negative offset
        result = handler.execute({"path": "test.txt", "offset": "-5"})
        assert not result.success
        assert result.error is not None
        assert "non-negative" in result.error

        # Invalid type
        result = handler.execute({"path": "test.txt", "offset": "abc"})
        assert not result.success
        assert result.error is not None
        assert "Invalid offset value" in result.error

    def test_invalid_limit(self, tmp_path: Path) -> None:
        """Test error handling for invalid limit value."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\n")

        handler = ReadFileHandler(cwd=tmp_path)

        # Zero limit
        result = handler.execute({"path": "test.txt", "limit": "0"})
        assert not result.success
        assert result.error is not None
        assert "must be positive" in result.error

        # Negative limit
        result = handler.execute({"path": "test.txt", "limit": "-5"})
        assert not result.success
        assert result.error is not None
        assert "must be positive" in result.error

        # Invalid type
        result = handler.execute({"path": "test.txt", "limit": "xyz"})
        assert not result.success
        assert result.error is not None
        assert "Invalid limit value" in result.error

    def test_limit_exceeds_remaining_lines(self, tmp_path: Path) -> None:
        """Test reading when limit exceeds remaining lines after offset."""
        test_file = tmp_path / "test.txt"
        lines = [f"Line {i}\n" for i in range(1, 6)]
        test_file.write_text("".join(lines))

        # Offset=2, limit=10 (but only 3 lines remain)
        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt", "offset": "2", "limit": "10"})

        assert result.success
        assert "[Showing lines 3-5 of 5 total lines]" in result.output
        assert "Line 3\n" in result.output
        assert "Line 5\n" in result.output

    def test_read_with_limit_bytes(self, tmp_path: Path) -> None:
        """Test reading a file with byte limit."""
        test_file = tmp_path / "test.txt"
        content = "This is a test file with some content.\n" * 10
        test_file.write_text(content)

        # Read only first 100 bytes
        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt", "limit_bytes": "100"})

        assert result.success
        assert "[Showing first" in result.output
        assert "bytes of" in result.output
        # Should have truncated content
        assert len(result.output.encode("utf-8")) < len(content.encode("utf-8"))

    def test_limit_and_limit_bytes_mutual_exclusion(self, tmp_path: Path) -> None:
        """Test that limit and limit_bytes cannot be used together."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\n")

        handler = ReadFileHandler(cwd=tmp_path)
        result = handler.execute({"path": "test.txt", "limit": "5", "limit_bytes": "100"})

        assert not result.success
        assert result.error is not None
        assert "Cannot use both" in result.error


class TestExplorationAgent:
    """Test the ExplorationAgent prebuilt agent."""

    def test_agent_initialization(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that ExplorationAgent initializes correctly."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        # Set fake API key to avoid initialization errors
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

        agent = ExplorationAgent(
            cwd=str(tmp_path),
            model_name="gpt-4.1-mini",
            verbose=False,
        )

        assert agent.cwd == tmp_path.resolve()
        assert agent.model_name == "gpt-4.1-mini"
        assert agent.max_file_size_bytes == 100_000  # Default
        assert agent.preview_kb == 50  # Default (when neither is set)
        assert agent.preview_lines is None
        assert agent.preview_bytes == 50 * 1024  # Converted to bytes

    def test_agent_with_context(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ExplorationAgent with context prompt."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        # Set fake API key
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

        context = "Focus on Python files and data schemas"
        agent = ExplorationAgent(
            cwd=str(tmp_path),
            context_prompt=context,
            verbose=False,
        )

        assert agent.context_prompt == context
        # Verify context is incorporated into planner prompt
        planner_prompt = agent._build_planner_prompt()
        assert context in planner_prompt

    def test_agent_output_path_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default output path is set correctly."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        # Set fake API key
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

        agent = ExplorationAgent(cwd=str(tmp_path), verbose=False)

        expected_path = tmp_path / "notes" / "exploration_report.md"
        assert agent.output_path == expected_path

    def test_agent_output_path_custom(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that custom output path is respected."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        # Set fake API key
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

        custom_path = tmp_path / "my_report.md"
        agent = ExplorationAgent(
            cwd=str(tmp_path),
            output_path=str(custom_path),
            verbose=False,
        )

        assert agent.output_path == custom_path

    def test_planner_prompt_structure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the planner prompt has the expected structure."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        # Set fake API key
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

        agent = ExplorationAgent(cwd=str(tmp_path), verbose=False)
        prompt = agent._build_planner_prompt()

        # Check for key sections
        assert "{task}" in prompt
        assert "{tool_instructions}" in prompt
        assert "Directory Listing" in prompt
        assert "Categorize Files" in prompt
        assert "Smart File Reading" in prompt
        assert "Data File Analysis" in prompt
        assert "Generate Markdown Report" in prompt
        assert "attempt_completion" in prompt

    def test_agent_with_preview_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ExplorationAgent with preview_lines parameter."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        # Set fake API key
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

        agent = ExplorationAgent(
            cwd=str(tmp_path),
            preview_lines=100,  # Use line-based preview
            verbose=False,
        )

        assert agent.preview_lines == 100
        assert agent.preview_kb is None
        assert agent.preview_bytes is None

    def test_data_file_extensions(self) -> None:
        """Test that expected data file extensions are defined."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        expected_extensions = {".csv", ".xlsx", ".h5", ".hdf5", ".parquet", ".json"}
        assert expected_extensions.issubset(ExplorationAgent.DATA_EXTENSIONS)

    def test_size_thresholds(self) -> None:
        """Test that size thresholds are defined."""
        try:
            from alfredo.prebuilt import ExplorationAgent
        except ImportError:
            pytest.skip("LangGraph not installed")

        assert ExplorationAgent.SIZE_SMALL == 10_000
        assert ExplorationAgent.SIZE_MEDIUM == 100_000
        assert ExplorationAgent.SIZE_LARGE == 1_000_000

    @pytest.mark.skipif(
        not pytest.importorskip("alfredo.prebuilt", reason="LangGraph not installed"),
        reason="Requires LangGraph",
    )
    def test_explore_creates_fixture_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test exploration of a simple fixture directory.

        Note: This is a minimal integration test that doesn't call the LLM.
        Full end-to-end testing would require API keys and is better done manually.
        """
        # Set fake API key
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

        # Create fixture directory structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main():\n    print('Hello')\n")
        (tmp_path / "README.md").write_text("# Test Project\n")
        (tmp_path / "data").mkdir()

        # Create a small CSV file
        csv_content = "id,name,value\n1,Alice,100\n2,Bob,200\n"
        (tmp_path / "data" / "test.csv").write_text(csv_content)

        # Just test initialization and prompt building, not actual execution
        # (which would require API keys)
        from alfredo.prebuilt import ExplorationAgent

        agent = ExplorationAgent(
            cwd=str(tmp_path),
            max_file_size_bytes=50_000,
            preview_kb=25,
            verbose=False,
        )

        # Verify agent is properly configured
        assert agent.cwd == tmp_path.resolve()

        # Verify planner prompt mentions data files and pandas
        prompt = agent._build_planner_prompt()
        assert "pandas" in prompt.lower()
        assert "csv" in prompt.lower()


class TestReadFileToolRegistration:
    """Test that the read_file tool is properly registered with new parameters."""

    def test_tool_spec_has_offset_parameter(self) -> None:
        """Test that read_file spec includes offset parameter."""
        spec = registry.get_spec("read_file")
        assert spec is not None

        param_names = [p.name for p in spec.parameters]
        assert "offset" in param_names

    def test_tool_spec_has_limit_parameter(self) -> None:
        """Test that read_file spec includes limit parameter."""
        spec = registry.get_spec("read_file")
        assert spec is not None

        param_names = [p.name for p in spec.parameters]
        assert "limit" in param_names

    def test_offset_parameter_is_optional(self) -> None:
        """Test that offset parameter is not required."""
        spec = registry.get_spec("read_file")
        assert spec is not None

        offset_param = next(p for p in spec.parameters if p.name == "offset")
        assert not offset_param.required

    def test_limit_parameter_is_optional(self) -> None:
        """Test that limit parameter is not required."""
        spec = registry.get_spec("read_file")
        assert spec is not None

        limit_param = next(p for p in spec.parameters if p.name == "limit")
        assert not limit_param.required
