"""Tests for the Agent class (agentic scaffold)."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Test if LangGraph is available
try:
    from alfredo import Agent

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_agent_initialization() -> None:
    """Test that agent initializes correctly."""
    agent = Agent(cwd=".")
    assert agent.cwd == "."
    assert agent.model_name == "gpt-4.1-mini"  # default
    assert agent.graph is not None
    assert agent.results is None  # No run yet


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_agent_run_simple_task() -> None:
    """Test running a simple file creation task."""
    with TemporaryDirectory() as tmpdir:
        agent = Agent(cwd=tmpdir, model_name="gpt-4.1-mini", verbose=False)

        # Run a simple task
        task = "Create a file called test.txt with the content 'Hello, World!'"
        result = agent.run(task)

        # Check result structure
        assert result is not None
        assert "messages" in result
        assert "task" in result
        assert "final_answer" in result
        assert "is_verified" in result

        # Check that agent stores results
        assert agent.results is not None
        assert agent.results == result

        # Check that file was created
        test_file = Path(tmpdir) / "test.txt"
        assert test_file.exists()
        assert "Hello, World!" in test_file.read_text()


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_agent_display_trace() -> None:
    """Test display_trace method."""
    with TemporaryDirectory() as tmpdir:
        agent = Agent(cwd=tmpdir, model_name="gpt-4.1-mini", verbose=False)

        # Run a task first
        task = "List all files in the current directory"
        agent.run(task)

        # display_trace should not raise an error
        # (We can't easily test the output, but we can ensure it doesn't crash)
        agent.display_trace()


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_agent_display_trace_without_run() -> None:
    """Test that display_trace raises error if run() wasn't called."""
    agent = Agent(cwd=".")

    with pytest.raises(RuntimeError, match="No execution results available"):
        agent.display_trace()


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_agent_custom_tools() -> None:
    """Test that agent accepts custom tools list."""
    from alfredo.integrations.langchain import create_langchain_tools

    # Create a limited toolset
    tools = create_langchain_tools(cwd=".", tool_ids=["list_files", "read_file"])

    agent = Agent(cwd=".", tools=tools, verbose=False)
    assert agent.tools is not None
    assert len(agent.tools) == 2


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_agent_custom_model_kwargs() -> None:
    """Test that agent accepts custom model kwargs."""
    agent = Agent(
        cwd=".",
        model_name="gpt-4.1-mini",
        temperature=0.5,
        max_tokens=1000,
        verbose=False,
    )

    assert agent.model_kwargs["temperature"] == 0.5
    assert agent.model_kwargs["max_tokens"] == 1000
