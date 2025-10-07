"""Tests for the agentic graph structure."""

import os

import pytest

try:
    from langchain_core.messages import AIMessage, HumanMessage

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_create_agentic_graph() -> None:
    """Test that the graph can be created without errors."""
    from alfredo.agentic import create_agentic_graph

    # Requires API key to initialize the model
    graph = create_agentic_graph(
        cwd=".",
        model_name="gpt-4o-mini",
        max_context_tokens=100000,
    )

    assert graph is not None


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_graph_has_expected_nodes() -> None:
    """Test that the graph contains all expected nodes."""
    from alfredo.agentic import create_agentic_graph

    graph = create_agentic_graph(cwd=".")

    # Get the graph structure
    nodes = graph.get_graph().nodes

    # Check that all expected nodes are present
    expected_nodes = {"planner", "agent", "tools", "verifier", "replan"}

    # Node keys include special START and END nodes
    node_names = {name for name in nodes.keys() if not name.startswith("__")}

    assert expected_nodes.issubset(node_names), f"Missing nodes: {expected_nodes - node_names}"


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_graph_edges() -> None:
    """Test that the graph has expected edge structure."""
    from alfredo.agentic import create_agentic_graph

    graph = create_agentic_graph(cwd=".")

    # Get the graph structure
    graph_obj = graph.get_graph()

    # Verify key edges exist
    # Note: edges are represented as (from_node, to_node) tuples
    edges = graph_obj.edges

    # Check that we have edges (exact structure depends on conditional edges)
    assert len(edges) > 0, "Graph should have edges"


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_should_continue_function() -> None:
    """Test the should_continue routing function."""
    from alfredo.agentic.graph import should_continue
    from alfredo.agentic.state import AgentState

    # Test with no messages
    state: AgentState = {
        "messages": [],
        "task": "test",
        "plan": "",
        "plan_iteration": 0,
        "max_context_tokens": 100000,
        "final_answer": None,
        "is_verified": False,
    }
    result = should_continue(state)
    assert result == "agent"

    # Test with AI message with tool calls (non-attempt_answer)
    from langchain_core.messages import AIMessage

    state_with_tool = state.copy()
    state_with_tool["messages"] = [
        AIMessage(
            content="Using tool",
            tool_calls=[{"name": "read_file", "args": {"path": "test.py"}, "id": "1"}],
        )
    ]
    result = should_continue(state_with_tool)
    assert result == "tools"

    # Test with attempt_completion tool call (should also go to tools first)
    state_with_answer = state.copy()
    state_with_answer["messages"] = [
        AIMessage(
            content="Completing task",
            tool_calls=[{"name": "attempt_completion", "args": {"result": "Done"}, "id": "2"}],
        )
    ]
    result = should_continue(state_with_answer)
    # ALL tool calls go to tools node now, then route_after_tools handles routing to verifier
    assert result == "tools"


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_verification_router() -> None:
    """Test the verification_router function."""
    from alfredo.agentic.graph import verification_router
    from alfredo.agentic.state import AgentState

    # Test verified case
    verified_state: AgentState = {
        "messages": [],
        "task": "test",
        "plan": "",
        "plan_iteration": 0,
        "max_context_tokens": 100000,
        "final_answer": "Answer",
        "is_verified": True,
    }
    result = verification_router(verified_state)
    assert result == "__end__"

    # Test not verified case
    not_verified_state: AgentState = {
        "messages": [],
        "task": "test",
        "plan": "",
        "plan_iteration": 0,
        "max_context_tokens": 100000,
        "final_answer": "Answer",
        "is_verified": False,
    }
    result = verification_router(not_verified_state)
    assert result == "replan"


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_format_execution_trace() -> None:
    """Test formatting execution trace from messages."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from alfredo.agentic.nodes import format_execution_trace

    # Test with no messages
    trace = format_execution_trace([])
    assert trace == "No actions recorded."

    # Test with tool calls and results
    messages = [
        HumanMessage(content="Task: Create a file"),
        AIMessage(
            content="I'll create the file",
            tool_calls=[{"name": "write_file", "args": {"path": "test.txt", "content": "hello"}, "id": "1"}],
        ),
        ToolMessage(content="File created successfully", tool_call_id="1"),
        AIMessage(
            content="Completing",
            tool_calls=[{"name": "attempt_completion", "args": {"result": "Done"}, "id": "2"}],
        ),
        ToolMessage(content="[TASK_COMPLETE]\nDone", tool_call_id="2"),
    ]

    trace = format_execution_trace(messages)

    # Should include tool names and results
    assert "write_file" in trace
    assert "attempt_completion" in trace
    assert "Step 1" in trace
    assert "Step 2" in trace
    assert "Task completion signal" in trace


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_extract_attempt_completion() -> None:
    """Test extracting result from attempt_completion tool call.

    The attempt_completion tool always returns [TASK_COMPLETE] marker,
    so we only need to check ToolMessages.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    from alfredo.agentic.nodes import extract_attempt_completion
    from alfredo.agentic.state import AgentState

    # Test with no messages
    state: AgentState = {
        "messages": [],
        "task": "test",
        "plan": "",
        "plan_iteration": 0,
        "max_context_tokens": 100000,
        "final_answer": None,
        "is_verified": False,
    }
    result = extract_attempt_completion(state)
    assert result == ""

    # Test with ToolMessage containing [TASK_COMPLETE] (standard case)
    state_with_completion = state.copy()
    state_with_completion["messages"] = [
        ToolMessage(
            content="[TASK_COMPLETE]\nTask completed successfully",
            tool_call_id="1",
        ),
    ]
    result = extract_attempt_completion(state_with_completion)
    assert result == "Task completed successfully"

    # Test with multi-line result
    state_multiline = state.copy()
    state_multiline["messages"] = [
        ToolMessage(
            content="[TASK_COMPLETE]\nI completed the task by:\n1. Creating file X\n2. Running tests",
            tool_call_id="1",
        ),
    ]
    result = extract_attempt_completion(state_multiline)
    assert result == "I completed the task by:\n1. Creating file X\n2. Running tests"

    # Test with command included
    state_with_command = state.copy()
    state_with_command["messages"] = [
        ToolMessage(
            content="[TASK_COMPLETE]\nTask done\n\nFinal command executed: pytest",
            tool_call_id="1",
        ),
    ]
    result = extract_attempt_completion(state_with_command)
    assert result == "Task done"

    # Test with other tool messages (no [TASK_COMPLETE])
    state_with_other = state.copy()
    state_with_other["messages"] = [
        ToolMessage(
            content="File contents: hello world",
            tool_call_id="2",
        )
    ]
    result = extract_attempt_completion(state_with_other)
    assert result == ""

    # Test with mixed messages (should find the completion)
    state_mixed = state.copy()
    state_mixed["messages"] = [
        AIMessage(content="Reading file"),
        ToolMessage(content="File read successfully", tool_call_id="1"),
        AIMessage(content="Completing task"),
        ToolMessage(content="[TASK_COMPLETE]\nAll done!", tool_call_id="2"),
    ]
    result = extract_attempt_completion(state_mixed)
    assert result == "All done!"
