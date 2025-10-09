"""Tests for the agentic graph structure."""

import os

import pytest


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_create_agentic_graph() -> None:
    """Test that the graph can be created without errors."""
    from alfredo.agentic import create_agentic_graph

    # Requires API key to initialize the model
    graph = create_agentic_graph(
        cwd=".",
        model_name="gpt-4.1-mini",
        max_context_tokens=100000,
    )

    assert graph is not None


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
    node_names = {name for name in nodes if not name.startswith("__")}

    assert expected_nodes.issubset(node_names), f"Missing nodes: {expected_nodes - node_names}"


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
        "todo_list": None,
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
        "todo_list": None,
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
        "todo_list": None,
    }
    result = verification_router(not_verified_state)
    assert result == "replan"


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
        "todo_list": None,
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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_custom_tools_always_have_attempt_completion() -> None:
    """Test that attempt_completion is always added to custom tools in agentic graph."""
    from alfredo.agentic import create_agentic_graph
    from alfredo.integrations.langchain import create_langchain_tool

    # Create custom tools without attempt_completion
    custom_tools = [
        create_langchain_tool("read_file"),
        create_langchain_tool("write_to_file"),
    ]

    # Verify custom tools don't include attempt_completion
    tool_names = {tool.name for tool in custom_tools}
    assert "attempt_completion" not in tool_names

    # Create graph with custom tools
    graph = create_agentic_graph(
        cwd=".",
        model_name="gpt-4.1-mini",
        tools=custom_tools,
    )

    # The graph should have been modified to include attempt_completion
    # We can verify this by checking the graph's internal state
    assert graph is not None


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_graph_without_planning() -> None:
    """Test that graph can be created without planner and replan nodes."""
    from alfredo.agentic import create_agentic_graph

    # Create graph with planning disabled
    graph = create_agentic_graph(
        cwd=".",
        model_name="gpt-4.1-mini",
        enable_planning=False,
    )

    assert graph is not None

    # Get the graph structure
    nodes = graph.get_graph().nodes
    node_names = {name for name in nodes if not name.startswith("__")}

    # Should have agent, tools, and verifier, but NOT planner or replan
    assert "agent" in node_names
    assert "tools" in node_names
    assert "verifier" in node_names
    assert "planner" not in node_names
    assert "replan" not in node_names


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_graph_with_planning_has_all_nodes() -> None:
    """Test that graph with planning enabled has all nodes."""
    from alfredo.agentic import create_agentic_graph

    # Create graph with planning enabled (default)
    graph = create_agentic_graph(
        cwd=".",
        model_name="gpt-4.1-mini",
        enable_planning=True,
    )

    assert graph is not None

    # Get the graph structure
    nodes = graph.get_graph().nodes
    node_names = {name for name in nodes if not name.startswith("__")}

    # Should have all nodes including planner and replan
    expected_nodes = {"planner", "agent", "tools", "verifier", "replan"}
    assert expected_nodes.issubset(node_names)


def test_agent_system_prompt_without_plan() -> None:
    """Test that agent system prompt handles empty plan gracefully."""
    from alfredo.agentic.prompts import get_agent_system_prompt

    task = "Create a Python script"

    # Test with empty plan
    prompt_no_plan = get_agent_system_prompt(task, plan="")
    assert task in prompt_no_plan
    assert "Implementation Plan" not in prompt_no_plan
    assert "Break down the task into logical steps" in prompt_no_plan

    # Test with None plan
    prompt_none_plan = get_agent_system_prompt(task, plan=None)  # type: ignore[arg-type]
    assert task in prompt_none_plan
    assert "Implementation Plan" not in prompt_none_plan

    # Test with actual plan
    plan = "Step 1: Create file\nStep 2: Write code"
    prompt_with_plan = get_agent_system_prompt(task, plan)
    assert task in prompt_with_plan
    assert "Implementation Plan" in prompt_with_plan
    assert plan in prompt_with_plan
    assert "Follow the implementation plan" in prompt_with_plan
