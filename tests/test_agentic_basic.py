"""Basic tests for the agentic scaffold."""


def test_imports() -> None:
    """Test that agentic modules can be imported."""
    from alfredo.agentic import AgentState, create_agentic_graph
    from alfredo.agentic.context_manager import ContextManager
    from alfredo.agentic.nodes import create_agent_node, create_planner_node
    from alfredo.agentic.prompts import get_agent_system_prompt, get_planning_prompt

    assert AgentState is not None
    assert create_agentic_graph is not None
    assert ContextManager is not None
    assert create_agent_node is not None
    assert create_planner_node is not None
    assert get_agent_system_prompt is not None
    assert get_planning_prompt is not None


def test_agent_state() -> None:
    """Test AgentState structure."""
    from alfredo.agentic.state import AgentState

    state: AgentState = {
        "messages": [],
        "task": "Test task",
        "plan": "Test plan",
        "plan_iteration": 0,
        "max_context_tokens": 100000,
        "final_answer": None,
        "is_verified": False,
        "todo_list": None,
    }

    assert state["task"] == "Test task"
    assert state["plan"] == "Test plan"
    assert state["plan_iteration"] == 0
    assert state["is_verified"] is False


def test_context_manager() -> None:
    """Test context manager token estimation."""
    from langchain_core.messages import HumanMessage

    from alfredo.agentic.context_manager import ContextManager

    cm = ContextManager(max_tokens=1000)

    # Test token estimation
    text = "Hello world"
    tokens = cm.estimate_tokens(text)
    assert tokens > 0

    # Test should_summarize
    messages = [HumanMessage(content="Short message")]
    assert not cm.should_summarize(messages)


def test_prompts() -> None:
    """Test prompt generation."""
    from alfredo.agentic.prompts import (
        get_agent_system_prompt,
        get_planning_prompt,
        get_verification_prompt,
    )

    task = "Create a Python script"
    plan = "Step 1: Create file\nStep 2: Write code"

    planning_prompt = get_planning_prompt(task)
    assert "Implementation Plan" in planning_prompt
    assert task in planning_prompt

    agent_prompt = get_agent_system_prompt(task, plan)
    assert task in agent_prompt
    assert plan in agent_prompt

    verification_prompt = get_verification_prompt(task, "I created the script")
    assert task in verification_prompt
    assert "VERIFIED" in verification_prompt


def test_attempt_completion_tool() -> None:
    """Test that attempt_completion tool is available from workflow handlers."""
    from alfredo.integrations.langchain import create_langchain_tools

    # Get all tools (includes attempt_completion)
    tools = create_langchain_tools(cwd=".")

    # Find the attempt_completion tool
    attempt_completion_tool = None
    for tool in tools:
        if tool.name == "attempt_completion":
            attempt_completion_tool = tool
            break

    assert attempt_completion_tool is not None, "attempt_completion tool should be in tools"
    assert "result" in str(attempt_completion_tool.args_schema.model_json_schema())  # type: ignore[union-attr]

    # Test invocation
    result = attempt_completion_tool.invoke({"result": "Task completed successfully"})
    assert "[TASK_COMPLETE]" in result
    assert "Task completed successfully" in result


def test_langgraph_imports() -> None:
    """Test that LangGraph imports work (required dependency)."""
    from langchain_core.messages import BaseMessage
    from langgraph.graph import StateGraph

    assert BaseMessage is not None
    assert StateGraph is not None
