"""LangGraph state graph for the agentic scaffold."""
# mypy: disable-error-code="no-any-unimported"

from typing import Any, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from alfredo.agentic.nodes import (
    create_agent_node,
    create_planner_node,
    create_replan_node,
    create_tools_node,
    create_verifier_node,
)
from alfredo.agentic.state import AgentState
from alfredo.integrations.langchain import create_all_langchain_tools

# Note: We use the existing attempt_completion tool from alfredo.tools.handlers.workflow
# No need to create a duplicate tool here


def should_continue(state: AgentState) -> Literal["tools", "agent"]:
    """Determine the next node based on the agent's output.

    Args:
        state: Current agent state

    Returns:
        Name of the next node to visit
    """
    messages = state["messages"]
    if not messages:
        return "agent"

    last_message = messages[-1]

    # Check if it's an AI message with tool calls
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
        if tool_calls:
            # ALL tool calls (including attempt_answer) go to tools node
            return "tools"

    # Default: continue agent loop
    return "agent"


def route_after_tools(state: AgentState) -> Literal["agent", "verifier"]:
    """Route after tools execution - check if attempt_completion was called.

    Args:
        state: Current agent state

    Returns:
        Name of the next node to visit
    """
    messages = state["messages"]
    if not messages:
        return "agent"

    # Check the last message - it should be a ToolMessage from the tools node
    last_message = messages[-1]

    # Check if it's a ToolMessage with attempt_completion content marker
    if hasattr(last_message, "content"):
        content = str(last_message.content)
        if "[TASK_COMPLETE]" in content:
            # Found attempt_completion tool response, go to verifier
            return "verifier"

    # Also check by tool name if available
    if hasattr(last_message, "name") and last_message.name == "attempt_completion":
        return "verifier"

    # No attempt_completion found, continue with agent
    return "agent"


def verification_router(state: AgentState) -> Literal["__end__", "replan"]:
    """Route based on verification result.

    Args:
        state: Current agent state

    Returns:
        Next node name or END
    """
    is_verified = state.get("is_verified", False)

    if is_verified:
        return "__end__"
    else:
        return "replan"


def create_agentic_graph(
    cwd: str = ".",
    model_name: str = "gpt-4.1-mini",
    max_context_tokens: int = 100000,
    tools: Optional[list] = None,
    recursion_limit: int = 50,
    **kwargs: Any,
) -> Any:
    """Create the agentic scaffold state graph.

    Args:
        cwd: Working directory for file operations
        model_name: Name of the model to use (default: gpt-4.1-mini)
        max_context_tokens: Maximum context window size in tokens
        tools: Optional list of tools. If None, uses all Alfredo tools.
        recursion_limit: Maximum number of graph steps before raising an error (default: 50)
        **kwargs: Additional keyword arguments to pass to the model

    Returns:
        Compiled LangGraph state graph
    """
    # Initialize model
    model = init_chat_model(model_name, **kwargs)
    # model = ChatOpenAI(
    #    base_url="https://api.z.ai/api/coding/paas/v4",
    #    api_key="b88128bec8274c7a9b2a2eec6ca4e9d1.cfzUIJLksdgoqyxy",  # type: ignore[arg-type]
    #    model="glm-4.6",
    # )

    # Get tools (includes attempt_completion from workflow handlers)
    if tools is None:
        # Get all Alfredo tools (includes attempt_completion)
        tools = create_all_langchain_tools(cwd=cwd)

    # Bind tools to model for agent node
    model_with_tools = model.bind_tools(tools)

    # Create nodes
    planner_node = create_planner_node(model)
    agent_node = create_agent_node(model_with_tools)
    tools_node = create_tools_node(tools)
    verifier_node = create_verifier_node(model)
    replan_node = create_replan_node(model)

    # Create state graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("replan", replan_node)

    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")

    # Conditional edge from agent (routes to tools or continues thinking)
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "agent": "agent",
        },
    )

    # Conditional edge from tools (routes to verifier if attempt_answer, otherwise back to agent)
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "agent": "agent",
            "verifier": "verifier",
        },
    )

    # Conditional edge from verifier
    graph.add_conditional_edges(
        "verifier",
        verification_router,
        {
            "__end__": END,
            "replan": "replan",
        },
    )

    # Edge from replan back to agent
    graph.add_edge("replan", "agent")

    # Compile and return
    return graph.compile()


def run_agentic_task(
    task: str,
    cwd: str = ".",
    model_name: str = "gpt-4.1-mini",
    max_context_tokens: int = 100000,
    verbose: bool = True,
    tools: Optional[list] = None,
    recursion_limit: int = 50,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run an agentic task from start to finish.

    This is a convenience function that creates an Agent and runs a task.
    For multiple tasks, consider creating an Agent instance and calling run() directly.

    Args:
        task: The task to accomplish
        cwd: Working directory for file operations
        model_name: Name of the model to use
        max_context_tokens: Maximum context window size
        verbose: Whether to print progress updates
        tools: Optional list of LangChain tools. If None, uses all Alfredo tools.
            Can include MCP tools loaded via alfredo.integrations.mcp
        recursion_limit: Maximum number of graph steps (default: 50)
        **kwargs: Additional keyword arguments to pass to the model

    Returns:
        Final state dictionary with results

    Example:
        >>> result = run_agentic_task("Create a hello world script", verbose=True)
        >>> print(result["final_answer"])
    """
    # Import here to avoid circular import
    from alfredo.agentic.agent import Agent

    # Create agent and run task
    agent = Agent(
        cwd=cwd,
        model_name=model_name,
        max_context_tokens=max_context_tokens,
        tools=tools,
        verbose=verbose,
        recursion_limit=recursion_limit,
        **kwargs,
    )

    return agent.run(task)
