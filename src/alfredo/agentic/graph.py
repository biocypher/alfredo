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
from alfredo.agentic.prompt_templates import PromptTemplates
from alfredo.agentic.state import AgentState
from alfredo.integrations.langchain import create_langchain_tools
from alfredo.tools.alfredo_tool import AlfredoTool

# Note: We use the existing attempt_completion tool from alfredo.tools.handlers.workflow
# No need to create a duplicate tool here


def _normalize_tools(tools: list[Any]) -> list[AlfredoTool]:
    """Normalize a list of tools to AlfredoTools.

    Wraps plain LangChain StructuredTools as AlfredoTools without instructions.
    AlfredoTools are passed through unchanged.

    Args:
        tools: List of tools (can be AlfredoTool or StructuredTool)

    Returns:
        List of AlfredoTool instances
    """
    normalized = []
    for tool in tools:
        if isinstance(tool, AlfredoTool):
            normalized.append(tool)
        else:
            # Wrap plain LangChain tool as AlfredoTool without instructions
            normalized.append(AlfredoTool.from_langchain(tool))
    return normalized


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
    enable_planning: bool = True,
    prompt_templates: Optional[PromptTemplates] = None,
    parse_reasoning: bool = False,
    **kwargs: Any,
) -> Any:
    """Create the agentic scaffold state graph.

    Args:
        cwd: Working directory for file operations
        model_name: Name of the model to use (default: gpt-4.1-mini)
        max_context_tokens: Maximum context window size in tokens
        tools: Optional list of tools. If None, uses all Alfredo tools.
        recursion_limit: Maximum number of graph steps before raising an error (default: 50)
        enable_planning: Whether to use the planner node. If False, starts directly at agent node (default: True)
        prompt_templates: Optional custom prompt templates for each node
        parse_reasoning: Whether to parse <think> tags from model responses (default: False)
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
        tools = create_langchain_tools(cwd=cwd)
    else:
        # Ensure attempt_completion is ALWAYS present (required to exit react loop)
        tool_names = {getattr(t, "name", "") for t in tools}
        if "attempt_completion" not in tool_names:
            # Add attempt_completion tool
            from alfredo.integrations.langchain import create_langchain_tool

            attempt_completion_tool = create_langchain_tool("attempt_completion", cwd=cwd)
            tools = [*list(tools), attempt_completion_tool]

    # Normalize tools (wrap plain StructuredTools as AlfredoTools)
    normalized_tools = _normalize_tools(tools)

    # Extract LangChain tools for model binding
    langchain_tools = [t.to_langchain_tool() for t in normalized_tools]

    # Bind tools to model for agent node
    model_with_tools = model.bind_tools(langchain_tools, tool_choice="auto")

    # Extract templates from prompt_templates if provided
    planner_template = prompt_templates.planner if prompt_templates else None
    agent_template = prompt_templates.agent if prompt_templates else None
    verifier_template = prompt_templates.verifier if prompt_templates else None
    replan_template = prompt_templates.replan if prompt_templates else None

    # Create nodes (pass normalized tools, templates, and parse_reasoning flag)
    agent_node = create_agent_node(
        model_with_tools, tools=normalized_tools, template=agent_template, parse_reasoning=parse_reasoning
    )
    tools_node = create_tools_node(langchain_tools)
    verifier_node = create_verifier_node(
        model, tools=normalized_tools, template=verifier_template, parse_reasoning=parse_reasoning
    )

    # Conditionally create planner and replan nodes
    if enable_planning:
        planner_node = create_planner_node(
            model, tools=normalized_tools, template=planner_template, parse_reasoning=parse_reasoning
        )
        replan_node = create_replan_node(
            model, tools=normalized_tools, template=replan_template, parse_reasoning=parse_reasoning
        )

    # Create state graph
    graph = StateGraph(AgentState)

    # Add nodes - planner and replan are optional
    if enable_planning:
        graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("verifier", verifier_node)
    if enable_planning:
        graph.add_node("replan", replan_node)

    # Add edges - conditional based on enable_planning
    if enable_planning:
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "agent")
    else:
        graph.add_edge(START, "agent")

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

    # Conditional edge from verifier - behavior depends on enable_planning
    if enable_planning:
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
    else:
        # Without planning, verification failure goes to END (no retry)
        graph.add_edge("verifier", END)

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
    enable_planning: bool = True,
    parse_reasoning: bool = False,
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
        enable_planning: Whether to use the planner node. If False, starts directly at agent (default: True)
        parse_reasoning: Whether to parse <think> tags from model responses (default: False)
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
        enable_planning=enable_planning,
        parse_reasoning=parse_reasoning,
        **kwargs,
    )

    return agent.run(task)
