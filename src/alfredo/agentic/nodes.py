"""Graph nodes for the agentic scaffold."""
# mypy: disable-error-code="no-any-unimported"

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from alfredo.agentic.prompts import (
    get_agent_system_prompt,
    get_planning_prompt,
    get_replan_prompt,
    get_verification_prompt,
)
from alfredo.agentic.state import AgentState


def create_planner_node(model: BaseChatModel, has_todo_tools: bool = False) -> Any:
    """Create the planner node that generates initial implementation plans.

    Args:
        model: The language model to use for planning
        has_todo_tools: Whether todo list tools are available

    Returns:
        Planner node function
    """

    def planner_node(state: AgentState) -> dict[str, Any]:
        """Generate an implementation plan for the task.

        Args:
            state: Current agent state

        Returns:
            Updated state with plan
        """
        task = state["task"]
        plan_iteration = state.get("plan_iteration", 0)

        # Get planning prompt
        planning_prompt = get_planning_prompt(task, has_todo_tools=has_todo_tools)

        # Generate plan
        messages = [HumanMessage(content=planning_prompt)]
        response = model.invoke(messages)

        plan = response.content if hasattr(response, "content") else str(response)

        return {
            "plan": plan,
            "plan_iteration": plan_iteration + 1,
            "messages": [HumanMessage(content=f"Task: {task}"), AIMessage(content=f"Plan created:\n\n{plan}")],
        }

    return planner_node


def create_agent_node(model: BaseChatModel, has_todo_tools: bool = False) -> Any:
    """Create the agent node that performs reasoning and tool calling.

    Args:
        model: The language model with tools bound
        has_todo_tools: Whether todo list tools are available

    Returns:
        Agent node function
    """

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Perform reasoning and decide on next action.

        Args:
            state: Current agent state

        Returns:
            Updated state with new message
        """
        task = state["task"]
        plan = state["plan"]
        messages = list(state["messages"])

        # Create system message with task and plan context
        system_msg = SystemMessage(content=get_agent_system_prompt(task, plan, has_todo_tools=has_todo_tools))

        # Invoke model with full context
        full_messages = [system_msg, *messages]
        response = model.invoke(full_messages)

        # Return updated messages
        return {"messages": [response]}

    return agent_node


def create_tools_node(tools: Sequence[Any]) -> Any:
    """Create the tools node that executes tool calls.

    This wraps LangGraph's ToolNode and adds special handling for todo list tools
    to sync the todo_list state between AgentState and TodoStateManager.

    Args:
        tools: Sequence of available tools (BaseTool or compatible)

    Returns:
        Tools node function
    """
    # Check if todo tools are present
    tool_names = {getattr(t, "name", "") for t in tools}
    has_todo_tools = "write_todo_list" in tool_names or "read_todo_list" in tool_names

    # If no todo tools, just return standard ToolNode
    if not has_todo_tools:
        return ToolNode(tools)

    # Create LangGraph's standard ToolNode
    base_tools_node = ToolNode(tools)

    def tools_node_wrapper(state: AgentState) -> dict[str, Any]:
        """Execute tools and sync todo list state.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results and synced todo_list
        """
        # Sync state to TodoStateManager before execution
        try:
            from alfredo.tools.handlers.todo import TodoStateManager

            manager = TodoStateManager()
            current_todo = state.get("todo_list")
            manager.set_todo_list(current_todo)
        except ImportError:
            # Todo tools not available, skip sync
            pass

        # Execute tools using LangGraph's ToolNode
        # ToolNode is callable and returns state updates
        result = base_tools_node.invoke(state)

        # Sync TodoStateManager back to state after execution
        try:
            from alfredo.tools.handlers.todo import TodoStateManager

            manager = TodoStateManager()
            updated_todo = manager.get_todo_list()
            # Add todo_list to result if it's a dict
            if isinstance(result, dict):
                result["todo_list"] = updated_todo
            else:
                # If result is not a dict, create a dict with messages and todo_list
                result = {"messages": result.get("messages", []), "todo_list": updated_todo}
        except ImportError:
            # Todo tools not available, skip sync
            pass

        return result  # type: ignore[no-any-return]

    return tools_node_wrapper


def format_execution_trace(messages: Sequence) -> str:  # noqa: C901
    """Format message history into a readable execution trace.

    Args:
        messages: Sequence of messages from the conversation

    Returns:
        Formatted trace string showing actions taken (full outputs, no truncation)
    """
    if not messages:
        return "No actions recorded."

    trace_lines = []
    step_num = 0

    for msg in messages:
        # Skip system messages and initial plan messages
        if isinstance(msg, SystemMessage):
            continue

        # Agent reasoning
        if isinstance(msg, AIMessage):
            content = str(msg.content) if hasattr(msg, "content") else ""

            # Check if it has tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    step_num += 1
                    tool_name = tool_call.get("name", "unknown")
                    args = tool_call.get("args", {})

                    trace_lines.append(f"\n**Step {step_num}: Called tool `{tool_name}`**")
                    if args:
                        # Format args - show full values for verification
                        args_str = ", ".join([f"{k}={v!r}" for k, v in args.items()])
                        trace_lines.append(f"  Arguments: {args_str}")
            elif (
                content and not content.startswith("Plan created:") and not content.startswith("Creating improved plan")
            ):
                # Agent thinking/reasoning (not tool calls)
                step_num += 1
                trace_lines.append(f"\n**Step {step_num}: Agent reasoning**")
                # Include full reasoning for verification
                trace_lines.append(f"  {content}")

        # Tool results
        elif isinstance(msg, ToolMessage):
            content = str(msg.content) if hasattr(msg, "content") else ""
            # Show full results - verifier needs complete information
            if "[TASK_COMPLETE]" in content:
                trace_lines.append("  → Result: Task completion signal received")
            else:
                trace_lines.append(f"  → Result: {content}")

        # Human messages (usually verification feedback)
        elif isinstance(msg, HumanMessage):
            content = str(msg.content) if hasattr(msg, "content") else ""
            if not content.startswith("Task:") and not content.startswith("Verification result:"):
                step_num += 1
                trace_lines.append(f"\n**Step {step_num}: User input**")
                trace_lines.append(f"  {content}")

    if not trace_lines:
        return "No significant actions recorded."

    return "\n".join(trace_lines)


def create_verifier_node(model: BaseChatModel) -> Any:
    """Create the verifier node that checks if answers satisfy the task.

    Args:
        model: The language model to use for verification

    Returns:
        Verifier node function
    """

    def verifier_node(state: AgentState) -> dict[str, Any]:
        """Verify if the final answer addresses the original task.

        Args:
            state: Current agent state

        Returns:
            Updated state with verification status
        """
        task = state["task"]

        # Extract result from attempt_completion tool call
        final_answer = extract_attempt_completion(state)

        if not final_answer:
            return {"is_verified": False, "final_answer": None}

        # Format execution trace from message history
        execution_trace = format_execution_trace(state["messages"])

        # Get verification prompt with full execution trace
        verification_prompt = get_verification_prompt(task, final_answer, execution_trace)

        # Check if answer is satisfactory
        messages = [HumanMessage(content=verification_prompt)]
        response = model.invoke(messages)

        response_text = str(response.content) if hasattr(response, "content") else str(response)

        # Parse verification result
        is_verified = response_text.strip().startswith("VERIFIED:")

        # Add verification message to history
        verification_msg = HumanMessage(content=f"Verification result: {response_text}")

        return {"is_verified": is_verified, "final_answer": final_answer, "messages": [verification_msg]}

    return verifier_node


def create_replan_node(model: BaseChatModel) -> Any:
    """Create the replan node that generates new plans after verification failure.

    Args:
        model: The language model to use for replanning

    Returns:
        Replan node function
    """

    def replan_node(state: AgentState) -> dict[str, Any]:
        """Generate a new plan based on verification feedback.

        Args:
            state: Current agent state

        Returns:
            Updated state with new plan
        """
        task = state["task"]
        previous_plan = state["plan"]
        plan_iteration = state["plan_iteration"]

        # Extract verification feedback from last message
        messages = state["messages"]
        verification_feedback = "No specific feedback available."

        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                verification_feedback = str(last_msg.content)

        # Get replanning prompt
        replan_prompt = get_replan_prompt(task, previous_plan, verification_feedback)

        # Generate new plan
        planning_messages = [HumanMessage(content=replan_prompt)]
        response = model.invoke(planning_messages)

        new_plan = response.content if hasattr(response, "content") else str(response)

        # Add replan message
        replan_msg = AIMessage(content=f"Creating improved plan (iteration {plan_iteration + 1}):\n\n{new_plan}")

        return {
            "plan": new_plan,
            "plan_iteration": plan_iteration + 1,
            "messages": [replan_msg],
            "final_answer": None,  # Reset final answer
            "is_verified": False,  # Reset verification
        }

    return replan_node


def extract_attempt_completion(state: AgentState) -> str:
    """Extract the result from an attempt_completion tool call.

    The attempt_completion tool always returns content starting with [TASK_COMPLETE],
    so we simply search for any ToolMessage containing that marker.

    Args:
        state: Current agent state

    Returns:
        The extracted result or empty string
    """
    messages = state["messages"]
    if not messages:
        return ""

    # Search backwards for a ToolMessage with [TASK_COMPLETE] marker
    # The attempt_completion tool always outputs: "[TASK_COMPLETE]\n{result}"
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and hasattr(msg, "content"):
            content = str(msg.content)
            if "[TASK_COMPLETE]" in content:
                # Extract the result (everything after [TASK_COMPLETE]\n)
                lines = content.split("\n", 1)
                if len(lines) > 1:
                    # Return everything after the marker, stripping "Final command executed:" if present
                    result = lines[1]
                    # If there's a "Final command executed:" line, keep everything before it
                    if "\nFinal command executed:" in result:
                        result = result.split("\nFinal command executed:")[0]
                    return result.strip()
                return ""

    return ""
