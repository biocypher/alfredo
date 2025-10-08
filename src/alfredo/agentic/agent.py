"""Class-based interface for the agentic scaffold."""
# mypy: disable-error-code="no-any-unimported"

from typing import Any, Optional

from alfredo.agentic.graph import create_agentic_graph
from alfredo.agentic.state import AgentState


class Agent:
    """Agentic scaffold with LangGraph-based plan-verify-replan loop.

    This is the recommended way to use Alfredo for autonomous task execution.
    The agent automatically plans, executes tools, and verifies completion.

    Example:
        >>> from alfredo import Agent
        >>> agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=True)
        >>> agent.run("Create a hello world Python script")
        >>> agent.display_trace()
        >>> print(agent.results["final_answer"])
    """

    def __init__(
        self,
        cwd: str = ".",
        model_name: str = "gpt-4.1-mini",
        max_context_tokens: int = 100000,
        tools: Optional[list] = None,
        verbose: bool = True,
        recursion_limit: int = 50,
        **kwargs: Any,
    ) -> None:
        """Initialize the agentic agent.

        Args:
            cwd: Working directory for file operations (default: ".")
            model_name: Name of the model to use (default: "gpt-4.1-mini")
            max_context_tokens: Maximum context window size in tokens
            tools: Optional list of LangChain tools. If None, uses all Alfredo tools.
                Can include MCP tools loaded via alfredo.integrations.mcp
            verbose: Whether to print progress updates during execution
            recursion_limit: Maximum number of graph steps (default: 50)
            **kwargs: Additional keyword arguments to pass to the model
                (e.g., temperature, base_url, api_key)
        """
        self.cwd = cwd
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.tools = tools
        self.verbose = verbose
        self.recursion_limit = recursion_limit
        self.model_kwargs = kwargs

        # Create the LangGraph
        self.graph = create_agentic_graph(
            cwd=cwd,
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            tools=tools,
            recursion_limit=recursion_limit,
            **kwargs,
        )

        # Storage for execution results
        self._results: Optional[dict[str, Any]] = None

    @property
    def results(self) -> Optional[dict[str, Any]]:
        """Get the results from the last run.

        Returns:
            Final state dictionary with keys: messages, task, plan, plan_iteration,
            final_answer, is_verified. Returns None if run() hasn't been called yet.
        """
        return self._results

    def run(self, task: str) -> dict[str, Any]:
        """Run an agentic task from start to finish.

        Args:
            task: The task to accomplish

        Returns:
            Final state dictionary with results

        Raises:
            Exception: If execution fails
        """
        # Initial state
        initial_state: AgentState = {
            "messages": [],
            "task": task,
            "plan": "",
            "plan_iteration": 0,
            "max_context_tokens": self.max_context_tokens,
            "final_answer": None,
            "is_verified": False,
        }

        if self.verbose:
            print(f"🚀 Starting agentic task: {task}\n")

        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state, config={"recursion_limit": self.recursion_limit})
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Error during execution: {e}")
            raise
        else:
            if self.verbose:
                print("\n✅ Task completed!")
                print(f"\n📝 Final Answer:\n{final_state.get('final_answer', 'No answer provided')}")

            # Store results
            self._results = final_state
            return final_state  # type: ignore[no-any-return]

    def display_trace(self) -> None:  # noqa: C901
        """Display a formatted trace of all messages and tool calls.

        This provides a detailed view of the agent's execution, including:
        - All messages exchanged (HumanMessage, AIMessage, ToolMessage, etc.)
        - Tool calls made by the agent with their arguments
        - Tool responses

        Raises:
            RuntimeError: If run() hasn't been called yet
        """
        if self._results is None:
            msg = "No execution results available. Call run() first."
            raise RuntimeError(msg)

        messages = self._results.get("messages", [])

        print("\n" + "=" * 80)
        print("EXECUTION TRACE")
        print("=" * 80)

        # Print summary stats
        print(f"\nTask: {self._results.get('task', 'N/A')}")
        print(f"Plan Iterations: {self._results.get('plan_iteration', 0)}")
        print(f"Total Messages: {len(messages)}")
        print(f"Verified: {self._results.get('is_verified', False)}")
        print(f"Final Answer: {self._results.get('final_answer', 'N/A')[:100]}...")

        print("\n" + "=" * 80)
        print("MESSAGES")
        print("=" * 80)

        # Print all messages
        for i, msg in enumerate(messages):
            print(f"\n--- Message {i + 1} ---")
            print(f"Type: {type(msg).__name__}")

            # Print message content
            if hasattr(msg, "content"):
                content = str(msg.content)
                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "... (truncated)"
                print(f"Content: {content}")

            # Print tool calls if present (AIMessage)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"Tool Calls ({len(msg.tool_calls)}):")
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    # Detect MCP tools (usually prefixed)
                    prefix = "🔬 [MCP]" if tool_name.startswith("bc_") else "🛠️  [Alfredo]"
                    print(f"  {prefix} {tool_name}")
                    if "args" in tc:
                        args_str = str(tc["args"])
                        if len(args_str) > 200:
                            args_str = args_str[:200] + "... (truncated)"
                        print(f"       Args: {args_str}")

            # Print tool call ID if present (ToolMessage)
            if hasattr(msg, "tool_call_id"):
                print(f"Tool Call ID: {msg.tool_call_id}")

            # Print name if present (ToolMessage)
            if hasattr(msg, "name"):
                print(f"Tool Name: {msg.name}")

        print("\n" + "=" * 80)
