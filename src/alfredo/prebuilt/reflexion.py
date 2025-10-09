"""Reflexion agent for research tasks with iterative self-critique and revision.

This module implements the Reflexion pattern from the LangGraph examples, where an agent:
1. Drafts an initial answer with self-reflection
2. Identifies information gaps and generates search queries
3. Executes searches to gather additional information
4. Revises the answer based on new information with citations
5. Repeats until max iterations reached

Based on: https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/
"""

import datetime
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, cast

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ============================================================================
# Pydantic Schemas for Structured Outputs
# ============================================================================


class Reflection(BaseModel):
    """Self-critique of an answer identifying what's missing or superfluous."""

    missing: str = Field(description="Critique of what is missing in the answer.")
    superfluous: str = Field(description="Critique of what is superfluous in the answer.")


class AnswerQuestion(BaseModel):
    """Initial answer to a question with self-reflection and search queries."""

    answer: str = Field(description="Detailed ~250 word answer to the question.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique."
    )
    reflection: Reflection = Field(description="Self-reflection on the initial answer.")


class ReviseAnswer(AnswerQuestion):
    """Revised answer incorporating new information with citations."""

    references: list[str] = Field(
        description="Citations motivating your updated answer in the format [1] URL, [2] URL, etc."
    )


# ============================================================================
# State Definition
# ============================================================================


class ReflexionState(TypedDict):
    """State for the reflexion agent.

    Attributes:
        messages: Conversation history between agent and tools
        question: Original research question provided by the user
        iteration: Current iteration count (incremented after each revision)
        max_iterations: Maximum number of revision iterations allowed
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    iteration: int
    max_iterations: int


# ============================================================================
# Node Creation Functions
# ============================================================================


def create_draft_node(model: Any) -> Any:
    """Create the initial draft node that generates an answer with self-reflection.

    Args:
        model: Language model to use

    Returns:
        Draft node function
    """
    # System prompt for initial drafting
    actor_prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert researcher.
Current time: {time}

1. Provide a detailed ~250 word answer to the question.
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, list 1-3 search queries separately for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]).partial(
        time=lambda: datetime.datetime.now().isoformat(),
    )

    # Bind model with AnswerQuestion tool
    chain = actor_prompt_template | model.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

    def draft_node(state: ReflexionState, config: RunnableConfig) -> dict[str, Any]:
        """Generate initial draft with self-reflection."""
        response = chain.invoke({"messages": state["messages"]}, config)
        return {"messages": [response]}

    return draft_node


def create_execute_tools_node(search_tool: Any) -> Any:
    """Create the tool execution node that runs searches from the agent's queries.

    Args:
        search_tool: Tool to use for executing searches (e.g., TavilySearch)

    Returns:
        Tool execution node function
    """

    def execute_tools_node(state: ReflexionState, config: RunnableConfig) -> dict[str, Any]:
        """Execute search queries from the last AI message."""
        messages = state["messages"]
        last_ai_message = messages[-1]

        if not isinstance(last_ai_message, AIMessage) or not hasattr(last_ai_message, "tool_calls"):
            return {"messages": []}

        tool_calls = last_ai_message.tool_calls
        if not tool_calls:
            return {"messages": []}

        # Process tool calls to extract search queries
        tool_messages = []

        for tool_call in tool_calls:
            if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
                call_id = tool_call["id"]
                search_queries = tool_call["args"].get("search_queries", [])

                # Execute each search query
                query_results = {}
                for query in search_queries:
                    try:
                        result = search_tool.invoke(query)
                        query_results[query] = result
                    except Exception as e:
                        query_results[query] = f"Error: {e}"

                # Create tool message with results
                tool_messages.append(ToolMessage(content=json.dumps(query_results), tool_call_id=call_id))

        # Increment iteration count
        new_iteration = state["iteration"] + 1

        return {"messages": tool_messages, "iteration": new_iteration}

    return execute_tools_node


def create_revisor_node(model: Any) -> Any:
    """Create the revisor node that improves the answer based on search results.

    Args:
        model: Language model to use

    Returns:
        Revisor node function
    """
    # System prompt for revision
    revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

    actor_prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, list 1-3 search queries separately for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Revise your answer above using the required format."),
    ]).partial(
        time=lambda: datetime.datetime.now().isoformat(),
        first_instruction=revise_instructions,
    )

    # Bind model with ReviseAnswer tool
    chain = actor_prompt_template | model.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

    def revisor_node(state: ReflexionState, config: RunnableConfig) -> dict[str, Any]:
        """Revise answer using search results."""
        response = chain.invoke({"messages": state["messages"]}, config)
        return {"messages": [response]}

    return revisor_node


# ============================================================================
# Routing Function
# ============================================================================


def should_continue(state: ReflexionState) -> Literal["execute_tools", "__end__"]:
    """Determine whether to continue the reflexion loop or end.

    Args:
        state: Current reflexion state

    Returns:
        Next node to visit ("execute_tools" to continue, "__end__" to stop)
    """
    if state["iteration"] >= state["max_iterations"]:
        return "__end__"
    return "execute_tools"


# ============================================================================
# Graph Builder
# ============================================================================


def create_reflexion_graph(
    model_name: str = "gpt-4.1-mini",
    max_iterations: int = 2,
    search_tool: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Create a compiled reflexion graph.

    Args:
        model_name: Name of the language model to use
        max_iterations: Maximum number of revision iterations
        search_tool: Tool for executing searches. If None, uses TavilySearch
        **kwargs: Additional arguments to pass to model initialization

    Returns:
        Compiled LangGraph StateGraph
    """
    # Initialize model
    model = init_chat_model(model_name, **kwargs)

    # Default to Tavily if no search tool provided
    if search_tool is None:
        try:
            from langchain_tavily import TavilySearch

            search_tool = TavilySearch(max_results=5)
        except ImportError as e:
            msg = (
                "TavilySearch not available. Install with: uv add langchain-tavily\n"
                "Or provide a custom search_tool parameter."
            )
            raise ImportError(msg) from e

    # Create nodes
    draft_node = create_draft_node(model)
    execute_tools_node = create_execute_tools_node(search_tool)
    revisor_node = create_revisor_node(model)

    # Build graph
    graph = StateGraph(ReflexionState)

    # Add nodes
    graph.add_node("draft", draft_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("revisor", revisor_node)

    # Add edges
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "execute_tools")
    graph.add_edge("execute_tools", "revisor")

    # Conditional edge from revisor
    graph.add_conditional_edges(
        "revisor",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "__end__": END,
        },
    )

    return graph.compile()


# ============================================================================
# ReflexionAgent Class
# ============================================================================


class ReflexionAgent:
    """Agent for research tasks using the Reflexion pattern.

    This agent implements iterative self-critique and revision:
    1. Drafts an initial answer with self-reflection
    2. Generates search queries based on identified gaps
    3. Executes searches to gather information
    4. Revises answer with citations based on new information
    5. Repeats until max iterations reached

    Example:
        >>> agent = ReflexionAgent(model_name="gpt-4.1-mini", max_iterations=2)
        >>> answer = agent.research("How can small businesses leverage AI to grow?")
        >>> print(answer)
        >>> agent.display_trace()
    """

    def __init__(
        self,
        cwd: str = ".",
        model_name: str = "gpt-4.1-mini",
        max_iterations: int = 2,
        search_tool: Optional[Any] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the reflexion agent.

        Args:
            cwd: Working directory (used for output path resolution)
            model_name: Name of the language model to use (default: "gpt-4.1-mini")
            max_iterations: Maximum number of revision iterations (default: 2)
            search_tool: Optional custom search tool. If None, uses TavilySearch
            output_path: Path to save research results. If None, saves to notes/reflexion_research.md
            verbose: Whether to print progress updates during execution
            **kwargs: Additional keyword arguments to pass to the model
                (e.g., temperature, base_url, api_key)
        """
        self.cwd = Path(cwd).resolve()
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.model_kwargs = kwargs

        # Determine output path
        if output_path is None:
            notes_dir = self.cwd / "notes"
            notes_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = notes_dir / "reflexion_research.md"
        else:
            self.output_path = Path(output_path)

        # Create the reflexion graph
        self.graph = create_reflexion_graph(
            model_name=model_name,
            max_iterations=max_iterations,
            search_tool=search_tool,
            **kwargs,
        )

        # Storage for execution results
        self._results: Optional[dict[str, Any]] = None

    @property
    def results(self) -> Optional[dict[str, Any]]:
        """Get the results from the last research run.

        Returns:
            Final state dictionary with keys: messages, question, iteration, max_iterations.
            Returns None if research() hasn't been called yet.
        """
        return self._results

    def research(self, question: str) -> str:
        """Run the reflexion research loop on a question.

        Args:
            question: Research question to answer

        Returns:
            Final answer with citations as a string

        Raises:
            RuntimeError: If research fails to produce an answer
        """
        # Initial state
        initial_state: ReflexionState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }

        if self.verbose:
            print(f"🔍 Starting reflexion research: {question}\n")
            print(f"📊 Max iterations: {self.max_iterations}\n")

        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Error during research: {e}")
            raise
        else:
            # Extract the final answer from the last message
            messages = final_state["messages"]
            final_answer = self._extract_answer_from_messages(messages)

            if final_answer is None:
                msg = "Research failed: No answer produced"
                raise RuntimeError(msg)

            if self.verbose:
                print("\n✅ Research completed!")
                print(f"📝 Iterations used: {final_state['iteration']}/{self.max_iterations}")
                print(f"\n📄 Final Answer:\n{final_answer[:500]}...\n")

            # Save to file if output path is set
            if self.output_path:
                self._save_research_report(question, final_answer, final_state)

            # Store results
            self._results = final_state
            return final_answer

    def _extract_answer_from_messages(self, messages: Sequence[BaseMessage]) -> Optional[str]:
        """Extract the final answer from the message history.

        Args:
            messages: Message history from the graph execution

        Returns:
            The final answer string, or None if not found
        """
        # Find the last AI message with a ReviseAnswer or AnswerQuestion tool call
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                tool_calls = msg.tool_calls
                for tool_call in reversed(tool_calls):
                    if tool_call["name"] in ["ReviseAnswer", "AnswerQuestion"]:
                        args = tool_call["args"]
                        answer = cast(str, args.get("answer", ""))
                        references = cast(list[str], args.get("references", []))

                        # Format answer with references if available
                        if references:
                            answer += "\n\n## References\n"
                            for ref in references:
                                answer += f"- {ref}\n"

                        return answer

        return None

    def _save_research_report(self, question: str, answer: str, state: dict[str, Any]) -> None:
        """Save the research report to a markdown file.

        Args:
            question: The research question
            answer: The final answer
            state: The final state containing execution details
        """
        report = f"""# Reflexion Research Report

**Question:** {question}

**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Model:** {self.model_name}

**Iterations:** {state["iteration"]}/{self.max_iterations}

---

## Answer

{answer}

---

*Generated by ReflexionAgent*
"""

        with open(self.output_path, "w") as f:
            f.write(report)

        if self.verbose:
            print(f"💾 Research report saved to: {self.output_path}")

    def display_trace(self) -> None:
        """Display a formatted trace of the research process.

        Shows:
        - Initial question
        - Draft answer with reflection
        - Search queries executed
        - Revisions made
        - Final answer

        Raises:
            RuntimeError: If research() hasn't been called yet
        """
        if self._results is None:
            msg = "No execution results available. Call research() first."
            raise RuntimeError(msg)

        messages = cast(Sequence[BaseMessage], self._results.get("messages", []))
        question = str(self._results.get("question", "N/A"))
        iterations = int(self._results.get("iteration", 0))

        self._print_trace_header(question, iterations, len(messages))

        current_iteration = 0
        for i, message in enumerate(messages):
            print(f"\n--- Message {i + 1} ({type(message).__name__}) ---")
            current_iteration = self._display_message(message, current_iteration)

        print("\n" + "=" * 80)

    def _print_trace_header(self, question: str, iterations: int, message_count: int) -> None:
        """Print the trace header with summary information."""
        print("\n" + "=" * 80)
        print("REFLEXION TRACE")
        print("=" * 80)
        print(f"\nQuestion: {question}")
        print(f"Iterations: {iterations}/{self.max_iterations}")
        print(f"Total Messages: {message_count}")
        print("\n" + "=" * 80)
        print("EXECUTION STEPS")
        print("=" * 80)

    def _display_message(self, msg: BaseMessage, current_iteration: int) -> int:
        """Display a single message in the trace.

        Args:
            msg: Message to display
            current_iteration: Current iteration counter

        Returns:
            Updated iteration counter
        """
        if isinstance(msg, HumanMessage):
            print(f"Question: {msg.content}")
        elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            self._display_ai_message(msg, current_iteration)
        elif isinstance(msg, ToolMessage):
            self._display_tool_message(msg)
            return current_iteration + 1

        return current_iteration

    def _display_ai_message(self, msg: AIMessage, current_iteration: int) -> None:
        """Display an AI message with tool calls."""
        tool_calls = msg.tool_calls
        if not tool_calls:
            return

        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})

            if tool_name == "AnswerQuestion":
                print(f"\n🎯 DRAFT (Iteration {current_iteration})")
            elif tool_name == "ReviseAnswer":
                print(f"\n✏️  REVISION (Iteration {current_iteration})")

            self._display_tool_call_args(args)

    def _display_tool_call_args(self, args: dict[str, Any]) -> None:
        """Display the arguments from a tool call."""
        answer = args.get("answer", "")
        reflection = args.get("reflection", {})
        search_queries = args.get("search_queries", [])
        references = args.get("references", [])

        print(f"\nAnswer ({len(answer)} chars):")
        print(f"{answer[:200]}..." if len(answer) > 200 else answer)

        if reflection:
            print("\n📋 Reflection:")
            print(f"  Missing: {reflection.get('missing', 'N/A')[:100]}...")
            print(f"  Superfluous: {reflection.get('superfluous', 'N/A')[:100]}...")

        if search_queries:
            print(f"\n🔎 Search Queries ({len(search_queries)}):")
            for query in search_queries:
                print(f"  - {query}")

        if references:
            print(f"\n📚 References ({len(references)}):")
            for ref in references[:3]:  # Show first 3
                print(f"  - {ref}")

    def _display_tool_message(self, msg: ToolMessage) -> None:
        """Display a tool message with search results."""
        print("\n🔍 SEARCH RESULTS")
        try:
            content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if isinstance(content, dict):
                print(f"  Queries executed: {len(content)}")
                for query, results in list(content.items())[:2]:  # Show first 2
                    print(f"  - {query}")
                    if isinstance(results, list):
                        print(f"    Results: {len(results)} items")
        except Exception:
            print(f"  Content: {str(msg.content)[:100]}...")
