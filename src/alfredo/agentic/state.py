"""State definitions for the agentic scaffold."""

from collections.abc import Sequence
from typing import Annotated, Optional

try:
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    BaseMessage = object  # type: ignore
    TypedDict = dict  # type: ignore


def check_langgraph_available() -> None:
    """Check if LangGraph is available and raise error if not."""
    if not LANGGRAPH_AVAILABLE:
        msg = "LangGraph is not installed. Install it with: uv add alfredo[agentic]"
        raise ImportError(msg)


if LANGGRAPH_AVAILABLE:

    class AgentState(TypedDict):
        """State for the agentic scaffold.

        Attributes:
            messages: Conversation history between agent and tools
            task: Original task provided by the user
            plan: Current implementation plan
            plan_iteration: Number of times the plan has been created/updated
            max_context_tokens: Maximum number of tokens allowed in context
            final_answer: Answer provided by attempt_answer tool (if any)
            is_verified: Whether the final answer has been verified
        """

        messages: Annotated[Sequence[BaseMessage], add_messages]
        task: str
        plan: str
        plan_iteration: int
        max_context_tokens: int
        final_answer: Optional[str]
        is_verified: bool

else:
    # Placeholder for when LangGraph is not available
    AgentState = dict  # type: ignore
