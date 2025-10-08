"""State definitions for the agentic scaffold."""
# mypy: disable-error-code="no-any-unimported"

from collections.abc import Sequence
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


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
        todo_list: Optional todo list for tracking task progress (numbered checklist)
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    plan: str
    plan_iteration: int
    max_context_tokens: int
    final_answer: Optional[str]
    is_verified: bool
    todo_list: Optional[str]
