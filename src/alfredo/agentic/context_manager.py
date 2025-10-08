"""Context management for handling token limits in the agentic scaffold."""
# mypy: disable-error-code="no-any-unimported"

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class ContextManager:
    """Manages conversation context to stay within token limits.

    This class provides utilities for:
    - Estimating token usage
    - Detecting when context is approaching limits
    - Summarizing message history to compress context
    """

    # Rough estimation: 4 characters ≈ 1 token (conservative)
    CHARS_PER_TOKEN = 4

    def __init__(self, max_tokens: int = 100000) -> None:
        """Initialize the context manager.

        Args:
            max_tokens: Maximum number of tokens allowed in context
        """
        self.max_tokens = max_tokens
        # Reserve 20% for responses and system prompts
        self.soft_limit = int(max_tokens * 0.8)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // self.CHARS_PER_TOKEN

    def count_message_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """Count total tokens across a sequence of messages.

        Args:
            messages: Messages to count tokens for

        Returns:
            Total estimated token count
        """
        total = 0
        for msg in messages:
            if hasattr(msg, "content"):
                total += self.estimate_tokens(str(msg.content))
        return total

    def should_summarize(self, messages: Sequence[BaseMessage], additional_text: str = "") -> bool:
        """Check if context should be summarized to avoid hitting limits.

        Args:
            messages: Current message history
            additional_text: Additional text that will be added (e.g., system prompt)

        Returns:
            True if summarization is recommended
        """
        current_tokens = self.count_message_tokens(messages)
        additional_tokens = self.estimate_tokens(additional_text)
        return (current_tokens + additional_tokens) > self.soft_limit

    def create_summary_message(self, summary: str) -> BaseMessage:
        """Create a system message containing a conversation summary.

        Args:
            summary: The summary text

        Returns:
            SystemMessage containing the summary
        """
        return SystemMessage(
            content=f"""# Context Summary

The following is a summary of previous actions and findings:

{summary}

# Current Task Continuation

You are now continuing from this point. Use the summary above as context for your next actions."""
        )

    def compress_messages(
        self, messages: Sequence[BaseMessage], summary: str, preserve_recent: int = 5
    ) -> list[BaseMessage]:
        """Compress message history by replacing old messages with a summary.

        Args:
            messages: Full message history
            summary: Summary of older messages
            preserve_recent: Number of recent messages to keep uncompressed

        Returns:
            Compressed message list with summary + recent messages
        """
        # Always preserve the most recent messages for immediate context
        recent_messages = list(messages[-preserve_recent:]) if len(messages) > preserve_recent else list(messages)

        # Create summary message
        summary_msg = self.create_summary_message(summary)

        # Return summary + recent messages
        return [summary_msg, *recent_messages]

    def get_context_info(self, messages: Sequence[BaseMessage], task: str = "", plan: str = "") -> str:
        """Get a string representation of current context usage.

        Args:
            messages: Current message history
            task: The task description
            plan: The current plan

        Returns:
            String describing context usage
        """
        msg_tokens = self.count_message_tokens(messages)
        task_tokens = self.estimate_tokens(task)
        plan_tokens = self.estimate_tokens(plan)
        total_tokens = msg_tokens + task_tokens + plan_tokens

        return f"""Context Usage:
- Messages: {msg_tokens:,} tokens
- Task: {task_tokens:,} tokens
- Plan: {plan_tokens:,} tokens
- Total: {total_tokens:,} / {self.max_tokens:,} tokens ({(total_tokens / self.max_tokens) * 100:.1f}%)
- Status: {"⚠️ APPROACHING LIMIT" if total_tokens > self.soft_limit else "✓ OK"}"""


def create_messages_summary(messages: Sequence[BaseMessage], max_messages: int = 20) -> str:
    """Create a text summary of recent messages for use in summarization prompts.

    Args:
        messages: Message history
        max_messages: Maximum number of recent messages to include

    Returns:
        Text summary of messages
    """
    recent = messages[-max_messages:] if len(messages) > max_messages else messages

    summary_parts = []
    for i, msg in enumerate(recent):
        role = "System"
        if isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "AI"

        content = str(msg.content)
        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."

        summary_parts.append(f"{i + 1}. [{role}] {content}")

    return "\n\n".join(summary_parts)
