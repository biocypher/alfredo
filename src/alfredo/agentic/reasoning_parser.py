"""Utility for parsing reasoning content from model responses."""

import re
from typing import Optional, Union

from langchain_core.messages import AIMessage, BaseMessage


def parse_reasoning_from_response(response: Union[AIMessage, BaseMessage]) -> Union[AIMessage, BaseMessage]:
    """Parse <think>...</think> tags from AIMessage and store in additional_kwargs.

    Extracts content enclosed in <think> tags and stores it in the 'reasoning' field
    of additional_kwargs. The thinking content is removed from the main content field.

    Args:
        response: AIMessage or BaseMessage from model invocation

    Returns:
        Modified AIMessage with reasoning extracted and stored in additional_kwargs.
        If the input is not an AIMessage, returns it unchanged.

    Example:
        >>> msg = AIMessage(content="<think>My reasoning</think>\\nMy answer")
        >>> parsed = parse_reasoning_from_response(msg)
        >>> parsed.additional_kwargs['reasoning']
        'My reasoning'
        >>> parsed.content
        'My answer'
    """
    # Only process AIMessages
    if not isinstance(response, AIMessage):
        return response

    if not hasattr(response, "content") or not response.content:
        return response

    content = str(response.content)

    # Look for <think>...</think> tags
    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, content, re.DOTALL)

    if not matches:
        # No thinking tags found
        return response

    # Extract all thinking content
    reasoning_parts = [match.strip() for match in matches]
    reasoning = "\n\n".join(reasoning_parts)

    # Remove thinking tags from content
    cleaned_content = re.sub(think_pattern, "", content, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned_content = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_content).strip()

    # Create new AIMessage with modified content and reasoning in additional_kwargs
    # Preserve all other attributes
    additional_kwargs = dict(response.additional_kwargs) if response.additional_kwargs else {}
    additional_kwargs["reasoning"] = reasoning

    # Create new message with updated fields
    new_message = AIMessage(
        content=cleaned_content,
        additional_kwargs=additional_kwargs,
        response_metadata=response.response_metadata if hasattr(response, "response_metadata") else {},
        id=response.id if hasattr(response, "id") else None,
        tool_calls=response.tool_calls if hasattr(response, "tool_calls") else [],
        invalid_tool_calls=response.invalid_tool_calls if hasattr(response, "invalid_tool_calls") else [],
        usage_metadata=response.usage_metadata if hasattr(response, "usage_metadata") else None,
    )

    return new_message


def get_reasoning_from_message(message: AIMessage) -> Optional[str]:
    """Extract reasoning content from an AIMessage's additional_kwargs.

    Args:
        message: AIMessage that may contain reasoning

    Returns:
        Reasoning string if present, None otherwise

    Example:
        >>> reasoning = get_reasoning_from_message(msg)
        >>> if reasoning:
        ...     print(f"Model's reasoning: {reasoning}")
    """
    if not hasattr(message, "additional_kwargs"):
        return None

    return message.additional_kwargs.get("reasoning")
