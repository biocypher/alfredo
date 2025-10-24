"""Tests for reasoning parser functionality."""

from langchain_core.messages import AIMessage

from alfredo.agentic.reasoning_parser import get_reasoning_from_message, parse_reasoning_from_response


def test_parse_reasoning_simple() -> None:
    """Test parsing a simple think tag."""
    msg = AIMessage(content="<think>My reasoning here</think>\nMy answer")
    parsed = parse_reasoning_from_response(msg)

    assert parsed.additional_kwargs["reasoning"] == "My reasoning here"
    assert parsed.content == "My answer"


def test_parse_reasoning_multiline() -> None:
    """Test parsing multiline reasoning."""
    content = """<think>
    First, I need to consider X
    Then, I should do Y
    Finally, Z makes sense
    </think>

    Here is my final answer."""

    msg = AIMessage(content=content)
    parsed = parse_reasoning_from_response(msg)

    expected_reasoning = "First, I need to consider X\n    Then, I should do Y\n    Finally, Z makes sense"
    assert parsed.additional_kwargs["reasoning"] == expected_reasoning
    assert "Here is my final answer." in parsed.content
    assert "<think>" not in parsed.content


def test_parse_reasoning_multiple_blocks() -> None:
    """Test parsing multiple think blocks."""
    content = """<think>First thought</think>
    Some content
    <think>Second thought</think>
    More content"""

    msg = AIMessage(content=content)
    parsed = parse_reasoning_from_response(msg)

    assert parsed.additional_kwargs["reasoning"] == "First thought\n\nSecond thought"
    assert "Some content" in parsed.content
    assert "More content" in parsed.content
    assert "<think>" not in parsed.content


def test_parse_reasoning_no_tags() -> None:
    """Test parsing when no think tags present."""
    msg = AIMessage(content="Just regular content")
    parsed = parse_reasoning_from_response(msg)

    assert parsed.content == "Just regular content"
    assert "reasoning" not in parsed.additional_kwargs


def test_parse_reasoning_preserves_tool_calls() -> None:
    """Test that parsing preserves tool calls."""
    msg = AIMessage(
        content="<think>I should use a tool</think>\nLet me call the tool",
        tool_calls=[{"name": "test_tool", "args": {"param": "value"}, "id": "call_123"}],
    )
    parsed = parse_reasoning_from_response(msg)

    assert parsed.additional_kwargs["reasoning"] == "I should use a tool"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "test_tool"


def test_get_reasoning_from_message() -> None:
    """Test extracting reasoning from parsed message."""
    msg = AIMessage(content="<think>My reasoning</think>\nAnswer", additional_kwargs={})
    parsed = parse_reasoning_from_response(msg)

    reasoning = get_reasoning_from_message(parsed)
    assert reasoning == "My reasoning"


def test_get_reasoning_from_message_no_reasoning() -> None:
    """Test extracting reasoning when none present."""
    msg = AIMessage(content="Just content")
    reasoning = get_reasoning_from_message(msg)
    assert reasoning is None


def test_parse_reasoning_empty_tags() -> None:
    """Test parsing empty think tags."""
    msg = AIMessage(content="<think></think>\nMy answer")
    parsed = parse_reasoning_from_response(msg)

    assert parsed.additional_kwargs["reasoning"] == ""
    assert parsed.content == "My answer"


def test_parse_reasoning_whitespace_cleaning() -> None:
    """Test that excessive whitespace is cleaned up."""
    content = """<think>Reasoning</think>




    Final answer"""

    msg = AIMessage(content=content)
    parsed = parse_reasoning_from_response(msg)

    assert parsed.additional_kwargs["reasoning"] == "Reasoning"
    assert parsed.content == "Final answer"
    # Should have cleaned up excessive newlines
    assert "\n\n\n" not in parsed.content
