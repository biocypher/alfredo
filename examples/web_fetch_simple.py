"""Simple example: web_fetch tool with LangChain agent.

This example shows a minimal working example of using the web_fetch tool
with a LangChain agent to fetch and analyze web content.

Requirements:
    pip install langchain-anthropic
    export ANTHROPIC_API_KEY=your_key_here
"""

import os
import sys

from alfredo.integrations.langchain import create_langchain_tool

# Check for API key
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    print("Please run: export ANTHROPIC_API_KEY=your_key_here")
    sys.exit(1)

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    print("Error: langchain-anthropic not installed")
    print("Please run: pip install langchain-anthropic")
    sys.exit(1)

# Create the web_fetch tool
print("Creating web_fetch tool for LangChain...")
web_fetch_tool = create_langchain_tool("web_fetch")

# Create Claude model with the tool
print("Initializing Claude model...")
model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
model_with_tools = model.bind_tools([web_fetch_tool])

# Define the URL to fetch
url = "https://python.org"
print(f"\nAsking Claude to fetch and analyze: {url}")

# Create initial message
messages = [
    {
        "role": "user",
        "content": f"Please fetch the content from {url} and tell me what you find. What is this website about?",
    }
]

# Get initial response (may include tool calls)
print("\nSending request to Claude...")
response = model_with_tools.invoke(messages)

# Check if Claude wants to use the tool
if response.tool_calls:
    print(f"\nClaude requested to use {len(response.tool_calls)} tool(s):")

    for tool_call in response.tool_calls:
        print(f"  - Tool: {tool_call['name']}")
        print(f"    Args: {tool_call['args']}")

        # Execute the tool
        print("\n  Executing tool...")
        tool_result = web_fetch_tool.invoke(tool_call["args"])
        print(f"  Fetched {len(tool_result)} characters")

        # Add tool execution to conversation
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})  # type: ignore[dict-item]
        messages.append({
            "role": "tool",
            "content": tool_result,
            "tool_call_id": tool_call["id"],
        })

    # Get final response after tool execution
    print("\nGetting final response from Claude...")
    final_response = model_with_tools.invoke(messages)
    print(f"\n{'=' * 80}")
    print("Claude's Analysis:")
    print(f"{'=' * 80}")
    print(final_response.content)
else:
    # No tool call needed
    print(f"\n{'=' * 80}")
    print("Claude's Response:")
    print(f"{'=' * 80}")
    print(response.content)

print(f"\n{'=' * 80}")
print("Example complete!")
print(f"{'=' * 80}")
