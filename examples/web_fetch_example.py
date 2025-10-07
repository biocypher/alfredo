"""Example: Using web_fetch tool with LangChain.

This example demonstrates how to use Alfredo's web_fetch tool
with LangChain agents to fetch and analyze web content.
"""

import os

from alfredo.integrations.langchain import create_langchain_tool

# Create the web_fetch tool for LangChain
web_fetch_tool = create_langchain_tool("web_fetch")

print("=" * 80)
print("Example 1: Direct Tool Invocation")
print("=" * 80)

# Example 1: Use the tool directly
result = web_fetch_tool.invoke({"url": "https://example.com"})
print(f"\nFetched content preview:\n{result[:500]}...")

print("\n" + "=" * 80)
print("Example 2: Using with LangChain Agent (requires API key)")
print("=" * 80)

# Example 2: Use with a LangChain agent
# This requires an API key set in environment variables
if os.getenv("ANTHROPIC_API_KEY"):
    from langchain_anthropic import ChatAnthropic

    # Create model with tools
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    model_with_tools = model.bind_tools([web_fetch_tool])

    # Example query
    messages = [
        {
            "role": "user",
            "content": "Please fetch the content from https://example.com and summarize what you find.",
        }
    ]

    print("\nSending request to Claude with web_fetch tool...")
    response = model_with_tools.invoke(messages)

    print(f"\nResponse: {response.content}")

    # Check if the model wants to use tools
    if response.tool_calls:
        print(f"\nTool calls requested: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call['name']}: {tool_call['args']}")

            # Execute the tool
            tool_result = web_fetch_tool.invoke(tool_call["args"])
            print(f"\n  Tool result preview: {tool_result[:200]}...")

            # Send tool result back to the model
            messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})  # type: ignore[dict-item]
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call["id"],
            })

            # Get final response
            final_response = model_with_tools.invoke(messages)
            print(f"\n  Final summary: {final_response.content}")
else:
    print("\nSkipping agent example (set ANTHROPIC_API_KEY to run)")

print("\n" + "=" * 80)
print("Example 3: Using with LangChain LCEL Chain")
print("=" * 80)

if os.getenv("ANTHROPIC_API_KEY"):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # Create a chain that uses the web_fetch tool
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Manual chain: fetch URL then analyze
    url = "https://example.com"
    print(f"\nFetching {url}...")
    web_content = web_fetch_tool.invoke({"url": url})

    # Now analyze with the model
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that analyzes web content."),
        ("user", "Here is the content from a webpage:\n\n{content}\n\nPlease provide a brief summary."),
    ])

    chain = prompt | model | StrOutputParser()
    summary = chain.invoke({"content": web_content[:2000]})  # Limit content length
    print(f"\nSummary: {summary}")
else:
    print("\nSkipping LCEL chain example (set ANTHROPIC_API_KEY to run)")

print("\n" + "=" * 80)
print("Example 4: Error Handling")
print("=" * 80)

# Example 4: Error handling
print("\nTesting with invalid URL...")
result = web_fetch_tool.invoke({"url": "not-a-valid-url"})
print(f"Result: {result}")

print("\nTesting with non-existent domain...")
result = web_fetch_tool.invoke({"url": "https://this-domain-definitely-does-not-exist-12345.com"})
print(f"Result: {result[:200]}...")

print("\n" + "=" * 80)
print("Example 5: Multiple Tools Combined")
print("=" * 80)

if os.getenv("ANTHROPIC_API_KEY"):
    from langchain_anthropic import ChatAnthropic

    from alfredo.integrations.langchain import create_all_langchain_tools

    # Get all Alfredo tools including web_fetch, list_files, etc.
    all_tools = create_all_langchain_tools()

    print(f"\nAvailable tools: {[tool.name for tool in all_tools]}")

    # Create agent with multiple tools
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    model_with_tools = model.bind_tools(all_tools)

    messages = [
        {
            "role": "user",
            "content": (
                "Please fetch content from https://example.com and save a summary to a file called summary.txt"
            ),
        }
    ]

    print("\nThis would allow the agent to:")
    print("  1. Use web_fetch to get the content")
    print("  2. Use write_to_file to save the summary")
    print("\n(Full implementation would require agent loop)")
else:
    print("\nSkipping multi-tool example (set ANTHROPIC_API_KEY to run)")

print("\n" + "=" * 80)
print("Examples complete!")
print("=" * 80)
