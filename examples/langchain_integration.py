"""Example of using Alfredo tools with LangChain/LangGraph.

This example demonstrates how to convert Alfredo tools into LangChain-compatible tools
that can be used with LangChain agents and workflows.

Requirements:
    uv add langchain-core langchain-anthropic
"""

from pathlib import Path

# Check if LangChain is available
try:
    from langchain_core.tools import tool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain is not installed. Install with: uv add langchain-core")
    exit(1)

from alfredo.integrations.langchain import (
    create_all_langchain_tools,
    create_langchain_tool,
)

# Import handlers to ensure tools are registered
from alfredo.tools.handlers import command, discovery, file_ops, workflow  # noqa: F401

print("=" * 80)
print("ALFREDO + LANGCHAIN INTEGRATION EXAMPLE")
print("=" * 80)

# Example 1: Convert a single tool
print("\n### Example 1: Convert a single Alfredo tool to LangChain ###\n")

read_file_tool = create_langchain_tool("read_file", cwd=str(Path.cwd()))

print(f"Tool name: {read_file_tool.name}")
print(f"Tool description: {read_file_tool.description}")
print(f"Tool args schema: {read_file_tool.args_schema.model_json_schema()}")

# Example 2: Convert all tools
print("\n### Example 2: Convert all Alfredo tools to LangChain ###\n")

all_tools = create_all_langchain_tools(cwd=str(Path.cwd()))

print(f"Total tools converted: {len(all_tools)}")
for tool in all_tools:
    print(f"  - {tool.name}: {tool.description[:60]}...")

# Example 3: Use a tool directly
print("\n### Example 3: Invoke a tool directly ###\n")

# Create a test file
test_file = Path("test_langchain.txt")
test_file.write_text("Hello from LangChain integration!")

# Use the read_file tool
result = read_file_tool.invoke({"path": "test_langchain.txt"})
print(f"Read file result:\n{result}")

# Clean up
test_file.unlink()

# Example 4: Using StructuredTool directly
print("\n### Example 4: Using tools individually ###\n")

list_tool = create_langchain_tool("list_files")

# Invoke the tool
print("Listing current directory:")
dir_result = list_tool.invoke({"path": ".", "recursive": "false"})
print(dir_result[:300] + "..." if len(dir_result) > 300 else dir_result)

# Example 5: Tool metadata for LangChain agents
print("\n### Example 5: Tool metadata for agents ###\n")

write_tool = create_langchain_tool("write_to_file")

print(f"Tool: {write_tool.name}")
print(f"Description: {write_tool.description}")
print("\nArgs schema:")
for field_name, field_info in write_tool.args_schema.model_fields.items():
    required = "required" if field_info.is_required() else "optional"
    print(f"  - {field_name} ({required}): {field_info.description}")

# Example 6: Using tools with LangChain agent (if Anthropic API key available)
print("\n### Example 6: Using with LangChain agent ###")

try:
    import os

    from langchain_anthropic import ChatAnthropic

    if os.getenv("ANTHROPIC_API_KEY"):
        print("\nCreating LangChain agent with Alfredo tools...")

        # Create model with tools
        model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        model_with_tools = model.bind_tools(all_tools[:3])  # Bind first 3 tools

        print(f"Model configured with {len(all_tools[:3])} tools")
        print("\nYou can now use this model in a LangChain agent or LangGraph workflow!")
        print("\nExample usage:")
        print("  response = model_with_tools.invoke('List files in the current directory')")
        print("  # The model will use the list_files tool")
    else:
        print("\nSet ANTHROPIC_API_KEY to run this example with an actual LLM.")
        print("Example: export ANTHROPIC_API_KEY=your-api-key")
except ImportError:
    print("\nInstall langchain-anthropic to use with Claude models:")
    print("  uv add langchain-anthropic")

print("\n" + "=" * 80)
print("INTEGRATION EXAMPLES COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Use these tools in a LangChain ReAct agent")
print("  2. Integrate into a LangGraph workflow")
print("  3. Create custom tool compositions")
print("\nSee LangChain docs: https://python.langchain.com/docs/how_to/custom_tools/")
