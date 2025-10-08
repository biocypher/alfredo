"""Example of using Alfredo tools with native OpenAI API.

This example demonstrates the native OpenAI integration without LangChain dependency.
For most use cases, prefer the LangGraph agentic scaffold (agentic_example.py).

Requirements:
    uv add openai
    export OPENAI_API_KEY=your-api-key
"""

import os
from pathlib import Path

# Check if OpenAI is available
try:
    from alfredo import OpenAIAgent, get_all_tools_openai_format, tool_spec_to_openai_format
except ImportError:
    print("OpenAI integration not available. Install with: uv add openai")
    exit(1)

# Import handlers to register tools
from alfredo.tools.handlers import file_ops  # noqa: F401
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily

print("=" * 80)
print("ALFREDO + OPENAI NATIVE INTEGRATION EXAMPLE")
print("=" * 80)

# Example 1: Convert tools to OpenAI format
print("\n### Example 1: Convert Alfredo tools to OpenAI format ###\n")

# Get a single tool spec and convert it
read_file_spec = registry.get_spec("read_file", ModelFamily.OPENAI)
if read_file_spec:
    openai_tool = tool_spec_to_openai_format(read_file_spec)
    print("Tool in OpenAI format:")
    import json

    print(json.dumps(openai_tool, indent=2))

# Example 2: Get all tools in OpenAI format
print("\n### Example 2: Get all tools in OpenAI format ###\n")

all_tools = get_all_tools_openai_format()
print(f"Total tools available: {len(all_tools)}")
for tool in all_tools[:3]:  # Show first 3
    print(f"  - {tool['function']['name']}: {tool['function']['description'][:60]}...")

# Example 3: Use OpenAI Agent
print("\n### Example 3: Use OpenAI Agent ###\n")

if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Set OPENAI_API_KEY environment variable to run this example")
    print("   Example: export OPENAI_API_KEY=your-api-key")
else:
    print("Creating OpenAI agent...")

    # Create agent
    agent = OpenAIAgent(
        cwd=str(Path.cwd()),
        model="gpt-4o-mini",  # or gpt-4, gpt-3.5-turbo, etc.
    )

    # Create a test file
    test_file = Path("test_openai_native.txt")
    test_file.write_text("Hello from OpenAI native integration!\nThis is line 2.\nThis is line 3.")
    print(f"✓ Created test file: {test_file}")

    # Example 3a: Simple file read
    print("\n--- Task: Read a file ---")
    result = agent.run(
        message=f"Read the file {test_file}",
        system_prompt="You are a helpful assistant. Use the provided tools to complete tasks.",
    )

    print(f"Response: {result['content'][:200]}...")
    print(f"Tools used: {len(result['tool_results'])}")
    for tool_result in result["tool_results"]:
        print(f"  - {tool_result['tool']}: {'✓' if tool_result['success'] else '✗'}")

    # Example 3b: Multi-step task
    print("\n--- Task: Multi-step file operations ---")
    result = agent.run(
        message="Read the test_openai_native.txt file, count the lines, and write a summary to summary.txt",
    )

    print(f"Response: {result['content']}")
    print(f"Tools used: {len(result['tool_results'])}")
    for tool_result in result["tool_results"]:
        status = "✓" if tool_result["success"] else "✗"
        print(f"  - {tool_result['tool']}: {status}")
        if tool_result["args"] and "path" in tool_result["args"]:
            # Show args for write operations
            print(f"    Path: {tool_result['args']['path']}")

    # Clean up
    test_file.unlink()
    summary_file = Path("summary.txt")
    if summary_file.exists():
        summary_file.unlink()
    print("\n✓ Cleaned up test files")

# Example 4: Custom system prompt
print("\n### Example 4: Custom system prompt ###\n")

if os.getenv("OPENAI_API_KEY"):
    agent = OpenAIAgent(cwd=".")

    custom_prompt = """You are a code analysis assistant specializing in Python projects.
When asked to analyze code, use the available tools to:
1. List files in the directory
2. Read relevant files
3. Provide insights about code structure and quality
"""

    result = agent.run(
        message="What Python files are in the current directory?",
        system_prompt=custom_prompt,
        max_iterations=5,
    )

    print(f"Response: {result['content'][:300]}...")

# Example 5: Compare with LangChain approach
print("\n### Example 5: Comparison with LangChain approach ###\n")

print("Native OpenAI Agent:")
print("  ✓ Direct OpenAI API control")
print("  ✓ No LangChain dependency")
print("  ✓ Simpler for basic tool calling")
print("  ✗ Manual conversation loop management")
print("  ✗ No built-in planning/verification")
print()
print("LangGraph Agentic Scaffold (RECOMMENDED):")
print("  ✓ Automatic planning and replanning")
print("  ✓ Task verification")
print("  ✓ Sophisticated tool orchestration")
print("  ✓ Works with any LLM provider")
print("  ✓ Production-ready")
print()
print("Use native OpenAI agent when:")
print("  - Building custom tool calling loops")
print("  - Need direct OpenAI API control")
print("  - Want minimal dependencies")
print()
print("Use LangGraph scaffold when:")
print("  - Need reliable task completion")
print("  - Want automatic planning/verification")
print("  - Building production agents")

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Try the LangGraph scaffold: uv run python examples/agentic_example.py")
print("  2. Build custom tool calling loops with OpenAIAgent")
print("  3. Add custom tools to the system")
print("\nSee docs: https://github.com/your-repo/alfredo")
