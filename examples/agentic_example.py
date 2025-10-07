"""Example of using the agentic scaffold for autonomous task execution.

This example demonstrates how to use the agentic scaffold to execute tasks
with automatic planning, execution, and verification.

Requirements:
    uv add alfredo[agentic]

Environment:
    Create a .env file in the project root with your API key:
        OPENAI_API_KEY=your-key-here

    Or set it manually:
        export OPENAI_API_KEY=your-key-here

    For other providers (Anthropic, etc.), set the appropriate key and change model_name
"""

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Check if LangGraph is available
try:
    from langchain_core.messages import HumanMessage

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph is not installed. Install with: uv add alfredo[agentic]")
    exit(1)

from alfredo.agentic.graph import run_agentic_task

print("=" * 80)
print("AGENTIC SCAFFOLD EXAMPLE")
print("=" * 80)

# Example 1: Simple file creation task
print("\n### Example 1: Simple Task Execution ###\n")

# Check if API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  OPENAI_API_KEY not set. Set it to run with OpenAI models.")
    print("\n   Option 1: Create a .env file in the project root:")
    print("   OPENAI_API_KEY=your-api-key")
    print("\n   Option 2: Export environment variable:")
    print("   export OPENAI_API_KEY=your-api-key")
    print("\n   You can also use other providers by changing model_name:")
    print("   - Anthropic: model_name='anthropic/claude-3-5-sonnet-20241022'")
    print("   - OpenRouter: model_name='openrouter/...'")
    print("\n   Skipping live examples...\n")
else:
    # Run a simple task
    task = """Create a Python script called 'hello_world.py' that prints "Hello, Agentic World!"
    and includes a docstring explaining what it does."""

    print(f"Task: {task}\n")
    print("Running agentic scaffold...\n")

    result = run_agentic_task(
        task=task,
        cwd=str(Path.cwd()),
        model_name="gpt-4.1-mini",  # Using GPT-4o-mini as default
        max_context_tokens=100000,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Result received!")
    print(f"Plan iterations: {result['plan_iteration']}")
    print(f"Total messages: {len(result['messages'])}")
    print(f"Verified: {result['is_verified']}")


# Example 2: Using the graph directly for more control (commented out - advanced usage)
# print("\n\n### Example 2: Direct Graph Usage ###\n")
#
# if os.getenv("OPENAI_API_KEY"):
#     # Create the graph
#     graph = create_agentic_graph(
#         cwd=str(Path.cwd()),
#         model_name="gpt-4.1-mini",
#         max_context_tokens=100000,
#     )
#
#     # Define initial state
#     from alfredo.agentic.state import AgentState
#
#     initial_state: AgentState = {
#         "messages": [],
#         "task": "List all Python files in the current directory and count how many there are.",
#         "plan": "",
#         "plan_iteration": 0,
#         "max_context_tokens": 100000,
#         "final_answer": None,
#         "is_verified": False,
#     }
#
#     print(f"Task: {initial_state['task']}\n")
#     print("Invoking graph...\n")
#
#     # Invoke the graph
#     final_state = graph.invoke(initial_state)
#
#     print("\n✅ Task completed!")
#     print(f"\n📋 Plan created:")
#     print(final_state["plan"][:200] + "...")
#     print(f"\n📝 Final Answer:")
#     print(final_state["final_answer"])


# Example 3: Streaming execution (commented out - advanced usage)
# print("\n\n### Example 3: Streaming Execution ###\n")
#
# if os.getenv("OPENAI_API_KEY"):
#     print("Task: Create a JSON file with project metadata\n")
#
#     graph = create_agentic_graph(
#         cwd=str(Path.cwd()),
#         model_name="gpt-4.1-mini",
#     )
#
#     initial_state: AgentState = {
#         "messages": [],
#         "task": "Create a JSON file called 'project.json' with fields: name, version, and description.",
#         "plan": "",
#         "plan_iteration": 0,
#         "max_context_tokens": 100000,
#         "final_answer": None,
#         "is_verified": False,
#     }
#
#     print("Streaming graph execution:\n")
#
#     # Stream the graph execution
#     for i, state in enumerate(graph.stream(initial_state)):
#         print(f"\n--- Step {i + 1} ---")
#         # state is a dict with node name as key
#         for node_name, node_state in state.items():
#             print(f"Node: {node_name}")
#             if "plan" in node_state and node_state["plan"]:
#                 print(f"Plan created (first 100 chars): {node_state['plan'][:100]}...")
#             if "final_answer" in node_state and node_state["final_answer"]:
#                 print(f"Answer: {node_state['final_answer'][:100]}...")
#
#     print("\n✅ Streaming complete!")


# Example 4: Information about the scaffold
print("\n\n### Example 4: Scaffold Architecture ###\n")

print("""
The agentic scaffold uses a LangGraph-based ReAct loop with the following components:

1. **Planner Node**: Creates an implementation plan for the task
2. **Agent Node**: Performs reasoning and calls tools
3. **Tools Node**: Executes tool calls (file ops, commands, etc.)
4. **Verifier Node**: Checks if the answer satisfies the task
5. **Replan Node**: Creates a new plan if verification fails

Graph Flow:
    START → planner → agent ⟷ tools → verifier → END
                              ↓          ↓
                              ← replan ←

Key Features:
- ✅ Automatic planning (no user approval needed)
- ✅ ReAct-style think-act-observe loop
- ✅ Mandatory attempt_answer tool call to complete
- ✅ Answer verification before completion
- ✅ Replanning on verification failure
- ✅ Model-agnostic (works with OpenAI, Anthropic, etc.)
- ✅ Context management for long conversations
- ✅ Access to all Alfredo tools (file ops, commands, web, etc.)

Tools Available:
- File operations (read, write, edit)
- Command execution
- File discovery (list, search)
- Code analysis
- Web fetching
- And more...
""")

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Set your API key: export OPENAI_API_KEY=your-key")
print("  2. Run this example: uv run python examples/agentic_example.py")
print("  3. Try your own tasks with run_agentic_task()")
print("  4. Customize with different models and tools")
print("\nDocumentation: See alfredo/src/alfredo/agentic/ for implementation details")
