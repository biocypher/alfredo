"""Minimal test to diagnose the recursion issue."""

import os
import sys

from dotenv import load_dotenv

from alfredo.agentic.graph import create_agentic_graph
from alfredo.agentic.state import AgentState

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set")
    sys.exit(1)

print("Creating graph...")
graph = create_agentic_graph(model_name="gpt-4o-mini", cwd=".")

initial_state: AgentState = {
    "messages": [],
    "task": "Create a file called test.txt with the word: Hello",
    "plan": "",
    "plan_iteration": 0,
    "max_context_tokens": 100000,
    "final_answer": None,
    "is_verified": False,
}

print("Task:", initial_state["task"])

print("Starting execution with streaming...\n")
step_count = 0
node_sequence = []

try:
    for step in graph.stream(initial_state):
        step_count += 1
        for node_name in step:
            node_sequence.append(node_name)
            print(f"Step {step_count}: {node_name}")

            # Stop if we're looping too much
            if step_count > 30:
                print("\n⚠️  Stopped after 30 steps to prevent timeout")
                print(f"Node sequence: {' → '.join(node_sequence)}")
                break

    print(f"\n✅ Completed in {step_count} steps")
    print(f"Sequence: {' → '.join(node_sequence)}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"Stopped at step {step_count}")
    print(f"Node sequence so far: {' → '.join(node_sequence)}")
