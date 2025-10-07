# Agentic Scaffold

An autonomous task execution framework built on LangGraph and Alfredo tools.

## Overview

The agentic scaffold provides a ReAct-based agent that can:
- Automatically create implementation plans for tasks
- Execute tasks through a think-act-observe loop
- Verify results before completion
- Replan and retry if verification fails
- Manage context to stay within token limits

## Architecture

### Graph Flow

```
START → planner → agent ⟷ tools → extract_answer → verifier → END
                    ↓                                  ↓
                    ← ← ← ← ← ← replan ← ← ← ← ← ← ←
```

### Nodes

1. **Planner**: Creates a comprehensive implementation plan
2. **Agent**: Performs reasoning and decides on tool calls
3. **Tools**: Executes tool calls (file ops, commands, etc.)
4. **Extract Answer**: Extracts the final answer from attempt_answer tool
5. **Verifier**: Checks if the answer satisfies the original task
6. **Replan**: Creates an improved plan if verification fails

### State

The agent state (`AgentState`) contains:
- `messages`: Conversation history (messages between agent and tools)
- `task`: Original task description
- `plan`: Current implementation plan
- `plan_iteration`: Number of planning iterations
- `max_context_tokens`: Token limit for context management
- `final_answer`: Final answer from attempt_answer tool
- `is_verified`: Whether the answer has been verified

## Installation

```bash
# Install with agentic dependencies
uv add alfredo[agentic]

# Or install dependencies separately
uv add langgraph langchain langchain-core langchain-openai
```

## Usage

### Simple Task Execution

```python
from alfredo.agentic.graph import run_agentic_task

result = run_agentic_task(
    task="Create a Python script that prints Hello World",
    cwd=".",
    model_name="gpt-4o-mini",
    verbose=True
)

print(result["final_answer"])
```

### Direct Graph Usage

```python
from alfredo.agentic import create_agentic_graph, AgentState

# Create graph
graph = create_agentic_graph(
    cwd=".",
    model_name="gpt-4o-mini",
    max_context_tokens=100000
)

# Initial state
initial_state: AgentState = {
    "messages": [],
    "task": "Your task here",
    "plan": "",
    "plan_iteration": 0,
    "max_context_tokens": 100000,
    "final_answer": None,
    "is_verified": False,
}

# Run
final_state = graph.invoke(initial_state)
```

### Streaming Execution

```python
# Stream graph execution to see intermediate steps
for state in graph.stream(initial_state):
    for node_name, node_state in state.items():
        print(f"Node: {node_name}")
        # Process node state...
```

## Configuration

### Model Selection

The scaffold is model-agnostic and uses LangChain's `init_chat_model`:

```python
# OpenAI (default)
graph = create_agentic_graph(model_name="gpt-4o-mini")

# Anthropic
graph = create_agentic_graph(model_name="anthropic/claude-3-5-sonnet-20241022")

# OpenRouter
graph = create_agentic_graph(model_name="openrouter/anthropic/claude-3.5-sonnet")

# Local models
graph = create_agentic_graph(model_name="ollama/llama2")
```

### Context Management

```python
# Set maximum context tokens
graph = create_agentic_graph(
    max_context_tokens=100000  # Adjust based on model
)
```

### Custom Tools

```python
from langchain_core.tools import tool

@tool
def custom_tool(param: str) -> str:
    """My custom tool."""
    return f"Result: {param}"

# Pass custom tools
graph = create_agentic_graph(
    tools=[custom_tool]  # Replaces default Alfredo tools
)
```

## Available Tools

When using default tools, the agent has access to:

### File Operations
- `read_file`: Read file contents
- `write_to_file`: Create or overwrite files
- `replace_in_file`: Edit files with SEARCH/REPLACE

### Discovery
- `list_files`: List directory contents
- `search_files`: Search for files using regex

### Commands
- `execute_command`: Run shell commands

### Code Analysis
- `list_code_definition_names`: Extract function/class names

### Web
- `web_fetch`: Fetch and process web content

### Workflow
- `attempt_answer`: Signal task completion (required)

## How It Works

### 1. Planning Phase

The planner node receives the task and creates a detailed implementation plan:

```
Task: Create a Python script that calculates fibonacci numbers

Plan:
1. Create a new file fibonacci.py
2. Implement the fibonacci function
3. Add a main block with example usage
4. Test the implementation
...
```

### 2. Execution Loop

The agent follows a ReAct loop:

```
Agent: "I'll start by creating the file"
→ Tool Call: write_to_file(path="fibonacci.py", content="...")

Tool Result: "File created successfully"
→ Agent: "Now I'll test it"
→ Tool Call: execute_command(command="python fibonacci.py")

Tool Result: "Output: 0, 1, 1, 2, 3, 5..."
→ Agent: "The script works correctly"
→ Tool Call: attempt_answer(answer="I created fibonacci.py with...")
```

### 3. Verification

The verifier checks if the answer addresses the original task:

```
Verifier: Does the answer solve "Create a Python script..."?
→ YES: Task complete
→ NO: Create new plan and retry
```

### 4. Replanning (if needed)

If verification fails:

```
Previous attempt didn't include error handling.
New plan:
1. Add try-except blocks
2. Validate input
3. Add docstrings
...
```

## Key Features

### Auto-Accept Planning

Unlike interactive agents, the planner automatically creates and uses plans without user approval.

### Mandatory Completion Signal

The agent must explicitly call `attempt_answer` to signal completion. This prevents premature exits.

### Verification Loop

Answers are verified against the original task before completion. Failed verification triggers replanning.

### Context Management

The context manager monitors token usage and can summarize history to stay within limits.

### Model Agnostic

Works with any LangChain-compatible model (OpenAI, Anthropic, local models, etc.).

## Examples

See `examples/agentic_example.py` for comprehensive examples including:
- Simple task execution
- Direct graph usage
- Streaming execution
- Custom configurations

## Implementation Details

### File Structure

```
agentic/
├── __init__.py          # Public API
├── state.py             # State definitions
├── prompts.py           # System prompts
├── nodes.py             # Graph node implementations
├── context_manager.py   # Token tracking and summarization
├── graph.py             # Main graph builder
└── README.md            # This file
```

### Extending

#### Custom Nodes

```python
def custom_node(state: AgentState) -> dict:
    # Your logic here
    return {"messages": [new_message]}

graph.add_node("custom", custom_node)
graph.add_edge("agent", "custom")
```

#### Custom Prompts

```python
from alfredo.agentic import prompts

# Override prompt functions
prompts.get_planning_prompt = my_planning_prompt
```

#### Custom Verification

```python
def custom_verifier_node(model):
    def verifier(state):
        # Your verification logic
        return {"is_verified": True}
    return verifier

graph.add_node("verifier", custom_verifier_node(model))
```

## Troubleshooting

### ImportError: LangGraph not installed

```bash
uv add alfredo[agentic]
# or
uv add langgraph langchain langchain-core langchain-openai
```

### API Key Issues

```bash
# OpenAI
export OPENAI_API_KEY=your-key

# Anthropic
export ANTHROPIC_API_KEY=your-key
```

### Context Limit Exceeded

Reduce `max_context_tokens` or enable context summarization:

```python
from alfredo.agentic.context_manager import ContextManager

cm = ContextManager(max_tokens=50000)
if cm.should_summarize(messages):
    # Trigger summarization
    pass
```

### Verification Always Fails

Check verification prompt or provide more detailed answers:

```python
# More detailed answer
attempt_answer(
    answer="""
    I completed the task by:
    1. Created file X with content Y
    2. Tested with command Z
    3. Result: Success
    """
)
```

## Testing

```bash
# Run agentic tests
uv run pytest tests/test_agentic.py -v

# Run with coverage
uv run pytest tests/test_agentic.py --cov=alfredo.agentic
```

## Contributing

When adding features to the agentic scaffold:

1. Add tests in `tests/test_agentic.py`
2. Update this README
3. Add examples if introducing new concepts
4. Ensure type hints are complete
5. Run formatting: `uv run ruff format src/alfredo/agentic/`

## License

Same as Alfredo project license.
