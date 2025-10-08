# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alfredo is a Python harness for AI agents, providing a comprehensive tool system that allows AI agents to interact with the file system, execute commands, and control workflow. The tool system is inspired by and ported from the Cline coding agent.

## Package Management

**IMPORTANT**: This project uses `uv` as the package manager, not pip.

### Setup and Installation

```bash
# Install dependencies
make install

# Or manually with uv
uv sync
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

### Running Commands

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_agent.py -v

# Run with coverage
uv run pytest --cov=src

# Run pre-commit hooks
uv run pre-commit run -a

# Run type checking
uv run mypy src

# Run linting
uv run ruff check src
```

### Development Workflow

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all quality checks
make check

# Run tests
make test

# Build documentation
uv run mkdocs serve
```

## LangChain/LangGraph Integration

**IMPORTANT**: LangGraph is now a **required dependency** (not optional). The project includes a full agentic scaffold.

### Convert Tools to LangChain

```python
from alfredo.integrations.langchain import create_all_langchain_tools

# Get all tools as LangChain StructuredTools
tools = create_all_langchain_tools(cwd="/path/to/workspace")

# Use with LangChain agents
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)
```

### Run Agentic Scaffold

```python
from alfredo.agentic.graph import run_agentic_task

# Run a task with plan-verify-replan loop
result = run_agentic_task(
    task="Create a hello world Python script",
    cwd=".",
    model_name="gpt-4o-mini",
    verbose=True,
    recursion_limit=50
)
```

### Examples

```bash
# Run basic LangChain example
uv run python examples/langchain_integration.py

# Run agentic scaffold example
uv run python examples/agentic_example.py

# Test agentic scaffold
uv run pytest tests/test_agentic_graph.py -v
```

## Architecture

Alfredo provides a layered architecture with multiple usage patterns to fit different use cases.

## Usage Patterns

### Primary Usage: Agentic Scaffold (Recommended)

**For most users, the LangGraph agentic scaffold is the recommended approach.** It provides:
- Automatic planning and replanning
- Verification of task completion
- Sophisticated tool orchestration
- Support for any LLM provider (OpenAI, Anthropic, etc.)

```python
from alfredo.agentic.graph import run_agentic_task

result = run_agentic_task(
    task="Create a hello world Python script",
    cwd=".",
    model_name="gpt-4o-mini",
    verbose=True
)
```

The agentic scaffold uses LangChain to automatically convert Alfredo tools to the native format of any LLM provider (OpenAI's function calling, Anthropic's tool use, etc.).

### Alternative Usage Patterns

For specialized use cases or direct control, Alfredo provides alternative execution modes:

**1. Native OpenAI Agent** - Direct OpenAI API integration without LangChain:

```python
from alfredo import OpenAIAgent

agent = OpenAIAgent(cwd=".", model="gpt-4o-mini")
result = agent.run("Read the file config.json")
```

Use this when:
- You want direct OpenAI API control without LangChain
- You're building custom tool calling loops
- You need minimal dependencies

**2. XML-Based Agent** - Simple XML parsing for text-based tool invocations:

```python
from alfredo import Agent

agent = Agent(cwd=".")
prompt = agent.get_system_prompt()  # Contains XML tool definitions
# Send prompt to LLM, get XML response
result = agent.execute_from_text(llm_response)
```

Use this when:
- You need a simple, minimal execution model
- You're using Anthropic's XML-style prompts directly
- You're building custom agent loops

## Core Architecture

### Tools System

The core tool system allows AI agents to interact with the environment through a modular handler architecture:

**Tool Categories:**
- **File Operations** (`file_ops.py`) - Read, write, diff-based edits
- **Command Execution** (`command.py`) - Shell commands with timeout control
- **File Discovery** (`discovery.py`) - List directories, search with regex
- **Code Analysis** (`code_analysis.py`) - List code definitions using tree-sitter
- **Web Tools** (`web.py`) - Fetch and convert web content to markdown
- **Workflow Control** (`workflow.py`) - Ask questions, signal completion

**Key Components:**
```
src/alfredo/
├── agent.py              # XML-based Agent class
├── tools/
│   ├── specs.py          # Tool specifications (ToolSpec, ToolParameter)
│   ├── registry.py       # Tool registry (singleton pattern)
│   ├── base.py           # BaseToolHandler class
│   └── handlers/         # Tool implementations
├── prompts/
│   └── builder.py        # System prompt generation
├── integrations/
│   ├── langchain.py      # LangChain/LangGraph integration
│   └── openai_native.py  # Native OpenAI API integration
└── agentic/              # LangGraph-based agentic scaffold
```

**Design Patterns:**
- **Tool Registry**: Singleton managing tool specs and handlers
- **Specification-based**: Tools defined via `ToolSpec` with parameters
- **Handler Pattern**: Each tool has a handler class implementing `execute()`
- **Model Family Variants**: Support for Anthropic/OpenAI/Gemini-specific prompts
- **Format Adapters**: XML format for text-based agents, native OpenAI/Anthropic formats via LangChain

### Agentic Scaffold Implementation

The agentic scaffold provides sophisticated LangGraph-based execution with plan-verify-replan loop:

**Graph Structure:**
```
START → planner → agent ⇄ tools → verifier
                   ↑              ↓
                   └── replan ←───┘
```

**Nodes:**
- `planner` - Creates initial implementation plan
- `agent` - Performs reasoning and calls tools
- `tools` - Executes tool calls (uses LangGraph's ToolNode)
- `verifier` - Checks if task is complete and satisfactory
- `replan` - Generates improved plan after verification failure

**State Management (`AgentState`):**
- `messages` - Conversation history (LangChain messages)
- `task` - Original user task
- `plan` - Current implementation plan
- `plan_iteration` - Plan version number
- `final_answer` - Result from `attempt_completion` tool
- `is_verified` - Whether verifier approved the answer

**Key Files:**
- `agentic/graph.py` - Graph construction and routing logic
- `agentic/nodes.py` - Node implementations
- `agentic/state.py` - State type definitions
- `agentic/prompts.py` - System prompts for each node

## Adding New Tools

To add a new tool:

1. **Create a handler class** in `src/alfredo/tools/handlers/`:

```python
from alfredo.tools.base import BaseToolHandler, ToolResult
from typing import Any

class MyToolHandler(BaseToolHandler):
    @property
    def tool_id(self) -> str:
        return "my_tool"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        self.validate_required_param(params, "param_name")
        # Your implementation
        return ToolResult.ok("Success!")
```

2. **Register the tool** with spec and handler:

```python
from alfredo.tools.specs import ToolSpec, ToolParameter, ModelFamily
from alfredo.tools.registry import registry

spec = ToolSpec(
    id="my_tool",
    name="my_tool",
    description="What this tool does",
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="param_name",
            required=True,
            instruction="How to use this parameter",
            usage="example value"
        )
    ]
)

registry.register_spec(spec)
registry.register_handler("my_tool", MyToolHandler)
```

3. **Add tests** in `tests/`:

```python
def test_my_tool() -> None:
    agent = Agent()
    text = """
    <my_tool>
    <param_name>value</param_name>
    </my_tool>
    """
    result = agent.execute_from_text(text)
    assert result.success
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_agent.py::test_agent_initialization -v

# Run with output
uv run pytest -v -s
```

## Code Style

- **Type hints**: Required for all functions and methods
- **Docstrings**: Google style for public APIs
- **Formatting**: Enforced by ruff
- **Line length**: 120 characters
- **Imports**: Sorted with isort (via ruff)

## Examples

```bash
# Basic agent usage (XML-based tool execution)
uv run python examples/basic_usage.py

# LangChain integration
uv run python examples/langchain_integration.py

# Agentic scaffold (plan-verify-replan)
uv run python examples/agentic_example.py

# Web fetch examples
uv run python examples/web_fetch_simple.py
```

## Common Tasks

### Run all quality checks
```bash
make check
```

### Format code
```bash
uv run ruff format src tests
```

### Update dependencies
```bash
uv lock --upgrade
```

### Build package
```bash
make build
```
