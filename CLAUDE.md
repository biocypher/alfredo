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

## LangChain Integration

Alfredo tools are fully compatible with LangChain and LangGraph! See `LANGCHAIN_INTEGRATION.md` for complete documentation.

### Quick Start

```bash
# Install with LangChain support
uv add alfredo[langchain]

# Or add langchain-core separately
uv add langchain-core
```

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

### Run Examples

```bash
# Test LangChain integration
uv run pytest tests/test_langchain_integration.py -v

# Run example
uv run python examples/langchain_integration.py
```

## Architecture

### Tools System

The core of Alfredo is a modular tool system that allows AI agents to:

1. **File Operations** (`src/alfredo/tools/handlers/file_ops.py`)
   - Read files
   - Write/create files
   - Apply diff-based edits with SEARCH/REPLACE blocks

2. **Command Execution** (`src/alfredo/tools/handlers/command.py`)
   - Execute shell commands with timeout control
   - Capture stdout/stderr

3. **File Discovery** (`src/alfredo/tools/handlers/discovery.py`)
   - List directory contents (recursive option)
   - Search files with regex patterns

4. **Workflow Control** (`src/alfredo/tools/handlers/workflow.py`)
   - Ask follow-up questions
   - Signal task completion

### Core Components

```
src/alfredo/
├── agent.py              # Main Agent class - orchestrates tool execution
├── tools/
│   ├── specs.py          # Tool specifications (ToolSpec, ToolParameter)
│   ├── registry.py       # Tool registry (singleton pattern)
│   ├── base.py           # Base handler classes
│   └── handlers/         # Tool implementations
│       ├── file_ops.py   # File read/write/edit
│       ├── command.py    # Command execution
│       ├── discovery.py  # List/search files
│       └── workflow.py   # Ask/completion
├── prompts/
│   └── builder.py        # System prompt generation
├── integrations/
│   └── langchain.py      # LangChain/LangGraph integration
```

### Key Design Patterns

1. **Tool Registry**: Singleton pattern for managing tool specifications and handlers
2. **Specification-based**: Tools are defined via `ToolSpec` objects with parameters and descriptions
3. **Handler Pattern**: Each tool has a handler class that implements execution logic
4. **Model Family Variants**: Support for different LLM-specific tool definitions
5. **XML Format**: Standard tool invocation format compatible with Claude and other models

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

See `examples/basic_usage.py` for a complete example of using the Agent class.

Run the example:

```bash
cd examples
uv run python basic_usage.py
```

## Documentation

Full tool documentation is in `TOOLS_README.md`.

Build and serve docs:

```bash
uv run mkdocs serve
```

## Common Tasks

### Run example
```bash
uv run python examples/basic_usage.py
```

### Check code quality
```bash
uv run ruff check src
uv run mypy src
```

### Format code
```bash
uv run ruff format src tests
```

### Update dependencies
```bash
uv lock --upgrade
```
