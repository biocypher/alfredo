# alfredo

[![Release](https://img.shields.io/github/v/release/fcarli/alfredo)](https://img.shields.io/github/v/release/fcarli/alfredo)
[![Build status](https://img.shields.io/github/actions/workflow/status/fcarli/alfredo/main.yml?branch=main)](https://github.com/fcarli/alfredo/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/fcarli/alfredo/branch/main/graph/badge.svg)](https://codecov.io/gh/fcarli/alfredo)
[![Commit activity](https://img.shields.io/github/commit-activity/m/fcarli/alfredo)](https://img.shields.io/github/commit-activity/m/fcarli/alfredo)
[![License](https://img.shields.io/github/license/fcarli/alfredo)](https://img.shields.io/github/license/fcarli/alfredo)

Python harness for AI agents with comprehensive tool execution capabilities.

- **Github repository**: <https://github.com/fcarli/alfredo/>
- **Documentation** <https://fcarli.github.io/alfredo/>

## Features

🤖 **Autonomous Agent** - Agent class with automatic planning, execution, and verification

✨ **Comprehensive Tool System** - 8 built-in tools for file operations, command execution, and workflow control

🔗 **MCP Integration** - Connect to any MCP server for extended capabilities

🔧 **LangChain/LangGraph Compatible** - Seamlessly integrate with LangChain agents and workflows

🎯 **Model Agnostic** - Support for different LLM providers (OpenAI, Anthropic, and more)

📊 **Execution Tracing** - View detailed traces of all agent actions and tool calls

📦 **Easy to Extend** - Simple API for creating custom tools

🔒 **Type Safe** - Full type hints and Pydantic validation

## Quick Start

### Installation

```bash
# Full installation with agentic scaffold
uv add alfredo

# Or install from PyPI
pip install alfredo
```

### Basic Usage - Agent Class (Recommended)

The Agent class provides autonomous task execution with automatic planning and verification:

```python
from alfredo import Agent

# Create an agent
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",  # or "anthropic/claude-3-5-sonnet-20241022"
    verbose=True
)

# Run a task
agent.run("Create a Python script that prints 'Hello, World!'")

# View execution trace
agent.display_trace()

# Access results
print(agent.results["final_answer"])
```

### LangChain Integration

Convert Alfredo tools to LangChain format for custom agent implementations:

```python
from alfredo.integrations.langchain import create_all_langchain_tools
from langchain_anthropic import ChatAnthropic

# Convert Alfredo tools to LangChain format
tools = create_all_langchain_tools(cwd=".")

# Use with any LangChain agent
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)
```

### MCP Integration

Combine Alfredo tools with MCP servers for extended capabilities:

```python
from alfredo import Agent
from alfredo.integrations.mcp import load_combined_tools_sync

# Configure MCP servers
server_configs = {
    "biocontext": {
        "transport": "streamable_http",
        "url": "https://mcp.biocontext.ai/mcp/",
    }
}

# Load tools
tools = load_combined_tools_sync(cwd=".", mcp_server_configs=server_configs)

# Create agent with combined toolset
agent = Agent(cwd=".", tools=tools)
agent.run("Get the interactors of TP53 in human")
```

## Available Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_to_file` | Create or overwrite files |
| `replace_in_file` | Apply diff-based edits |
| `execute_command` | Run shell commands with timeout |
| `list_files` | List directory contents (recursive option) |
| `search_files` | Regex search across files |
| `ask_followup_question` | Request user input |
| `attempt_completion` | Signal task completion |

## Documentation

- [Tools Documentation](TOOLS_README.md) - Complete guide to all tools
- [LangChain Integration](LANGCHAIN_INTEGRATION.md) - Using Alfredo with LangChain/LangGraph
- [Developer Guide](CLAUDE.md) - Contributing and development setup

## Examples

### Agent Class Usage

The Agent class autonomously plans and executes tasks:

```python
from alfredo import Agent

# Create agent
agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=True)

# File operations
agent.run("Read config.yaml and create a summary in summary.txt")

# Command execution
agent.run("Run the test suite and report any failures")

# File discovery
agent.run("Find all Python test files and count how many there are")

# Display execution trace
agent.display_trace()
```

### Functional API

For one-off tasks without creating an agent instance:

```python
from alfredo.agentic.graph import run_agentic_task

result = run_agentic_task(
    task="Create a hello world Python script",
    cwd=".",
    model_name="gpt-4.1-mini",
    verbose=True
)

print(result["final_answer"])
```

### OpenAI Native Integration

Direct OpenAI API integration without LangChain:

```python
from alfredo import OpenAIAgent

agent = OpenAIAgent(cwd=".", model="gpt-4.1-mini")
result = agent.run("Read the file config.json and summarize it")
print(result["final_answer"])
```

## Development

This project uses `uv` as the package manager.

### Setup

```bash
# Clone the repository
git clone https://github.com/fcarli/alfredo.git
cd alfredo

# Install dependencies
make install

# Or manually
uv sync
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_agent.py -v
```

### Code Quality

```bash
# Format code
uv run ruff format src tests

# Lint
uv run ruff check src

# Type check
uv run mypy src
```

## Architecture

Alfredo's tool system is inspired by the [Cline](https://github.com/cline/cline) coding agent and provides:

- **Tool Specifications** - Declarative tool definitions with parameters
- **Tool Registry** - Centralized management of available tools
- **Tool Handlers** - Modular execution logic
- **Prompt Builder** - Automatic system prompt generation
- **LangChain Adapter** - Convert tools to LangChain format

See [TOOLS_README.md](TOOLS_README.md) for detailed architecture documentation.

## License

Released under the [MIT License](LICENSE).

## Credits

Tool system design inspired by [Cline](https://github.com/cline/cline) - an AI coding agent for VSCode.

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
