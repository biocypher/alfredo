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

✨ **Comprehensive Tool System** - 8 built-in tools for file operations, command execution, and workflow control

🔧 **LangChain/LangGraph Compatible** - Seamlessly integrate with LangChain agents and workflows

🎯 **Model Agnostic** - Support for different LLM families (Anthropic, OpenAI, Gemini)

📦 **Easy to Extend** - Simple API for creating custom tools

🔒 **Type Safe** - Full type hints and Pydantic validation

## Quick Start

### Installation

```bash
# Basic installation
uv add alfredo

# With LangChain support
uv add alfredo[langchain]
```

### Basic Usage

```python
from alfredo import Agent

# Create an agent
agent = Agent()

# Get system prompt with tool definitions
prompt = agent.get_system_prompt(include_examples=True)

# Parse and execute tools from model output
model_output = """
<read_file>
<path>config.yaml</path>
</read_file>
"""

result = agent.execute_from_text(model_output)
if result.success:
    print(result.output)
```

### LangChain Integration

```python
from alfredo.integrations.langchain import create_all_langchain_tools
from langchain_anthropic import ChatAnthropic

# Convert Alfredo tools to LangChain format
tools = create_all_langchain_tools()

# Use with any LangChain agent
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)
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

### File Operations

```python
from alfredo import Agent

agent = Agent(cwd="/path/to/project")

# Read a file
result = agent.execute_from_text("""
<read_file>
<path>src/main.py</path>
</read_file>
""")

# Write a file
result = agent.execute_from_text("""
<write_to_file>
<path>output.txt</path>
<content>Hello, World!</content>
</write_to_file>
""")
```

### Command Execution

```python
# Execute commands
result = agent.execute_from_text("""
<execute_command>
<command>pytest tests/</command>
<timeout>60</timeout>
</execute_command>
""")
```

### File Discovery

```python
# Search for patterns
result = agent.execute_from_text("""
<search_files>
<path>.</path>
<regex>def test_.*</regex>
<file_pattern>*.py</file_pattern>
</search_files>
""")
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
