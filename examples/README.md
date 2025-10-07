# Alfredo Examples

This directory contains examples demonstrating how to use Alfredo tools with various frameworks.

## Available Examples

### 1. `basic_usage.py`
Basic usage of Alfredo's Agent class without any framework integration.

**Run:**
```bash
uv run python examples/basic_usage.py
```

### 2. `langchain_integration.py`
Comprehensive example showing LangChain integration with multiple tools.

**Requirements:**
```bash
uv add langchain-core langchain-anthropic
export ANTHROPIC_API_KEY=your_key_here
```

**Run:**
```bash
uv run python examples/langchain_integration.py
```

### 3. `web_fetch_simple.py` ⭐ NEW
Simple, focused example of using the `web_fetch` tool with a LangChain agent.

**Requirements:**
```bash
uv add langchain-anthropic
export ANTHROPIC_API_KEY=your_key_here
```

**Run:**
```bash
uv run python examples/web_fetch_simple.py
```

**What it does:**
- Fetches content from a URL (e.g., https://python.org)
- Converts HTML to markdown
- Uses Claude to analyze the content
- Returns a natural language summary

### 4. `web_fetch_example.py` ⭐ NEW
Comprehensive example showing multiple use cases for the `web_fetch` tool:
- Direct tool invocation
- Integration with LangChain agents
- LCEL chain usage
- Error handling
- Combining multiple tools

**Requirements:**
```bash
uv add langchain-core langchain-anthropic
export ANTHROPIC_API_KEY=your_key_here  # Optional for some examples
```

**Run:**
```bash
uv run python examples/web_fetch_example.py
```

## Tool-Specific Examples

### web_fetch Tool

The `web_fetch` tool fetches web content and converts it to markdown format.

**Direct usage:**
```python
from alfredo.integrations.langchain import create_langchain_tool

# Create the tool
web_fetch = create_langchain_tool("web_fetch")

# Use it
result = web_fetch.invoke({"url": "https://example.com"})
print(result)
```

**With LangChain agent:**
```python
from langchain_anthropic import ChatAnthropic
from alfredo.integrations.langchain import create_langchain_tool

# Setup
web_fetch = create_langchain_tool("web_fetch")
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools([web_fetch])

# Use in conversation
messages = [{"role": "user", "content": "Fetch https://python.org"}]
response = model_with_tools.invoke(messages)
```

### list_code_definition_names Tool

Parse source code to extract function and class definitions.

**Example:**
```python
from alfredo.integrations.langchain import create_langchain_tool

# Create the tool
code_parser = create_langchain_tool("list_code_definition_names", cwd="/path/to/project")

# Use it
result = code_parser.invoke({"path": "src/"})
print(result)  # Lists all functions, classes, methods
```

## Environment Setup

### With uv (recommended)
```bash
# Install dependencies
uv sync

# Add LangChain support
uv add alfredo[langchain]

# Add specific LangChain packages
uv add langchain-anthropic

# Run examples
uv run python examples/web_fetch_simple.py
```

### With pip
```bash
# Install alfredo with LangChain support
pip install -e ".[langchain]"

# Install LangChain packages
pip install langchain-anthropic

# Run examples
python examples/web_fetch_simple.py
```

## API Keys

Most examples require an API key for the LLM provider:

**Anthropic Claude:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

**OpenAI:**
```bash
export OPENAI_API_KEY=your_key_here
```

## Quick Start

The fastest way to try the new tools:

```bash
# 1. Install dependencies
uv sync
uv add langchain-anthropic

# 2. Set API key
export ANTHROPIC_API_KEY=your_key_here

# 3. Run simple example
uv run python examples/web_fetch_simple.py
```

## Available Tools

All tools are compatible with LangChain:

| Tool | Description | Example Use Case |
|------|-------------|------------------|
| `web_fetch` | Fetch and convert web content | Analyze documentation, scrape content |
| `list_code_definition_names` | Parse code structure | Understand codebase architecture |
| `read_file` | Read file contents | Load configuration files |
| `write_to_file` | Create/overwrite files | Save generated code |
| `replace_in_file` | Apply diffs to files | Make precise edits |
| `list_files` | List directory contents | Explore project structure |
| `search_files` | Search with regex | Find specific patterns |
| `execute_command` | Run shell commands | Run tests, build projects |
| `ask_followup_question` | Request user input | Interactive workflows |
| `attempt_completion` | Signal task completion | End agent loop |

## Need Help?

- Check the main [README.md](../README.md)
- See [LANGCHAIN_INTEGRATION.md](../LANGCHAIN_INTEGRATION.md) for detailed integration docs
- See [TOOLS_README.md](../TOOLS_README.md) for tool documentation

## Contributing Examples

Have a cool example? Submit a PR! We'd love to see:
- Integration with other frameworks (CrewAI, AutoGPT, etc.)
- Real-world use cases
- Tool combinations
- Performance optimizations
