# Alfredo Tools System

This document describes the tool system implemented in Alfredo, ported from the Cline coding agent.

## Overview

Alfredo now includes a comprehensive tool system that allows AI agents to:
- **Read, write, and edit files** with diff-based edits
- **Execute shell commands** with timeout control
- **Discover files** through listing and regex search
- **Control workflow** by asking questions and signaling completion
- **Generate prompts** automatically with tool definitions

## Architecture

The tool system consists of four main components:

### 1. Tool Specifications (`alfredo/tools/specs.py`)

Defines metadata for tools:
- **ToolParameter**: Parameter definition with name, type, instructions, and usage examples
- **ToolSpec**: Complete tool specification including description, parameters, and model family variants
- **ModelFamily**: Enum for different LLM families (GENERIC, ANTHROPIC, OPENAI, GEMINI)

### 2. Tool Registry (`alfredo/tools/registry.py`)

Manages tool registration and retrieval:
- Singleton pattern for global tool access
- Model family-based tool variants
- Handler class registration
- Thread-safe operations

### 3. Tool Handlers (`alfredo/tools/handlers/`)

Implement actual tool execution:
- **Base classes**: `BaseToolHandler` and `AsyncToolHandler` for synchronous and async operations
- **File operations**: Read, write, and diff-based editing
- **Command execution**: Shell command execution with timeout
- **File discovery**: List directories and search file contents
- **Workflow control**: Ask questions and signal completion

### 4. Agent & Prompt Builder (`alfredo/agent.py`, `alfredo/prompts/builder.py`)

Orchestrates tool usage:
- **Agent**: Main interface for tool execution
- **PromptBuilder**: Generates system prompts with tool definitions
- **Tool parsing**: Extracts tool invocations from model output

## Available Tools

### File Operations

#### `read_file`
Read the contents of a file.

**Parameters:**
- `path` (required): File path relative to working directory

**Example:**
```xml
<read_file>
<path>src/main.py</path>
</read_file>
```

#### `write_to_file`
Create or overwrite a file with new content.

**Parameters:**
- `path` (required): File path relative to working directory
- `content` (required): Complete file content

**Example:**
```xml
<write_to_file>
<path>output.txt</path>
<content>
File content here.
Can span multiple lines.
</content>
</write_to_file>
```

#### `replace_in_file`
Edit a file using SEARCH/REPLACE blocks.

**Parameters:**
- `path` (required): File path to edit
- `diff` (required): SEARCH/REPLACE blocks

**Example:**
```xml
<replace_in_file>
<path>config.py</path>
<diff>
------- SEARCH
DEBUG = False
=======
DEBUG = True
+++++++ REPLACE
</diff>
</replace_in_file>
```

### Command Execution

#### `execute_command`
Run shell commands.

**Parameters:**
- `command` (required): Shell command to execute
- `timeout` (optional): Timeout in seconds (default: 120)

**Example:**
```xml
<execute_command>
<command>ls -la</command>
<timeout>60</timeout>
</execute_command>
```

### File Discovery

#### `list_files`
List directory contents.

**Parameters:**
- `path` (required): Directory path
- `recursive` (optional): "true" for recursive listing

**Example:**
```xml
<list_files>
<path>src/</path>
<recursive>true</recursive>
</list_files>
```

#### `search_files`
Search for regex patterns in files.

**Parameters:**
- `path` (required): Directory to search
- `regex` (required): Regular expression pattern
- `file_pattern` (optional): Glob pattern for file filtering (e.g., "*.py")

**Example:**
```xml
<search_files>
<path>.</path>
<regex>def hello</regex>
<file_pattern>*.py</file_pattern>
</search_files>
```

### Workflow Control

#### `ask_followup_question`
Request additional information from the user.

**Parameters:**
- `question` (required): Question to ask

**Example:**
```xml
<ask_followup_question>
<question>What should be the output format?</question>
</ask_followup_question>
```

#### `attempt_completion`
Signal that the task is complete.

**Parameters:**
- `result` (optional): Summary of what was accomplished
- `command` (optional): Final command that was executed

**Example:**
```xml
<attempt_completion>
<result>Created 3 files and ran all tests successfully.</result>
</attempt_completion>
```

## Usage

### Basic Agent Setup

```python
from alfredo import Agent
from pathlib import Path

# Create an agent
agent = Agent(cwd=str(Path.cwd()))

# Get system prompt for your LLM
system_prompt = agent.get_system_prompt(include_examples=True)
print(system_prompt)

# List available tools
tools = agent.get_available_tools()
print(f"Available tools: {tools}")
```

### Executing Tools

```python
# Simulate model output with a tool invocation
model_output = """
I'll read the configuration file.

<read_file>
<path>config.yaml</path>
</read_file>
"""

# Parse and execute
result = agent.execute_from_text(model_output)

if result:
    if result.success:
        print(f"Success! Output:\n{result.output}")
    else:
        print(f"Error: {result.error}")
```

### Creating Custom Tools

```python
from alfredo.tools.base import BaseToolHandler, ToolResult
from alfredo.tools.specs import ToolSpec, ToolParameter, ModelFamily
from alfredo.tools.registry import registry
from typing import Any

class MyCustomHandler(BaseToolHandler):
    @property
    def tool_id(self) -> str:
        return "my_custom_tool"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        # Your implementation here
        return ToolResult.ok("Custom tool executed!")

# Register the tool
spec = ToolSpec(
    id="my_custom_tool",
    name="my_custom_tool",
    description="Does something custom",
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="input",
            required=True,
            instruction="Input value",
            usage="value here"
        )
    ]
)

registry.register_spec(spec)
registry.register_handler("my_custom_tool", MyCustomHandler)
```

## Design Patterns from Cline

This implementation preserves several key patterns from Cline:

1. **Tool Specifications**: Separate metadata from implementation
2. **Model Family Variants**: Different tool definitions for different LLMs
3. **Registry Pattern**: Centralized tool management
4. **XML Format**: Standard tool invocation format compatible with Claude and other models
5. **Context Requirements**: Conditional tool availability based on runtime context
6. **Path Resolution**: Safe path handling with working directory context

## LangChain/LangGraph Integration

Alfredo tools are fully compatible with LangChain and LangGraph! See [LANGCHAIN_INTEGRATION.md](LANGCHAIN_INTEGRATION.md) for details.

### Quick Example

```python
from alfredo.integrations.langchain import create_all_langchain_tools

# Convert all Alfredo tools to LangChain format
tools = create_all_langchain_tools()

# Use with any LangChain agent or LangGraph workflow
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)
```

## Testing

Run the test suite:

```bash
cd alfredo
uv run pytest tests/test_agent.py -v
uv run pytest tests/test_langchain_integration.py -v
```

Run the examples:

```bash
cd alfredo
uv run python examples/basic_usage.py
uv run python examples/langchain_integration.py
```

## Future Enhancements

Potential additions based on Cline's full feature set:

- **Browser automation** (via Selenium/Playwright)
- **Web fetching** with content extraction
- **MCP (Model Context Protocol)** integration
- **AST-based code analysis** (list_code_definition_names)
- **Async tool handlers** for long-running operations
- **Approval workflows** for dangerous operations
- **Tool result streaming** for real-time feedback

## Comparison with Cline

| Feature | Cline (TypeScript) | Alfredo (Python) | Status |
|---------|-------------------|------------------|--------|
| File operations | ✅ | ✅ | Complete |
| Command execution | ✅ | ✅ | Complete |
| File discovery | ✅ | ✅ | Complete |
| Search files | ✅ | ✅ | Complete |
| Workflow control | ✅ | ✅ | Complete |
| Prompt generation | ✅ | ✅ | Complete |
| Tool registry | ✅ | ✅ | Complete |
| Model variants | ✅ | ✅ | Complete |
| Browser automation | ✅ | ⏳ | Planned |
| MCP integration | ✅ | ⏳ | Planned |
| Approval workflows | ✅ | ⏳ | Planned |
| Diff view | ✅ | ⚠️ | Partial (text-based) |

## License

Same as Alfredo project.
