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
from alfredo.integrations.langchain import create_langchain_tools

# Get all tools as LangChain StructuredTools
tools = create_langchain_tools(cwd="/path/to/workspace")

# Use with LangChain agents
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)
```

### Using the Agent Class (Recommended)

```python
from alfredo import Agent

# Create an agent instance
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",
    verbose=True
)

# Run a task
result = agent.run("Create a hello world Python script")

# Display execution trace
agent.display_trace()

# Access results
print(agent.results["final_answer"])
```

### Functional API (Alternative)

You can also use the functional API for one-off tasks:

```python
from alfredo.agentic.graph import run_agentic_task

# Run a task with plan-verify-replan loop
result = run_agentic_task(
    task="Create a hello world Python script",
    cwd=".",
    model_name="gpt-4.1-mini",
    verbose=True,
    recursion_limit=50
)
```

### Examples

```bash
# Run Agent class example (recommended)
uv run python examples/agent_example.py

# Run functional API example
uv run python examples/agentic_example.py

# Run basic LangChain example
uv run python examples/langchain_integration.py

# Test agentic scaffold
uv run pytest tests/test_agentic_graph.py -v
```

## MCP (Model Context Protocol) Integration

Alfredo supports loading and using MCP-compatible tools alongside native Alfredo tools. This allows you to extend the agent's capabilities with any MCP server.

### Installation

```bash
# Add MCP adapters package
uv add langchain-mcp-adapters
```

### Loading MCP Tools

```python
from alfredo.integrations.mcp import load_mcp_tools_sync

# Configure MCP servers
server_configs = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    }
}

# Load MCP tools
mcp_tools = load_mcp_tools_sync(server_configs)
```

### Combining Alfredo + MCP Tools

```python
from alfredo import Agent
from alfredo.integrations.mcp import load_combined_tools_sync

# Load both Alfredo and MCP tools in one call
tools = load_combined_tools_sync(
    cwd=".",
    mcp_server_configs=server_configs
)

# Create agent with combined toolset
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",
    tools=tools,
    verbose=True
)

# Run task
agent.run("Your task here")
agent.display_trace()
```

### Custom Tool Selection

```python
from alfredo import Agent
from alfredo.integrations.langchain import create_langchain_tools
from alfredo.integrations.mcp import load_mcp_tools_sync

# Load only specific Alfredo tools
alfredo_tools = create_langchain_tools(
    cwd=".",
    tool_ids=["read_file", "write_file", "list_files", "attempt_completion"]
)

# Load MCP tools
mcp_tools = load_mcp_tools_sync(server_configs)

# Combine manually
custom_tools = alfredo_tools + mcp_tools

# Create agent with custom toolset
agent = Agent(cwd=".", tools=custom_tools)
agent.run("Your task here")
```

### Async Usage

```python
import asyncio
from alfredo import Agent
from alfredo.integrations.mcp import load_combined_tools

async def main():
    tools = await load_combined_tools(
        cwd=".",
        mcp_server_configs=server_configs
    )

    # Create agent with loaded tools
    agent = Agent(cwd=".", tools=tools)
    agent.run("Your task here")

asyncio.run(main())
```

### Popular MCP Servers

Install MCP servers via npm:

```bash
# Filesystem access
npm install -g @modelcontextprotocol/server-filesystem

# GitHub integration
npm install -g @modelcontextprotocol/server-github

# Or use npx (no installation needed)
# Just use "npx" as the command in server_configs
```

Server configuration format:

**Local MCP Server (stdio):**
```python
server_configs = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
        "transport": "stdio"
    }
}
```

**Remote MCP Server (HTTP/SSE):**
```python
server_configs = {
    "remote_api": {
        "transport": "streamable_http",  # or "sse"
        "url": "https://api.example.com/mcp",
        "headers": {  # Optional authentication
            "Authorization": "Bearer your-token-here",
            "X-API-Key": "your-api-key"
        }
    }
}
```

**Mixed Local + Remote:**
```python
server_configs = {
    "local_fs": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    },
    "remote_api": {
        "transport": "streamable_http",
        "url": "https://api.example.com/mcp",
        "headers": {"Authorization": "Bearer token"}
    }
}
```

### Examples

```bash
# Run local MCP integration example
uv run python examples/mcp_example.py

# Run remote MCP server example
uv run python examples/mcp_remote_example.py
```

For more MCP servers: https://github.com/modelcontextprotocol/servers

## Architecture

Alfredo provides a layered architecture with multiple usage patterns to fit different use cases.

## Usage Patterns

### Primary Usage: Agent Class (Recommended)

**For most users, the `Agent` class is the recommended approach.** It provides:
- Automatic planning and replanning (optional)
- Verification of task completion
- Sophisticated tool orchestration
- Support for any LLM provider (OpenAI, Anthropic, etc.)
- Execution trace display
- Clean, object-oriented interface

```python
from alfredo import Agent

# Create an agent with planning (default)
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",
    verbose=True
)

# Run tasks
agent.run("Create a hello world Python script")

# View execution trace
agent.display_trace()

# Access results
print(agent.results["final_answer"])
```

#### Planning Options

By default, the Agent uses a **plan-verify-replan** loop:
1. **Planner** creates an implementation plan
2. **Agent** executes the plan using tools
3. **Verifier** checks if task is complete
4. **Replan** creates improved plan if verification fails

You can disable planning to start execution directly:

```python
# Create an agent without planning
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",
    enable_planning=False,  # Skip planner node
    verbose=True
)

# Agent starts directly with ReAct loop
agent.run("Create a hello world Python script")
```

**When to disable planning:**
- Simple, straightforward tasks
- When you want faster startup (no planning overhead)
- When you prefer the agent to explore freely without a predefined plan
- For interactive tasks where planning may be overly rigid

**Graph flow:**
- With planning (default): `START → planner → agent ⇄ tools → verifier → replan → agent`
- Without planning: `START → agent ⇄ tools → verifier → END`

The Agent class uses LangChain to automatically convert Alfredo tools to the native format of any LLM provider (OpenAI's function calling, Anthropic's tool use, etc.).

### Customizing System Prompts

You can customize the system prompts for any node in the agentic scaffold (planner, agent, verifier, replan). Alfredo supports two strategies for customizing prompts:

#### Strategy 1: Plain Text (Auto-wrapping)

Provide plain text without placeholders. The system automatically:
- **Prepends** dynamic variables (task, plan, etc.) at the beginning
- **Appends** tool-specific instructions at the end
- Keeps your custom content in the middle

```python
from alfredo import Agent

agent = Agent(cwd=".", model_name="gpt-4.1-mini")

# Set custom planner prompt (plain text)
agent.set_planner_prompt("""
Your job is to create a detailed implementation plan.

## Format Requirements
1. Start with a brief overview (2-3 sentences)
2. Break down into numbered sequential steps
3. Include rationale for each step
4. Define clear success criteria

Be thorough and specific in your planning.
""")

# The system automatically adds:
# - Task context at the beginning
# - Tool instructions at the end

# Run with custom prompt
agent.run("Create a Python REST API")
```

#### Strategy 2: Explicit Placeholders (Validation)

Provide a template with `{placeholder}` variables. The system:
- **Validates** that all required placeholders are present
- **Formats** the template with actual values
- Raises `ValueError` if any required placeholders are missing

```python
from alfredo import Agent

agent = Agent(cwd=".", model_name="gpt-4.1-mini")

# Set custom agent prompt with explicit placeholders
agent.set_agent_prompt("""
# Original Task
{task}

# Current Plan
{plan}

# Your Mission
Execute the plan step by step. After each action:
1. Reflect on the result
2. Check if you're following the plan
3. Adjust if necessary

Be methodical and careful.

{tool_instructions}

**CRITICAL**: Call attempt_completion when done!
""")

agent.run("Build a calculator")
```

#### Required Placeholders by Node

Each node has specific required variables:

| Node | Required Placeholders | Description |
|------|----------------------|-------------|
| **planner** | `{task}`, `{tool_instructions}` | Creates implementation plan |
| **agent** | `{task}`, `{plan}`, `{tool_instructions}` | Executes the plan |
| **verifier** | `{task}`, `{answer}`, `{trace_section}`, `{tool_instructions}` | Verifies completion |
| **replan** | `{task}`, `{previous_plan}`, `{verification_feedback}`, `{tool_instructions}` | Creates improved plan |

**Important**: `{tool_instructions}` is **always required** and should be included in your template (or auto-appended in plain text mode).

#### Additional Examples

**Customize verifier to be more strict:**
```python
agent.set_verifier_prompt("""
You are a strict verification agent.

Task: {task}
Proposed Answer: {answer}

{trace_section}

## Verification Checklist
✓ Task fully completed?
✓ All requirements met?
✓ Evidence in trace?
✓ No errors or failures?

Be highly critical. Only approve if ALL criteria met.

{tool_instructions}

Output: VERIFIED: [reason] or NOT_VERIFIED: [reason]
""")
```

**Customize replan to focus on efficiency:**
```python
agent.set_replan_prompt("""
Task: {task}

Previous attempt: {previous_plan}

Why it failed: {verification_feedback}

Create a NEW, MORE EFFICIENT plan that:
- Addresses the feedback directly
- Uses fewer steps if possible
- Has clearer success criteria

{tool_instructions}
""")
```

#### Managing Custom Prompts

```python
# Get current template for a node
template = agent.get_prompt_template("planner")
if template:
    print(f"Using custom planner template: {template}")

# Reset all prompts to defaults
agent.reset_prompts()

# Reset specific node by setting it again
agent.set_planner_prompt(None)  # Not supported - use reset_prompts() instead

# Set multiple prompts
agent.set_planner_prompt("Create a concise plan.")
agent.set_agent_prompt("Work carefully and methodically.")
agent.set_verifier_prompt("Verify thoroughly.")
```

#### Best Practices

1. **Plain text for simplicity**: If you don't need precise control over variable placement, use plain text mode
2. **Placeholders for full control**: Use explicit placeholders when you need exact formatting
3. **Always include tool_instructions**: This ensures tools with custom instructions (like todo lists) work properly
4. **Test your templates**: Run `agent.get_system_prompts()` to preview prompts before running tasks
5. **Keep it focused**: Shorter, clearer prompts often work better than verbose ones

### Alternative Usage Patterns

For specialized use cases or direct control, Alfredo provides alternative execution modes:

**1. Functional API** - For one-off tasks without creating an agent instance:

```python
from alfredo.agentic.graph import run_agentic_task

result = run_agentic_task(
    task="Create a hello world Python script",
    cwd=".",
    model_name="gpt-4.1-mini",
    verbose=True
)
```

**2. Native OpenAI Agent** - Direct OpenAI API integration without LangChain:

```python
from alfredo import OpenAIAgent

agent = OpenAIAgent(cwd=".", model="gpt-4.1-mini")
result = agent.run("Read the file config.json")
```

Use this when:
- You want direct OpenAI API control without LangChain
- You're building custom tool calling loops
- You need minimal dependencies

## Pre-built Agents

Alfredo provides pre-built specialized agents for common tasks. These agents wrap the base `Agent` class with custom prompts and configurations optimized for specific use cases.

### ExplorationAgent

The `ExplorationAgent` explores directories and generates comprehensive markdown reports about their contents. It features:

- **Smart file size handling** - Automatically adjusts reading strategy based on file size
- **Data file analysis** - Generates pandas analysis for CSV, Excel, Parquet, HDF5, JSON files
- **Categorization** - Groups files by type (code, data, config, docs, binary)
- **Context steering** - Use prompts to focus exploration on specific aspects
- **Configurable thresholds** - Customize size limits and preview line counts

#### Basic Usage

```python
from alfredo import ExplorationAgent

# Basic exploration
agent = ExplorationAgent(
    cwd="./my_project",
    output_path="./reports/project_overview.md"
)
report = agent.explore()
print(report)
```

#### With Context Steering

```python
# Focus exploration with context prompt
agent = ExplorationAgent(
    cwd="./data_pipeline",
    context_prompt=(
        "Focus on data schemas, transformations, and quality checks. "
        "Note any data validation logic and preprocessing steps."
    ),
    model_name="gpt-4.1-mini"
)
report = agent.explore()
```

#### Advanced Configuration

```python
agent = ExplorationAgent(
    cwd="./large_project",
    context_prompt="Focus on API endpoints and authentication mechanisms",
    max_file_size_bytes=50_000,  # More aggressive size limit (50KB)
    preview_kb=25,  # Preview first 25KB for large files
    output_path="./reports/api_exploration.md",
    model_name="gpt-4o-mini",
    verbose=True
)
report = agent.explore()

# View execution trace
agent.display_trace()

# Alternative: Use line-based preview instead of KB
agent2 = ExplorationAgent(
    cwd="./code_project",
    preview_lines=100,  # Preview first 100 lines for large files
    verbose=True
)
report2 = agent2.explore()
```

#### How It Works

The ExplorationAgent:

1. **Lists files recursively** to understand directory structure
2. **Categorizes files** by type (source code, data, config, docs, binary)
3. **Smart reading**:
   - Small files (<10KB): Reads fully
   - Medium files (10KB-100KB): Reads with `limit` parameter (first 100 lines)
   - Large files (>100KB): Peeks at first 50 lines
   - Very large files (>1MB): Just notes metadata
4. **Data analysis**: For CSV, Excel, Parquet, HDF5, JSON files:
   - Writes Python analysis script with pandas
   - Executes script using `execute_command` tool
   - Captures shape, columns, dtypes, head(), describe(), missing values
5. **Generates markdown report** with structure:
   - Overview (file counts, sizes)
   - Directory structure
   - Files by category
   - Data analysis results
   - Summary insights

#### Supported Data Formats

- **CSV/TSV** - Analyzed with `pandas.read_csv()`
- **Excel** (.xlsx, .xls) - Analyzed with `pandas.read_excel()`
- **Parquet** - Analyzed with `pandas.read_parquet()`
- **HDF5** (.h5, .hdf5) - Analyzed with `pandas.read_hdf()`
- **JSON/JSONL** - Analyzed with `pandas.read_json()`
- **Feather** - Analyzed with `pandas.read_feather()`

#### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cwd` | `"."` | Directory to explore |
| `context_prompt` | `None` | Optional context to steer exploration focus |
| `model_name` | `"gpt-4.1-mini"` | Model to use for exploration |
| `max_file_size_bytes` | `100_000` | Threshold for limited reading (100KB) |
| `preview_kb` | `50`* | Size in KB to preview for large files |
| `preview_lines` | `None` | Number of lines to preview (alternative to preview_kb) |
| `output_path` | `None` | Report save path (default: `notes/exploration_report.md`) |
| `verbose` | `True` | Print progress updates |

*Default is 50KB when neither `preview_kb` nor `preview_lines` is specified

#### Example Run

```bash
# Run the exploration example
uv run python examples/prebuilt_explore_example.py
```

#### Dependencies

The ExplorationAgent requires:
- `alfredo[agentic]` - LangGraph and agentic scaffold
- `pandas` - For data file analysis (recommended)
- `openpyxl` - For Excel file support (optional)
- `pyarrow` - For Parquet file support (optional)
- `h5py` or `tables` - For HDF5 file support (optional)

```bash
# Install with data analysis support
uv add alfredo[agentic] pandas openpyxl pyarrow h5py
```

## Core Architecture

### Tools System

The core tool system allows AI agents to interact with the environment through a modular handler architecture:

**Tool Categories:**
- **File Operations** (`file_ops.py`) - Read, write, diff-based edits
  - `read_file` - Read files with optional `offset` and `limit` parameters for partial reading
  - `write_to_file` - Create or overwrite files
  - `replace_in_file` - Apply search/replace diffs
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

# Pre-built exploration agent
uv run python examples/prebuilt_explore_example.py

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
