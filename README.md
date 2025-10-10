# Alfredo

[![Release](https://img.shields.io/github/v/release/biocypher/alfredo)](https://img.shields.io/github/v/release/biocypher/alfredo)
[![Build status](https://img.shields.io/github/actions/workflow/status/biocypher/alfredo/main.yml?branch=main)](https://github.com/biocypher/alfredo/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/biocypher/alfredo/branch/main/graph/badge.svg)](https://codecov.io/gh/biocypher/alfredo)
[![License](https://img.shields.io/github/license/biocypher/alfredo)](https://img.shields.io/github/license/biocypher/alfredo)

**A Python harness for building AI agents with comprehensive tool execution and autonomous task completion.**

Alfredo provides a LangGraph-based agentic scaffold that combines planning, execution, and verification into a cohesive agent framework. Built with extensibility in mind, it supports custom tools, MCP integration, and comes with specialized prebuilt agents.

## Key Features

🤖 **Autonomous Agent** - Plan-verify-replan loop with automatic task decomposition
🔧 **10 Built-in Tools** - File ops, commands, discovery, code analysis, and workflow control
🔗 **MCP Integration** - Connect to any Model Context Protocol server
🎯 **Model Agnostic** - OpenAI, Anthropic, or any LangChain-supported LLM
📊 **Execution Tracing** - Detailed visibility into agent actions
🛠️ **Custom System Prompts** - Fine-tune node behavior with AlfredoTool
📦 **Prebuilt Agents** - ExplorationAgent and ReflexionAgent ready to use

## Installation

**Note:** Alfredo is not yet available on PyPI. Install directly from GitHub using `uv`:

```bash
# Clone the repository
git clone https://github.com/biocypher/alfredo.git
cd alfredo

# Install dependencies
uv sync

# Or add as a dependency to your project
uv add git+https://github.com/biocypher/alfredo.git
```

## Quick Start

```python
from alfredo import Agent

# Create an agent
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",  # or "anthropic/claude-3-5-sonnet-20241022"
    verbose=True
)

# Run a task - agent will plan, execute, and verify
agent.run("Create a Python script that computes an approximation of pi using the Monte Carlo method")

# View execution trace
agent.display_trace()

# Access results
print(agent.results["final_answer"])
```

## Architecture

Alfredo uses a **LangGraph state graph** with the following nodes:

```
START → planner → agent ⇄ tools → verifier
                   ↑              ↓
                   └── replan ←───┘
```

- **planner**: Creates implementation plan
- **agent**: Performs ReAct-style reasoning and tool calls
- **tools**: Executes tool calls
- **verifier**: Checks if task is complete
- **replan**: Generates improved plan if verification fails

Planning can be disabled to start execution directly at the agent node.

**[Read More: Agent Architecture →](https://github.com/biocypher/alfredo/blob/main/docs/agent-architecture.md)**

## Available Tools

Alfredo includes 10 built-in tools organized by category:

| Category | Tools |
|----------|-------|
| **File Operations** | `read_file`, `write_to_file`, `replace_in_file` |
| **Discovery** | `list_files`, `search_files` |
| **Code Analysis** | `list_code_definition_names` |
| **Commands** | `execute_command` |
| **Workflow** | `ask_followup_question`, `attempt_completion` |

**[Read More: Tools Documentation →](https://github.com/biocypher/alfredo/blob/main/docs/tools.md)**

## MCP Integration

Extend Alfredo with any MCP-compatible server:

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

# Load Alfredo + MCP tools
tools = load_combined_tools_sync(cwd=".", mcp_server_configs=server_configs)

# Create agent with combined toolset
agent = Agent(cwd=".", tools=tools)
agent.run("Get the interactors of TP53 in human and save the results to a file called tp53_interactors.txt")
```

**[Read More: MCP Integration →](https://github.com/biocypher/alfredo/blob/main/docs/mcp-integration.md)**

## Customizing System Prompts

Use `AlfredoTool` to add node-specific instructions to any tool:

```python
from alfredo.tools.alfredo_tool import AlfredoTool

# Add instructions that only appear in specific nodes
tool = AlfredoTool.from_alfredo(
    tool_id="write_todo_list",
    cwd=".",
    system_instructions={
        "planner": "After making your plan, create an initial checklist to keep track of your progress",
    }
)

# Instructions are dynamically injected into node system prompts
agent = Agent(cwd=".", tools=[tool])
```

**[Read More: AlfredoTool & System Prompts →](https://github.com/biocypher/alfredo/blob/main/docs/alfredo-tools.md)**

## Prebuilt Agents

### ExplorationAgent

Explore directories and generate comprehensive markdown reports with smart file reading and data analysis:

```python
from alfredo.prebuilt import ExplorationAgent

agent = ExplorationAgent(
    cwd="./my_project",
    context_prompt="Data belongs to a transcriptomic study on cancer cell lines"
)
report = agent.explore()
```

### ReflexionAgent

Research agent with iterative self-critique and revision using web search:

```python
from alfredo.prebuilt import ReflexionAgent

agent = ReflexionAgent(model_name="gpt-4.1-mini", max_iterations=2)
answer = agent.research("Create a detailed report about the roles of TP53 in cancer cell lines and save the results to a file called tp53_roles.md")
agent.display_trace()
```

**[Read More: Prebuilt Agents →](https://github.com/biocypher/alfredo/blob/main/docs/prebuilt-agents.md)**

## Documentation

- **[Agent Architecture](https://github.com/biocypher/alfredo/blob/main/docs/agent-architecture.md)** - Deep dive into the LangGraph scaffold
- **[Tools](https://github.com/biocypher/alfredo/blob/main/docs/tools.md)** - Complete tool reference and creating custom tools
- **[MCP Integration](https://github.com/biocypher/alfredo/blob/main/docs/mcp-integration.md)** - Using Model Context Protocol servers
- **[AlfredoTool](https://github.com/biocypher/alfredo/blob/main/docs/alfredo-tools.md)** - Customizing system prompts per node
- **[Prebuilt Agents](https://github.com/biocypher/alfredo/blob/main/docs/prebuilt-agents.md)** - ExplorationAgent and ReflexionAgent

## Development

```bash
# Install dependencies
make install

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src

# Lint and type check
uv run ruff check src
uv run mypy src
```

## License

Released under the [MIT License](LICENSE).

## Credits

Tool system design inspired by [Cline](https://github.com/cline/cline) - an AI coding agent for VSCode.

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
