# Alfredo

[![Release](https://img.shields.io/github/v/release/biocypher/alfredo)](https://img.shields.io/github/v/release/biocypher/alfredo)
[![Build status](https://img.shields.io/github/actions/workflow/status/biocypher/alfredo/main.yml?branch=main)](https://github.com/biocypher/alfredo/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/biocypher/alfredo/branch/main/graph/badge.svg)](https://codecov.io/gh/biocypher/alfredo)
[![License](https://img.shields.io/github/license/biocypher/alfredo)](https://img.shields.io/github/license/biocypher/alfredo)

**A Python harness for building AI agents with comprehensive tool execution and autonomous task completion.**

Alfredo provides a LangGraph-based agentic scaffold that combines planning, execution, and verification into a cohesive agent framework. Built with extensibility in mind, it supports custom tools, MCP integration, and comes with specialized prebuilt agents.

## Key Features

- 🤖 **Autonomous Agent** - Plan-verify-replan loop with automatic task decomposition
- 🔧 **11 Built-in Tools** - File ops, commands, discovery, code analysis, vision, and workflow control
- 👁️ **Vision Capabilities** - Analyze images with multimodal models
- 🔗 **MCP Integration** - Connect to any Model Context Protocol server
- 🎯 **Model Agnostic** - OpenAI, Anthropic, or any LangChain-supported LLM
- 📊 **Execution Tracing** - Detailed visibility into agent actions
- 🛠️ **Custom System Prompts** - Fine-tune node behavior with AlfredoTool
- 📦 **Prebuilt Agents** - ExplorationAgent and ReflexionAgent ready to use

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

## Vision-Enabled Agents

Alfredo agents can analyze images using vision models, enabling powerful workflows where agents can verify their own visual outputs:

```python
from alfredo import Agent

# Create an agent with vision capabilities
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",       # Main model for planning and reasoning
    vision_model="gpt-4.1-mini",     # Vision model for image analysis
    parse_reasoning=True
)

# Agent creates visualization AND verifies it by looking at it
agent.run("""
Create a Python script that computes an approximation of pi using the Monte Carlo method.
Also create a visualization of the results and save it as a PNG file.
Make sure that the plot is correct by analyzing the image.
If you miss some package, use uv to initialize a venv and then add what is missing.
""")

# View execution trace to see vision tool in action
agent.display_trace()
```

**What happens:**
1. Agent writes the Monte Carlo simulation code
2. Agent runs the code and generates a plot
3. Agent uses `analyze_image` to verify the visualization is correct
4. Agent iterates if issues are found

**Use cases for vision:**
- 📸 Screenshot analysis for UI testing
- 📊 Chart and diagram verification
- 📝 OCR and document processing
- 🎨 Image description and captioning
- ✅ Visual quality assurance

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

Alfredo includes 11 built-in tools organized by category:

| Category | Tools |
|----------|-------|
| **File Operations** | `read_file`, `write_to_file`, `replace_in_file` |
| **Discovery** | `list_files`, `search_files` |
| **Code Analysis** | `list_code_definition_names` |
| **Commands** | `execute_command` |
| **Vision** | `analyze_image` |
| **Workflow** | `ask_followup_question`, `attempt_completion` |

**[Read More: Tools Documentation →](https://github.com/biocypher/alfredo/blob/main/docs/tools.md)**

## MCP Integration

Alfredo supports two modes of MCP integration:

### CodeAct Mode

Generate importable Python modules from MCP servers, allowing agents to use tools as regular functions instead of through ReAct loops:

```python
from alfredo import Agent

# Configure remote MCP server (supports both local and remote)
agent = Agent(
    cwd="./workspace",
    model_name="gpt-4.1-mini",
    codeact_mcp_functions={
        "biocontext": {
            "url": "https://mcp.biocontext.ai/mcp/",  # Remote or local server
        }
    },
    verbose=True
)

# Agent can now import and use MCP tools in scripts:
agent.run("Write a script that gets the interactors of gene ENSG00000141510 and save to interactors.txt")

# Agent generates code like:
# from biocontext_mcp import bc_get_protein_interactors
# result = bc_get_protein_interactors(gene_id="ENSG00000141510")
```

**Features:**
- ✅ Works with **remote** servers (e.g., `https://mcp.biocontext.ai/mcp/`)
- ✅ Works with **local** servers (e.g., `http://localhost:8000`)
- ✅ Auto-generates typed Python wrapper modules
- ✅ Supports SSE and JSON-RPC 2.0 protocols
- ✅ Session management with automatic retry
- ✅ Script-based tool chaining

### ReAct Mode

Use MCP tools directly through LangChain's tool calling:

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
