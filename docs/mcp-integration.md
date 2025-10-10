# MCP Integration

Alfredo supports the **Model Context Protocol (MCP)**, allowing you to extend your agent's capabilities with any MCP-compatible server. This enables integration with external services, databases, APIs, and specialized tools.

## What is MCP?

The **Model Context Protocol** is an open standard that allows AI agents to connect to external data sources and tools through a unified interface. MCP servers expose tools, resources, and prompts that agents can use.

**Benefits of MCP**:
- **Extensibility**: Add new capabilities without modifying alfredo
- **Reusability**: Use existing MCP servers from the community
- **Standardization**: Common protocol for tool integration
- **Isolation**: Tools run in separate processes for safety

**Learn more**: [Model Context Protocol](https://modelcontextprotocol.io)

## Installation

MCP integration requires the `langchain-mcp-adapters` package:

```bash
# Add to your project
uv add langchain-mcp-adapters

# Or install in alfredo development environment
cd alfredo
uv add langchain-mcp-adapters
```

## Server Configuration

MCP servers can be **local** (stdio) or **remote** (HTTP/SSE).

### Local Servers (stdio)

Local servers run as subprocess that communicate via standard input/output:

```python
server_configs = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    }
}
```

**Configuration Fields**:
- `command`: The executable to run (e.g., "npx", "python", "node")
- `args`: List of command-line arguments
- `transport`: Must be `"stdio"` for local servers

### Remote Servers (HTTP/SSE)

Remote servers expose tools via HTTP endpoints:

```python
# HTTP with streaming (streamable_http)
server_configs = {
    "biocontext": {
        "transport": "streamable_http",
        "url": "https://mcp.biocontext.ai/mcp/",
        "headers": {  # Optional
            "Authorization": "Bearer your-token",
            "X-API-Key": "your-api-key"
        }
    }
}

# Server-Sent Events (SSE)
server_configs = {
    "weather": {
        "transport": "sse",
        "url": "https://weather.api.com/mcp",
        "headers": {
            "X-API-Key": "your-key"
        }
    }
}
```

**Configuration Fields**:
- `transport`: `"streamable_http"` or `"sse"`
- `url`: The server endpoint URL
- `headers`: Optional HTTP headers for authentication

### Mixing Local and Remote

You can use both types in the same configuration:

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

## Loading MCP Tools

### Synchronous Loading (Recommended)

Use `load_mcp_tools_sync()` for simple use cases:

```python
from alfredo.integrations.mcp import load_mcp_tools_sync

server_configs = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    }
}

# Load MCP tools
mcp_tools = load_mcp_tools_sync(server_configs)

print(f"Loaded {len(mcp_tools)} MCP tools")
for tool in mcp_tools:
    print(f"  - {tool.name}: {tool.description}")
```

### Asynchronous Loading

Use `load_mcp_tools()` when already in async context:

```python
import asyncio
from alfredo.integrations.mcp import load_mcp_tools

async def main():
    mcp_tools = await load_mcp_tools(server_configs)
    print(f"Loaded {len(mcp_tools)} tools")

asyncio.run(main())
```

### Combining with Alfredo Tools

Use `load_combined_tools_sync()` to get both Alfredo and MCP tools:

```python
from alfredo.integrations.mcp import load_combined_tools_sync

# Load Alfredo + MCP tools in one call
tools = load_combined_tools_sync(
    cwd=".",
    mcp_server_configs=server_configs
)

# tools now contains all Alfredo tools + MCP tools
```

## Using MCP Tools with Agent

### Example 1: Basic MCP Integration

```python
from alfredo import Agent
from alfredo.integrations.mcp import load_combined_tools_sync

# Configure MCP servers
server_configs = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    }
}

# Load combined toolset
tools = load_combined_tools_sync(
    cwd=".",
    mcp_server_configs=server_configs
)

# Create agent with all tools
agent = Agent(cwd=".", tools=tools, verbose=True)

# Run task - agent can use both Alfredo and MCP tools
agent.run("List all files in /tmp and create a summary report")

# View execution trace (MCP tools marked with 🔬)
agent.display_trace()
```

### Example 2: Remote MCP Server (Biocontext)

```python
from alfredo import Agent
from alfredo.integrations.mcp import load_combined_tools_sync

# Configure remote biocontext server
server_configs = {
    "biocontext": {
        "transport": "streamable_http",
        "url": "https://mcp.biocontext.ai/mcp/",
    }
}

# Load tools
tools = load_combined_tools_sync(
    cwd=".",
    mcp_server_configs=server_configs
)

# Create agent
agent = Agent(cwd=".", model_name="gpt-4.1-mini", tools=tools)

# Run biological research task
agent.run("""
Get the interactors of TP53 in human and save the results
to a file called tp53_interactors.txt
""")

agent.display_trace()
```

### Example 3: Selective Tool Loading

```python
from alfredo.integrations.langchain import create_langchain_tools
from alfredo.integrations.mcp import load_mcp_tools_sync
from alfredo import Agent

# Load only specific Alfredo tools
alfredo_tools = create_langchain_tools(
    cwd=".",
    tool_ids=["read_file", "write_to_file", "attempt_completion"]
)

# Load MCP tools
mcp_tools = load_mcp_tools_sync(server_configs)

# Combine manually
custom_tools = alfredo_tools + mcp_tools

# Create agent with custom toolset
agent = Agent(cwd=".", tools=custom_tools)
```

## MCP Tool Detection

Alfredo automatically detects MCP tools vs Alfredo tools:

```python
agent = Agent(cwd=".", tools=tools)

# Get tool descriptions
tool_descriptions = agent.get_tool_descriptions()

for tool in tool_descriptions:
    print(f"{tool['name']}: {tool['tool_type']}")
    # tool_type is either "alfredo" or "mcp"

# Execution trace marks tools differently:
agent.display_trace()
# Output:
# 🛠️  [Alfredo] read_file
# 🔬 [MCP] fs_read_file
```

## Popular MCP Servers

### Official MCP Servers

Install via npm:

```bash
# Filesystem access
npm install -g @modelcontextprotocol/server-filesystem

# GitHub integration
npm install -g @modelcontextprotocol/server-github

# Google Drive
npm install -g @modelcontextprotocol/server-gdrive

# PostgreSQL
npm install -g @modelcontextprotocol/server-postgres

# Slack
npm install -g @modelcontextprotocol/server-slack
```

Or use `npx` (no installation needed):

```python
server_configs = {
    "github": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-github",
            "--owner", "fcarli",
            "--repo", "alfredo"
        ],
        "transport": "stdio",
        "env": {
            "GITHUB_TOKEN": "your-github-token"
        }
    }
}
```

### Configuration Examples

**Filesystem Server**:
```python
{
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
        "transport": "stdio"
    }
}
```

**GitHub Server**:
```python
{
    "github": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-github",
            "--owner", "your-username",
            "--repo", "your-repo"
        ],
        "transport": "stdio",
        "env": {"GITHUB_TOKEN": "your-token"}
    }
}
```

**PostgreSQL Server**:
```python
{
    "postgres": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-postgres",
            "postgresql://user:password@localhost/dbname"
        ],
        "transport": "stdio"
    }
}
```

**Custom Python MCP Server**:
```python
{
    "custom": {
        "command": "python",
        "args": ["/path/to/your_mcp_server.py"],
        "transport": "stdio"
    }
}
```

Find more servers: [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

## Advanced: Wrapping MCP Tools with AlfredoTool

You can wrap MCP tools to add node-specific system prompt instructions:

```python
from alfredo.integrations.mcp import load_mcp_tools_sync, wrap_mcp_tools
from alfredo import Agent

# Load MCP tools
mcp_tools = load_mcp_tools_sync(server_configs)

# Wrap with instructions
wrapped_mcp = wrap_mcp_tools(
    mcp_tools,
    instruction_configs={
        "fs_read_file": {
            "agent": "Use this for reading files from the external filesystem",
            "planner": "Include external file reads in your plan"
        },
        "fs_write_file": {
            "agent": "Use this to write to the external filesystem",
            "verifier": "Verify that external files were written correctly"
        }
    }
)

# Load Alfredo tools
from alfredo.integrations.langchain import create_alfredo_tools

alfredo_tools = create_alfredo_tools(cwd=".")

# Combine
all_tools = alfredo_tools + wrapped_mcp

# Create agent
agent = Agent(cwd=".", tools=all_tools)
```

**Learn more**: [AlfredoTool Documentation](alfredo-tools.md)

## Async/Await Pattern

For advanced control over async operations:

```python
import asyncio
from alfredo import Agent
from alfredo.integrations.mcp import load_combined_tools

async def main():
    # Configure servers
    server_configs = {
        "biocontext": {
            "transport": "streamable_http",
            "url": "https://mcp.biocontext.ai/mcp/",
        }
    }

    # Load tools asynchronously
    tools = await load_combined_tools(
        cwd=".",
        mcp_server_configs=server_configs
    )

    print(f"Loaded {len(tools)} tools")

    # Create agent (still synchronous)
    agent = Agent(cwd=".", tools=tools)

    # Run task
    result = agent.run("Get interactors of TP53")

    print(f"Task completed: {result['is_verified']}")

# Run
asyncio.run(main())
```

## Functional API with MCP

You can also use MCP tools with the functional API:

```python
from alfredo.agentic.graph import run_agentic_task
from alfredo.integrations.mcp import load_combined_tools_sync

# Load tools
tools = load_combined_tools_sync(
    cwd=".",
    mcp_server_configs=server_configs
)

# Run task with tools
result = run_agentic_task(
    task="Use biocontext to find TP53 interactors",
    cwd=".",
    model_name="gpt-4.1-mini",
    tools=tools,  # Pass MCP + Alfredo tools
    verbose=True
)

print(result["final_answer"])
```

## Troubleshooting

### ImportError: langchain-mcp-adapters not installed

```python
# Solution: Install the package
uv add langchain-mcp-adapters
```

### MCP Server Not Starting

**Check command path**:
```python
# Good - uses npx to auto-install
"command": "npx"
"args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

# Bad - assumes global install
"command": "@modelcontextprotocol/server-filesystem"
```

**Check permissions**:
```python
# Ensure npx is executable
# Ensure specified directories are accessible
```

### No Tools Loaded from MCP Server

**Verify server config**:
```python
mcp_tools = load_mcp_tools_sync(server_configs)
print(f"Loaded {len(mcp_tools)} tools")

if len(mcp_tools) == 0:
    print("No tools loaded - check server configuration")
```

**Test server independently**:
```bash
# Test filesystem server
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### Async/Sync Compatibility Issues

Alfredo wraps async MCP tools to work with LangGraph's synchronous ToolNode:

```python
# This is handled automatically
mcp_tools = load_mcp_tools_sync(server_configs, make_sync=True)

# If you need to disable sync wrapping (advanced)
mcp_tools = load_mcp_tools_sync(server_configs, make_sync=False)
```

## Security Considerations

### 1. Limit Filesystem Access

```python
# Good - restrict to specific directory
{
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/safe_dir"],
        "transport": "stdio"
    }
}

# Bad - full system access
{
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/"],
        "transport": "stdio"
    }
}
```

### 2. Validate Remote Server URLs

```python
# Use HTTPS
"url": "https://api.example.com/mcp"

# Include authentication
"headers": {
    "Authorization": "Bearer your-token",
    "X-API-Key": "your-api-key"
}
```

### 3. Review MCP Server Code

Before using third-party MCP servers, review their source code to understand what tools they expose and what operations they perform.

### 4. Environment Variables

Store sensitive data in environment variables:

```python
import os

server_configs = {
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "transport": "stdio",
        "env": {
            "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")
        }
    }
}
```

## Example: Complete MCP Workflow

Here's a complete example combining local and remote MCP servers:

```python
import os
from pathlib import Path
from alfredo import Agent
from alfredo.integrations.mcp import load_combined_tools_sync

# Configure multiple MCP servers
server_configs = {
    # Local filesystem server (limited to /tmp)
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    },

    # Remote biocontext API
    "biocontext": {
        "transport": "streamable_http",
        "url": "https://mcp.biocontext.ai/mcp/",
    }
}

# Load all tools (Alfredo + MCP)
tools = load_combined_tools_sync(
    cwd=str(Path.cwd()),
    mcp_server_configs=server_configs
)

print(f"Loaded {len(tools)} total tools")

# Create agent with combined toolset
agent = Agent(
    cwd=str(Path.cwd()),
    model_name="gpt-4.1-mini",
    tools=tools,
    verbose=True
)

# Complex task using multiple tool sources
task = """
Research the protein interactions of TP53 using biocontext.
Save the results to a file in /tmp called tp53_research.txt.
Then read back the file to confirm it was written correctly.
"""

# Run task
result = agent.run(task)

# Display execution trace
print("\n" + "=" * 80)
print("EXECUTION TRACE")
print("=" * 80)
agent.display_trace()

# Show results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Task verified: {result['is_verified']}")
print(f"\nFinal answer:\n{result['final_answer']}")
```

## Related Documentation

- **[Agent Architecture](agent-architecture.md)** - How tools integrate with the graph
- **[Tools](tools.md)** - Built-in Alfredo tools
- **[AlfredoTool](alfredo-tools.md)** - Wrapping MCP tools with custom instructions
