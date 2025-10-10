# AlfredoTool: Customizing System Prompts

**AlfredoTool** is a wrapper class that allows you to add **node-specific system prompt instructions** to any tool. This enables fine-grained control over how tools are described and used in different parts of the agent graph.

## Why AlfredoTool?

By default, tools have a single description that appears in all node system prompts. AlfredoTool allows you to:

- **Customize instructions per node** - Different guidance for planner vs agent vs verifier
- **Add context-specific details** - Explain when/how to use tools in specific scenarios
- **Improve agent behavior** - Steer tool usage based on the node's role
- **Maintain compatibility** - Works with Alfredo, MCP, and LangChain tools

## How It Works

### Standard Tool Behavior

Without AlfredoTool, a tool's description appears the same in all nodes:

```python
# Tool has single description
tool = StructuredTool.from_function(
    func=my_function,
    name="my_tool",
    description="Does something useful"
)

# System prompts for all nodes show the same description:
# Agent node:     "my_tool: Does something useful"
# Planner node:   "my_tool: Does something useful"
# Verifier node:  "my_tool: Does something useful"
```

### With AlfredoTool

AlfredoTool wraps the tool and injects custom instructions into specific node system prompts:

```python
from alfredo.tools.alfredo_tool import AlfredoTool

# Wrap tool with node-specific instructions
tool = AlfredoTool.from_langchain(
    langchain_tool,
    system_instructions={
        "agent": "Use this early in your workflow",
        "planner": "Include this in step 2 of your plan",
        "verifier": "Check that this was called successfully"
    }
)

# Now each node sees different instructions:
# Agent node:    "Use this early in your workflow"
# Planner node:  "Include this in step 2 of your plan"
# Verifier node: "Check that this was called successfully"
```

## Creating AlfredoTools

### From Alfredo Tools

```python
from alfredo.tools.alfredo_tool import AlfredoTool

# Create from Alfredo tool ID
tool = AlfredoTool.from_alfredo(
    tool_id="write_todo_list",
    cwd=".",
    system_instructions={
        "agent": "Track your progress with sequential checklist items",
        "planner": "Create initial checklist after making your plan",
        "verifier": "Verify all todo items are completed"
    }
)
```

### From MCP Tools

```python
from alfredo.integrations.mcp import load_mcp_tools_sync
from alfredo.tools.alfredo_tool import AlfredoTool

# Load MCP tools
mcp_tools = load_mcp_tools_sync(server_configs)

# Wrap first tool with instructions
wrapped = AlfredoTool.from_mcp(
    mcp_tool=mcp_tools[0],
    system_instructions={
        "agent": "Use this for external file operations",
        "verifier": "Confirm external files were modified correctly"
    }
)
```

### From Any LangChain Tool

```python
from langchain_core.tools import StructuredTool
from alfredo.tools.alfredo_tool import AlfredoTool

# Create a custom LangChain tool
custom_tool = StructuredTool.from_function(
    func=my_function,
    name="custom_tool",
    description="My custom tool"
)

# Wrap it
wrapped = AlfredoTool.from_langchain(
    langchain_tool=custom_tool,
    system_instructions={
        "agent": "Use carefully - has side effects",
        "planner": "Plan for potential failures"
    }
)
```

## System Instructions Structure

### Targeting Specific Nodes

You can target these graph nodes:

- **`planner`** - Appears when creating the initial plan
- **`agent`** - Appears during execution and tool selection
- **`verifier`** - Appears when verifying task completion
- **`replan`** - Appears when creating improved plans after failure

```python
system_instructions = {
    "planner": "Instructions for planning phase",
    "agent": "Instructions during execution",
    "verifier": "Instructions for verification",
    "replan": "Instructions when replanning"
}
```

### Partial Targeting

You don't need to specify instructions for all nodes:

```python
# Only appears in agent node
system_instructions = {
    "agent": "Use this to save checkpoints during execution"
}

# Only appears in verifier node
system_instructions = {
    "verifier": "Confirm that output files exist and are non-empty"
}

# Multiple nodes
system_instructions = {
    "agent": "Execute this before other operations",
    "verifier": "Check that this was called first"
}
```

## Instruction Injection

### Where Instructions Appear

When a node generates its system prompt, it includes:

1. **Tool base description** - The tool's main description
2. **Tool parameters** - Parameter descriptions
3. **Node-specific instructions** (if present) - Your custom instructions

**Example System Prompt (Agent Node)**:

```
Available Tools:

## write_todo_list
Base description: Create or update a numbered todo list

Parameters:
- items: List of todo items

**Instructions for this node:**
Track your progress with sequential checklist items. After completing
each major step, update the todo list to mark it as done.

[Rest of system prompt...]
```

### Multi-Tool Instructions

If multiple tools have instructions for a node, all are included:

```python
tools = [
    AlfredoTool.from_alfredo(
        "write_todo_list",
        system_instructions={"agent": "Use for progress tracking"}
    ),
    AlfredoTool.from_alfredo(
        "execute_command",
        system_instructions={"agent": "Run tests after each change"}
    )
]

# Agent node system prompt will include both instructions
```

## Complete Example

### Building a Custom Toolset

```python
from alfredo import Agent
from alfredo.tools.alfredo_tool import AlfredoTool
from alfredo.integrations.langchain import create_langchain_tools
from alfredo.integrations.mcp import load_mcp_tools_sync

# 1. Load base Alfredo tools
base_tools = create_langchain_tools(cwd=".")

# 2. Wrap specific tools with custom instructions
todo_tool = AlfredoTool.from_alfredo(
    tool_id="write_todo_list",
    cwd=".",
    system_instructions={
        "planner": """
After creating your implementation plan, create an initial todo list
that breaks down each step into checkable items.
        """,
        "agent": """
Update the todo list after completing each major step. This helps
track progress and provides visibility into what remains.
        """,
        "verifier": """
Check if all todo items are marked as complete. Incomplete items
suggest the task is not fully finished.
        """
    }
)

# 3. Load and wrap MCP tools
mcp_server_configs = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    }
}

mcp_tools = load_mcp_tools_sync(mcp_server_configs)

# Wrap MCP filesystem tool
fs_tool = AlfredoTool.from_mcp(
    mcp_tool=next(t for t in mcp_tools if t.name == "read_file"),
    system_instructions={
        "agent": "Use for reading files from external /tmp directory",
        "verifier": "Confirm files from /tmp were read correctly"
    }
)

# 4. Combine all tools
custom_tools = base_tools + [todo_tool, fs_tool]

# 5. Create agent with custom toolset
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",
    tools=custom_tools,
    verbose=True
)

# 6. Run task
agent.run("""
Create a Python script that reads data from /tmp/input.csv,
processes it, and writes results to output.txt.
Track your progress with a todo list.
""")

# 7. View execution
agent.display_trace()
```

## Bulk Wrapping

### Wrapping MCP Tools

Use the helper function to wrap multiple MCP tools at once:

```python
from alfredo.integrations.mcp import load_mcp_tools_sync, wrap_mcp_tools

# Load MCP tools
mcp_tools = load_mcp_tools_sync(server_configs)

# Wrap all with instructions
wrapped_mcp = wrap_mcp_tools(
    mcp_tools,
    instruction_configs={
        "read_file": {
            "agent": "Use for external file reads",
            "verifier": "Confirm read was successful"
        },
        "write_file": {
            "agent": "Use for external file writes",
            "verifier": "Confirm write was successful"
        }
    }
)
```

### Combining with Alfredo Tools

```python
from alfredo.integrations.langchain import create_alfredo_tools

# Load Alfredo tools (as AlfredoTools with todo list instructions)
alfredo_tools = create_alfredo_tools(
    cwd=".",
    tool_configs={
        "write_todo_list": {
            "agent": "Track progress sequentially",
            "planner": "Create initial checklist",
            "verifier": "Verify all items complete"
        }
    }
)

# Combine with wrapped MCP tools
all_tools = alfredo_tools + wrapped_mcp

# Use with agent
agent = Agent(cwd=".", tools=all_tools)
```

## Inspecting AlfredoTools

### Getting Instructions

```python
tool = AlfredoTool.from_alfredo(
    "write_todo_list",
    system_instructions={
        "agent": "Track progress",
        "verifier": "Check completion"
    }
)

# Get instruction for specific node
agent_instruction = tool.get_instruction_for_node("agent")
print(agent_instruction)  # "Track progress"

verifier_instruction = tool.get_instruction_for_node("verifier")
print(verifier_instruction)  # "Check completion"

# Get all targeted nodes
target_nodes = tool.get_target_nodes()
print(target_nodes)  # ["agent", "verifier"]

# Check if tool targets a node
has_agent_instruction = tool.is_available_for_node("agent")
print(has_agent_instruction)  # True
```

### Viewing Tool Descriptions

The Agent class provides methods to inspect tools:

```python
agent = Agent(cwd=".", tools=custom_tools)

# Display all tools with descriptions
agent.display_tool_descriptions()

# Save to file
agent.display_tool_descriptions(save_to_file=True)
# Saves to: alfredo/notes/tool_descriptions.md

# Get tool info programmatically
tools_info = agent.get_tool_descriptions()

for tool in tools_info:
    print(f"\nTool: {tool['name']}")
    print(f"Type: {tool['tool_type']}")  # 'alfredo' or 'mcp'
    print(f"Target Nodes: {tool['target_nodes']}")  # ['agent', 'verifier', ...]
```

## Use Cases

### 1. Progress Tracking

Add todo list with specific instructions for each node:

```python
todo_tool = AlfredoTool.from_alfredo(
    "write_todo_list",
    cwd=".",
    system_instructions={
        "planner": "Create checklist with 3-5 high-level milestones",
        "agent": "Update after each completed step. Mark items DONE as you finish.",
        "verifier": "Ensure all items marked DONE before approving",
        "replan": "Revise todo list to reflect new plan"
    }
)
```

### 2. External Service Integration

Guide usage of MCP tools for external APIs:

```python
bio_tool = AlfredoTool.from_mcp(
    mcp_tool,
    system_instructions={
        "agent": """
This tool queries the biocontext API for protein interactions.
Use it BEFORE attempting to write results - gather data first.
        """,
        "verifier": """
Confirm that biocontext data was retrieved AND saved to a file.
Check both the API call and file write operations.
        """
    }
)
```

### 3. Workflow Enforcement

Enforce specific execution order:

```python
setup_tool = AlfredoTool.from_alfredo(
    "execute_command",
    cwd=".",
    system_instructions={
        "agent": """
Run 'npm install' FIRST before any other commands.
This ensures dependencies are available.
        """,
        "verifier": """
Check that setup commands (npm install, etc.) were executed
before application commands.
        """
    }
)
```

### 4. Safety Guardrails

Add warnings for destructive operations:

```python
delete_tool = AlfredoTool.from_langchain(
    dangerous_tool,
    system_instructions={
        "agent": """
WARNING: This tool deletes files permanently.
ALWAYS list files first to confirm targets before deletion.
        """,
        "planner": """
Plan includes file deletion - ensure backup step is included first.
        """,
        "verifier": """
Confirm that only intended files were deleted.
Check file listing to verify correct targets.
        """
    }
)
```

### 5. Performance Optimization

Guide efficient tool usage:

```python
search_tool = AlfredoTool.from_alfredo(
    "search_files",
    cwd=".",
    system_instructions={
        "agent": """
Use file_pattern parameter to narrow search scope.
Example: file_pattern="*.py" for Python files only.
This is much faster than searching all files.
        """,
        "planner": """
Include file search early in plan to understand codebase structure
before making modifications.
        """
    }
)
```

## Best Practices

### 1. Be Specific and Actionable

```python
# Good - specific guidance
"agent": "Call this tool AFTER reading the config file to get database credentials"

# Bad - vague
"agent": "Use this sometimes"
```

### 2. Explain the "Why"

```python
# Good - includes reasoning
"agent": "Use limit=100 for large files to avoid reading everything. This speeds up initial exploration."

# Bad - just commands
"agent": "Use limit=100"
```

### 3. Target Relevant Nodes Only

```python
# Good - only targets nodes where instruction is relevant
system_instructions={
    "agent": "Execute this before database operations",
    "verifier": "Confirm database was initialized"
}

# Avoid - instructions for nodes that don't need them
system_instructions={
    "planner": "Some instruction",
    "agent": "Some instruction",
    "verifier": "Some instruction",
    "replan": "Some instruction"  # If not needed, don't include
}
```

### 4. Use Examples

```python
"agent": """
Use this to query protein interactions.
Example: get_interactions(protein="TP53", organism="human")
"""
```

### 5. Highlight Prerequisites

```python
"agent": """
PREREQUISITE: Ensure API key is set in environment (BIOCONTEXT_API_KEY).
Use this tool to query biological databases.
"""
```

## Metadata Support

AlfredoTool supports optional metadata for additional context:

```python
tool = AlfredoTool.from_alfredo(
    "my_tool",
    cwd=".",
    system_instructions={...},
    metadata={
        "version": "1.0",
        "author": "Your Name",
        "category": "external_api",
        "requires_auth": True
    }
)

# Access metadata
print(tool.metadata)
```

## Converting to LangChain

AlfredoTool wraps LangChain tools, so you can extract the underlying tool:

```python
alfredo_tool = AlfredoTool.from_alfredo("read_file")

# Get underlying LangChain tool
langchain_tool = alfredo_tool.to_langchain_tool()

# Use with any LangChain agent or workflow
```

## Related Documentation

- **[Agent Architecture](agent-architecture.md)** - How nodes use system prompts
- **[Tools](tools.md)** - Built-in Alfredo tools
- **[MCP Integration](mcp-integration.md)** - Wrapping MCP tools with instructions
