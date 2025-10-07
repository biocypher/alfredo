# LangChain Integration Guide

Alfredo tools can be seamlessly integrated with LangChain and LangGraph workflows. This guide shows you how to convert Alfredo tools into LangChain-compatible tools.

## Installation

Install Alfredo with LangChain support:

```bash
uv add alfredo[langchain]
```

Or install LangChain separately:

```bash
uv add langchain-core
```

## Quick Start

### Convert a Single Tool

```python
from alfredo.integrations.langchain import create_langchain_tool

# Convert Alfredo's read_file tool to LangChain
read_file_tool = create_langchain_tool("read_file", cwd="/path/to/workspace")

# Use it like any LangChain tool
result = read_file_tool.invoke({"path": "config.yaml"})
print(result)
```

### Convert All Tools

```python
from alfredo.integrations.langchain import create_all_langchain_tools

# Get all Alfredo tools as LangChain tools
tools = create_all_langchain_tools(cwd="/path/to/workspace")

print(f"Available tools: {len(tools)}")
for tool in tools:
    print(f"  - {tool.name}")
```

### Use with LangChain Agent

```python
from langchain_anthropic import ChatAnthropic
from alfredo.integrations.langchain import create_all_langchain_tools

# Create tools
tools = create_all_langchain_tools()

# Create model with tools bound
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)

# Use in agent
response = model_with_tools.invoke("List all Python files in the current directory")
```

## How It Works

### Tool Conversion

Alfredo tools are converted to LangChain `StructuredTool` instances:

1. **Tool Specification** → LangChain tool name and description
2. **Tool Parameters** → Pydantic model for args_schema
3. **Tool Handler** → Wrapped function that LangChain invokes

### Schema Generation

Alfredo automatically generates Pydantic models from tool specifications:

```python
from alfredo.integrations.langchain import create_pydantic_model_from_spec
from alfredo.tools.registry import registry

# Get an Alfredo tool spec
spec = registry.get_spec("read_file")

# Generate Pydantic model for LangChain
model = create_pydantic_model_from_spec(spec)

# Inspect the schema
print(model.schema())
```

## Available Tools

All Alfredo tools are compatible with LangChain:

| Tool ID | Description | LangChain Compatible |
|---------|-------------|---------------------|
| `read_file` | Read file contents | ✅ |
| `write_to_file` | Create/overwrite files | ✅ |
| `replace_in_file` | Apply diff edits | ✅ |
| `execute_command` | Run shell commands | ✅ |
| `list_files` | List directory contents | ✅ |
| `search_files` | Search with regex | ✅ |
| `ask_followup_question` | Request user input | ✅ |
| `attempt_completion` | Signal completion | ✅ |

## Advanced Usage

### Using the @tool Decorator Pattern

```python
from langchain_core.tools import tool
from alfredo.integrations.langchain import as_langchain_tool

@tool
@as_langchain_tool("read_file")
def read_file_tool(path: str) -> str:
    """Read a file from the filesystem.

    Args:
        path: Path to the file to read

    Returns:
        File contents as string
    """
    pass  # Implementation handled by decorator

# Use it
result = read_file_tool.invoke({"path": "README.md"})
```

### Working Directory Context

All file operation tools respect the `cwd` parameter:

```python
from alfredo.integrations.langchain import create_langchain_tool

# Tools will operate in /path/to/project
tool = create_langchain_tool("read_file", cwd="/path/to/project")

# Relative paths are resolved from cwd
result = tool.invoke({"path": "src/main.py"})  # Reads /path/to/project/src/main.py
```

### Model Family Variants

Alfredo supports different tool definitions for different model families:

```python
from alfredo.tools.specs import ModelFamily
from alfredo.integrations.langchain import create_all_langchain_tools

# Get tools optimized for Anthropic models
anthropic_tools = create_all_langchain_tools(model_family=ModelFamily.ANTHROPIC)

# Or for OpenAI models
openai_tools = create_all_langchain_tools(model_family=ModelFamily.OPENAI)
```

### Selective Tool Conversion

Convert only specific tools you need:

```python
from alfredo.integrations.langchain import create_all_langchain_tools

# Only convert file operation tools
file_tools = create_all_langchain_tools(
    tool_ids=["read_file", "write_to_file", "list_files"]
)
```

## LangGraph Integration

### Basic LangGraph Workflow

```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from alfredo.integrations.langchain import create_all_langchain_tools

# Create tools
tools = create_all_langchain_tools()

# Create tool node for LangGraph
tool_node = ToolNode(tools)

# Build your graph
graph = StateGraph()
graph.add_node("agent", agent_function)
graph.add_node("tools", tool_node)
# ... configure edges and compile
```

### Tool Calling in LangGraph

```python
from langchain_anthropic import ChatAnthropic
from alfredo.integrations.langchain import create_all_langchain_tools

# Setup
tools = create_all_langchain_tools()
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)

# In your graph node
def agent_node(state):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

## Error Handling

Alfredo tools return error messages instead of raising exceptions for graceful handling:

```python
tool = create_langchain_tool("read_file")

# File doesn't exist
result = tool.invoke({"path": "nonexistent.txt"})
# Result: "Error: File not found: nonexistent.txt"

# You can check for errors
if "Error:" in result:
    print("Tool execution failed")
```

## Testing

Test your LangChain integration:

```bash
# Run integration tests
uv run pytest tests/test_langchain_integration.py -v

# Run the example
uv run python examples/langchain_integration.py
```

## Examples

### Example 1: File Analysis Agent

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from alfredo.integrations.langchain import create_all_langchain_tools

# Create tools
tools = create_all_langchain_tools()

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful file analysis assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create agent
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent = create_tool_calling_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Use it
result = executor.invoke({
    "input": "List all Python files in the src directory"
})
```

### Example 2: Code Modification Workflow

```python
from alfredo.integrations.langchain import create_langchain_tool

# Get specific tools
read = create_langchain_tool("read_file")
write = create_langchain_tool("write_to_file")
replace = create_langchain_tool("replace_in_file")

# Read file
content = read.invoke({"path": "config.py"})

# Modify (using LLM to generate the diff)
# ... LLM generates SEARCH/REPLACE blocks ...

# Apply changes
result = replace.invoke({
    "path": "config.py",
    "diff": search_replace_blocks
})
```

## API Reference

### `create_langchain_tool(tool_id, cwd=None, model_family=ModelFamily.GENERIC)`

Convert a single Alfredo tool to LangChain.

**Parameters:**
- `tool_id` (str): Alfredo tool identifier
- `cwd` (str, optional): Working directory for file operations
- `model_family` (ModelFamily): Model family for tool variants

**Returns:**
- `StructuredTool`: LangChain tool instance

### `create_all_langchain_tools(cwd=None, model_family=ModelFamily.GENERIC, tool_ids=None)`

Convert all or selected Alfredo tools to LangChain.

**Parameters:**
- `cwd` (str, optional): Working directory for file operations
- `model_family` (ModelFamily): Model family for tool variants
- `tool_ids` (list[str], optional): Specific tools to convert

**Returns:**
- `list[StructuredTool]`: List of LangChain tools

### `create_pydantic_model_from_spec(spec)`

Generate Pydantic model from ToolSpec.

**Parameters:**
- `spec` (ToolSpec): Alfredo tool specification

**Returns:**
- `Type[BaseModel]`: Pydantic model class

## Troubleshooting

### ImportError: No module named 'langchain_core'

Install LangChain:
```bash
uv add langchain-core
```

### Tool not found in registry

Make sure handlers are imported:
```python
# Import to register tools
from alfredo.tools.handlers import file_ops, command, discovery, workflow
```

### Type validation errors

Ensure parameters match the tool's schema:
```python
tool = create_langchain_tool("read_file")

# Correct
result = tool.invoke({"path": "file.txt"})

# Incorrect - missing required parameter
result = tool.invoke({})  # Will fail validation
```

## Best Practices

1. **Set working directory**: Always specify `cwd` for file operations
2. **Handle errors**: Check tool results for error messages
3. **Use specific tools**: Only convert tools you need for better performance
4. **Type everything**: LangChain uses Pydantic models for validation
5. **Test locally**: Run examples before deploying to production

## Resources

- [LangChain Tools Documentation](https://python.langchain.com/docs/concepts/tools/)
- [LangGraph Tools Guide](https://langchain-ai.github.io/langgraph/concepts/tools/)
- [Alfredo Tools README](TOOLS_README.md)
- [Examples Directory](examples/)

## Next Steps

- Explore the [LangChain integration example](examples/langchain_integration.py)
- Build a custom agent with Alfredo tools
- Integrate with LangGraph for complex workflows
- Create custom tool compositions
