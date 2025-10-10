# Tools Documentation

Alfredo includes 10 built-in tools that enable agents to interact with the file system, execute commands, and control workflow. All tools are designed to be safe, descriptive, and easy to use.

## Tool Categories

- **[File Operations](#file-operations)** - Read, write, and edit files
- **[File Discovery](#file-discovery)** - List and search files
- **[Code Analysis](#code-analysis)** - Parse source code structures
- **[Command Execution](#command-execution)** - Run shell commands
- **[Workflow Control](#workflow-control)** - Ask questions and signal completion

---

## File Operations

### `read_file`

Read the contents of a file with optional partial reading support.

**Description**: Request to read the contents of a file at the specified path. Use this when you need to examine the contents of an existing file. For large files, you can use `limit_bytes` to read only the first N bytes/KB, or use `offset` and `limit` to read a specific range of lines.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | The path of the file to read (relative to the current working directory) |
| `offset` | integer | No | Line number to start reading from (0-indexed). Use this to skip the first N lines. Defaults to 0. Cannot be used with `limit_bytes`. |
| `limit` | integer | No | Maximum number of lines to read from the offset position. Use this to peek at large files. Cannot be used with `limit_bytes`. |
| `limit_bytes` | integer | No | Maximum number of bytes to read from the beginning. Use this to preview large files by size (e.g., 50000 for 50KB). Handles UTF-8 safely. Cannot be used with `offset` or `limit`. |

**Examples**:

```python
# Read entire file
agent.run("Read the file config.yaml")

# Read first 100 lines
agent.run("Read the first 100 lines of large_file.txt")
# Agent will use: read_file(path="large_file.txt", limit=100)

# Read lines 100-200
agent.run("Read lines 100 to 200 of large_file.txt")
# Agent will use: read_file(path="large_file.txt", offset=100, limit=100)

# Read first 50KB
agent.run("Peek at the first 50KB of large_data.json")
# Agent will use: read_file(path="large_data.json", limit_bytes=51200)
```

**Use Cases**:
- Reading configuration files
- Examining source code
- Previewing large files without reading everything
- Analyzing specific sections of files

---

### `write_to_file`

Create a new file or overwrite an existing file with content.

**Description**: Request to write content to a file at the specified path. If the file exists, it will be overwritten. If it doesn't exist, it will be created. This tool will automatically create any directories needed.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | The path of the file to write to (relative to the current working directory) |
| `content` | string | Yes | The content to write to the file. ALWAYS provide the COMPLETE intended content, without any truncation or omissions. |

**Examples**:

```python
# Create a new file
agent.run("Create a Python script hello.py that prints 'Hello, World!'")

# Overwrite existing file
agent.run("Replace the contents of config.yaml with the default configuration")

# Create file in nested directory (creates directories automatically)
agent.run("Write a README to docs/api/README.md")
```

**Use Cases**:
- Creating new files
- Generating code or configuration
- Replacing file contents entirely
- Writing reports or documentation

**Important**: Always provide the **complete** file content. The tool overwrites the entire file.

---

### `replace_in_file`

Apply targeted edits to specific sections of a file using SEARCH/REPLACE blocks.

**Description**: Request to replace sections of content in an existing file using SEARCH/REPLACE blocks. Use this when you need to make targeted changes to specific parts of a file.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | The path of the file to modify (relative to the current working directory) |
| `diff` | string | Yes | One or more SEARCH/REPLACE blocks (see format below) |

**SEARCH/REPLACE Format**:

```
------- SEARCH
[exact content to find]
=======
[new content to replace with]
+++++++ REPLACE
```

**Critical Rules**:
1. SEARCH content must match **EXACTLY** (character-for-character)
2. Each block replaces only the **FIRST** occurrence
3. To delete code, use an empty REPLACE section
4. List multiple blocks in the order they appear in the file

**Examples**:

```python
# Replace a function
diff = """
------- SEARCH
def old_function():
    return "old"
=======
def new_function():
    return "new"
+++++++ REPLACE
"""

# Delete a section
diff = """
------- SEARCH
# TODO: Remove this
debug_code()
=======
+++++++ REPLACE
"""

# Multiple replacements
diff = """
------- SEARCH
version = "1.0.0"
=======
version = "1.1.0"
+++++++ REPLACE

------- SEARCH
DEBUG = True
=======
DEBUG = False
+++++++ REPLACE
"""
```

**Use Cases**:
- Updating specific functions or classes
- Fixing bugs in existing code
- Modifying configuration values
- Removing deprecated code sections

---

## File Discovery

### `list_files`

List files and directories within a specified path.

**Description**: Request to list files and directories within the specified directory. If recursive is true, it will list all files and directories recursively. If recursive is false or not provided, it will only list the top-level contents.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | The path of the directory to list contents for (relative to the current working directory) |
| `recursive` | string | No | Whether to list files recursively. Use 'true' for recursive listing, 'false' or omit for top-level only. |

**Output Format**:
```
[DIR]  path/to/directory/
[FILE] path/to/file.txt (1234 bytes)
```

**Examples**:

```python
# List current directory
agent.run("List all files in the current directory")

# List recursively
agent.run("List all Python files in the src directory recursively")
# Agent will use: list_files(path="src", recursive="true")

# List specific subdirectory
agent.run("What files are in the tests directory?")
```

**Use Cases**:
- Understanding directory structure
- Finding files before reading them
- Checking if files exist
- Getting file size information

---

### `search_files`

Perform regex search across files in a directory.

**Description**: Request to perform a regex search across files in a specified directory. This tool searches for patterns or specific content across multiple files, displaying each match with its line number and context.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | The path of the directory to search in (relative to the current working directory). This directory will be recursively searched. |
| `regex` | string | Yes | The regular expression pattern to search for. Uses Python regex syntax. |
| `file_pattern` | string | No | Glob pattern to filter files (e.g., '*.py' for Python files). If not provided, searches all files. |

**Examples**:

```python
# Search for function definitions in Python files
agent.run("Find all function definitions in the src directory")
# Agent will use: search_files(path="src", regex="def \\w+\\(", file_pattern="*.py")

# Search for TODO comments
agent.run("Find all TODO comments in the codebase")
# Agent will use: search_files(path=".", regex="TODO:", file_pattern="*")

# Search for imports
agent.run("Find all imports of pandas in Python files")
# Agent will use: search_files(path=".", regex="import pandas|from pandas", file_pattern="*.py")
```

**Output Format**:
```
Found 3 match(es):

src/main.py:
  12: import pandas as pd
  45: from pandas import DataFrame

src/analysis.py:
  5: import pandas as pd
```

**Use Cases**:
- Finding specific code patterns
- Locating function or class definitions
- Searching for configuration values
- Finding TODO/FIXME comments

---

## Code Analysis

### `list_code_definition_names`

List all code definitions (classes, functions, methods) in source files.

**Description**: List definition names (classes, functions, methods, etc.) in source code files at the top level of the specified directory. This provides insights into codebase structure and important constructs. Supports Python, JavaScript, TypeScript.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | The directory path (relative to working directory) to analyze for code definitions |

**Supported Languages**:
- **Python**: Classes, functions
- **JavaScript**: Classes, functions, methods, variables, arrow functions
- **TypeScript**: Classes, functions, methods, interfaces, type aliases, enums

**Examples**:

```python
# Analyze Python project
agent.run("List all classes and functions in the src directory")

# Analyze JavaScript project
agent.run("Show me all the exported functions in the lib directory")

# Get overview of codebase
agent.run("What are the main code structures in this project?")
```

**Output Format**:
```
Code definitions in src/:

src/agent.py:
  - Agent (line 17)
  - run_agentic_task (line 253)

src/tools/base.py:
  - BaseToolHandler (line 10)
  - ToolResult (line 45)
```

**Use Cases**:
- Understanding codebase structure
- Finding specific classes or functions
- Generating code documentation
- Code exploration and navigation

**Note**: Requires tree-sitter language parsers to be installed.

---

## Command Execution

### `execute_command`

Execute shell commands on the system.

**Description**: Request to execute a CLI command on the system. Use this when you need to perform system operations or run specific commands. Commands are executed in the current working directory.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `command` | string | Yes | The CLI command to execute. This should be valid for the current operating system. |
| `timeout` | integer | No | Timeout in seconds. Default is 120 seconds (2 minutes). |

**Examples**:

```python
# Run tests
agent.run("Run the pytest test suite")
# Agent will use: execute_command(command="pytest")

# Check Python version
agent.run("What version of Python is installed?")
# Agent will use: execute_command(command="python --version")

# Install package with timeout
agent.run("Install the requests package using pip")
# Agent will use: execute_command(command="pip install requests", timeout=300)

# Run custom script
agent.run("Execute the data_analysis.py script")
# Agent will use: execute_command(command="python data_analysis.py")
```

**Output Format**:
```
STDOUT:
[command output]

STDERR:
[error output if any]

Command exited with code 0
```

**Use Cases**:
- Running tests
- Installing packages
- Executing scripts
- Checking system information
- Running build commands

**Safety Notes**:
- Commands run in the working directory specified when creating the agent
- Timeout prevents hanging commands
- Both stdout and stderr are captured

---

## Workflow Control

### `ask_followup_question`

Request additional information from the user.

**Description**: Request to ask the user a follow-up question. Use this when you need additional information or clarification to proceed with the task. The user will be prompted with your question and can provide a response.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question to ask the user. Be clear and specific about what information you need. |

**Examples**:

```python
# Agent encounters ambiguity
# Agent calls: ask_followup_question(
#     question="Which directory should I create the config file in: 'config/' or 'src/config/'?"
# )

# Need clarification on requirements
# Agent calls: ask_followup_question(
#     question="Should the script process all CSV files or only files matching a specific pattern?"
# )
```

**Use Cases**:
- Resolving ambiguities in task description
- Getting user preferences when multiple options exist
- Requesting missing information needed to complete task
- Confirming destructive operations

**Note**: This creates an interactive workflow where the agent pauses for user input.

---

### `attempt_completion`

Signal task completion and provide a summary.

**Description**: **COMPLETE THE TASK** - Call this when you have finished the task. This is the **ONLY** way to complete the task and end execution. You MUST call this tool when: (1) You have completed all required steps, (2) The task requirements are satisfied, (3) You have verified your work. If you don't call this tool, the task will never complete.

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `result` | string | Yes | A comprehensive summary explaining: 1) What you did to complete the task, 2) What files were created/modified, 3) Any commands you ran and their results, 4) Confirmation that the task is complete. |
| `command` | string | No | If you executed a final command as part of completing the task, include it here for reference. |

**Examples**:

```python
# Successful file creation
attempt_completion(
    result="""I successfully completed the task by:
1. Created hello.py with a print statement that outputs 'Hello, World!'
2. Tested the script with 'python hello.py' - output was correct
3. Verified the file was created and contains the expected code

The task is now complete.""",
    command="python hello.py"
)

# Successful data analysis
attempt_completion(
    result="""I analyzed the sales data and completed all requirements:
1. Read sales.csv (500 rows, 8 columns)
2. Calculated total revenue: $125,430
3. Identified top 3 products by sales volume
4. Created summary report in sales_summary.txt
5. Generated visualization chart.png

All analysis steps completed successfully."""
)
```

**Use Cases**:
- Signaling that all task requirements are met
- Providing comprehensive completion summary
- Triggering the verifier node to review work
- Exiting the agent execution loop

**Critical**: This is the **only** way to exit the agent's ReAct loop. Without calling this tool, the agent will continue indefinitely (until recursion limit).

---

## Creating Custom Tools

You can extend Alfredo with custom tools by following the handler pattern:

### Step 1: Create a Handler Class

```python
from alfredo.tools.base import BaseToolHandler, ToolResult
from typing import Any

class MyToolHandler(BaseToolHandler):
    @property
    def tool_id(self) -> str:
        return "my_tool"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        # Validate parameters
        self.validate_required_param(params, "param_name")

        # Your implementation
        try:
            result = do_something(params["param_name"])
            return ToolResult.ok(f"Success: {result}")
        except Exception as e:
            return ToolResult.err(f"Error: {e}")
```

### Step 2: Register the Tool

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

### Step 3: Use the Tool

```python
from alfredo.integrations.langchain import create_langchain_tools
from alfredo import Agent

# Load all tools (includes your custom tool)
tools = create_langchain_tools(cwd=".")

agent = Agent(cwd=".", tools=tools)
agent.run("Use my custom tool to do something")
```

## Tool Design Best Practices

### 1. **Descriptive Errors**

```python
# Good
return ToolResult.err(f"File not found: {path}. Check that the path is correct.")

# Bad
return ToolResult.err("Error")
```

### 2. **Informative Success Messages**

```python
# Good
return ToolResult.ok(f"Created file: {rel_path} with {len(content)} characters")

# Bad
return ToolResult.ok("Done")
```

### 3. **Validate Inputs**

```python
self.validate_required_param(params, "path")

# Custom validation
if not path.exists():
    return ToolResult.err(f"Path does not exist: {path}")
```

### 4. **Use Relative Paths**

```python
# Good - shows paths relative to cwd
rel_path = self.get_relative_path(file_path)

# Bad - shows absolute paths
abs_path = str(file_path.absolute())
```

### 5. **Handle Exceptions Gracefully**

```python
try:
    result = risky_operation()
    return ToolResult.ok(result)
except SpecificError as e:
    return ToolResult.err(f"Specific error occurred: {e}")
except Exception as e:
    return ToolResult.err(f"Unexpected error: {e}")
```

## Tool Selection by Agent

The agent automatically selects tools based on:

1. **Task requirements** - What needs to be done
2. **Tool descriptions** - How each tool is described in system prompt
3. **Current context** - What information is available
4. **Plan guidance** - What the plan suggests (if planning enabled)

**Example Selection Process**:

```
Task: "Create a Python script that prints 'Hello, World!'"

Agent reasoning:
1. Need to write a file → considers write_to_file
2. File doesn't exist yet → write_to_file is appropriate
3. Knows Python syntax → can generate content
4. After writing → uses attempt_completion to finish

Tool calls:
1. write_to_file(path="hello.py", content="print('Hello, World!')")
2. attempt_completion(result="Created hello.py successfully")
```

## Viewing Available Tools

You can inspect tools at runtime:

```python
agent = Agent(cwd=".")

# Display all tools with descriptions
agent.display_tool_descriptions()

# Save to file
agent.display_tool_descriptions(save_to_file=True)
# Saves to: alfredo/notes/tool_descriptions.md

# Get tools programmatically
tools = agent.get_tool_descriptions()
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")
    print(f"  Type: {tool['tool_type']}")  # 'alfredo' or 'mcp'
    print(f"  Parameters: {tool['parameters']}")
```

## Related Documentation

- **[Agent Architecture](agent-architecture.md)** - How tools integrate with the agent graph
- **[MCP Integration](mcp-integration.md)** - Using external MCP tools
- **[AlfredoTool](alfredo-tools.md)** - Customizing tool system prompts per node
