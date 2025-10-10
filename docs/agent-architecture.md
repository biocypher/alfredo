# Agent Architecture

Alfredo's Agent class provides an autonomous task execution system built on **LangGraph**, featuring automatic planning, execution, and verification with retry logic.

## Overview

The Agent class wraps a LangGraph state graph that orchestrates multiple specialized nodes to complete tasks autonomously. Each node has a specific role in the execution pipeline, working together to plan, execute, verify, and refine task completion.

## The State Graph

### Full Graph (Planning Enabled)

```
START → planner → agent ⇄ tools → verifier
                   ↑              ↓
                   └── replan ←───┘
                                  ↓
                                 END
```

### Simplified Graph (Planning Disabled)

```
START → agent ⇄ tools → verifier → END
```

## Graph Nodes

### 1. **planner** (Optional)

**Purpose**: Creates an initial implementation plan for the task.

**Inputs**:
- `task`: The user's original task description

**Outputs**:
- `plan`: A structured implementation plan
- `plan_iteration`: Counter (starts at 1)
- `messages`: Initial plan message added to history

**Behavior**:
- Analyzes the task and available tools
- Generates a step-by-step implementation plan
- Considers tool capabilities and task requirements
- Can be disabled via `enable_planning=False`

**When to disable planning**:
- Simple, straightforward tasks
- When you want faster startup (no planning overhead)
- For interactive tasks where planning may be overly rigid
- When you prefer the agent to explore freely

**Example**:
```python
# With planning (default)
agent = Agent(cwd=".", model_name="gpt-4.1-mini")

# Without planning - starts directly at agent node
agent = Agent(cwd=".", enable_planning=False)
```

### 2. **agent**

**Purpose**: Performs reasoning and decides which tools to call using the ReAct pattern.

**Inputs**:
- `task`: Original task
- `plan`: Current implementation plan
- `messages`: Conversation history
- `tools`: Available tools (bound to model)

**Outputs**:
- `messages`: AI message with tool calls or reasoning

**Behavior**:
- Receives a system prompt with task context, plan, and tool descriptions
- Reasons about next actions based on current state
- Calls tools as needed to make progress
- Continues in a loop until task completion

**System Prompt Structure**:
```
Task: {task}
Plan: {plan}

{tool_instructions}

[Instructions on how to use tools and signal completion]
```

**Routing from agent**:
- Has tool calls → routes to `tools` node
- No tool calls → loops back to `agent` (continues reasoning)

### 3. **tools**

**Purpose**: Executes tool calls made by the agent.

**Inputs**:
- `messages`: Contains AIMessage with tool_calls

**Outputs**:
- `messages`: ToolMessage with results for each tool call

**Behavior**:
- Uses LangGraph's `ToolNode` for standard tool execution
- Wraps tools node to sync todo list state (if todo tools present)
- Executes all tool calls from the last AI message
- Returns results as ToolMessages

**Special Handling**:
- Detects `attempt_completion` tool (marked by `[TASK_COMPLETE]`)
- Syncs todo list state between AgentState and TodoStateManager

**Routing from tools**:
- If `attempt_completion` called → routes to `verifier`
- Otherwise → routes back to `agent`

### 4. **verifier**

**Purpose**: Verifies if the task is complete and the answer is satisfactory.

**Inputs**:
- `task`: Original task
- `final_answer`: Result from `attempt_completion` tool
- `messages`: Full execution trace

**Outputs**:
- `is_verified`: Boolean indicating if task is complete
- `final_answer`: The extracted answer
- `messages`: Verification feedback message

**Behavior**:
- Extracts the result from `attempt_completion` tool call
- Formats execution trace showing all actions taken
- Uses model to verify if task requirements are met
- Checks for completeness, accuracy, and satisfaction

**Verification Prompt**:
```
Task: {task}
Answer: {answer}

Execution Trace:
{formatted_trace}

Is this answer complete and satisfactory?
```

**Routing from verifier**:
- **With planning**:
  - `VERIFIED:` → END (task complete)
  - `NOT_VERIFIED:` → `replan` (create improved plan)
- **Without planning**:
  - Always → END (no retry)

### 5. **replan** (Planning Mode Only)

**Purpose**: Creates an improved plan after verification failure.

**Inputs**:
- `task`: Original task
- `previous_plan`: The plan that failed
- `verification_feedback`: Why it failed
- `plan_iteration`: Current iteration count

**Outputs**:
- `plan`: New improved plan
- `plan_iteration`: Incremented counter
- `final_answer`: Reset to None
- `is_verified`: Reset to False
- `messages`: Replan message

**Behavior**:
- Analyzes verification feedback to understand failures
- Creates an improved plan addressing the issues
- Increments plan iteration counter
- Resets verification state for new attempt

**Routing from replan**:
- Always → `agent` (start execution with new plan)

## State Management

### AgentState Type

The state graph uses a TypedDict to track execution state:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    plan: str
    plan_iteration: int
    max_context_tokens: int
    final_answer: Optional[str]
    is_verified: bool
    todo_list: Optional[str]
```

**Field Descriptions**:

- **messages**: Conversation history (LangChain messages)
  - Uses `add_messages` to append new messages
  - Contains HumanMessage, AIMessage, ToolMessage, SystemMessage

- **task**: The original user task (immutable throughout execution)

- **plan**: Current implementation plan
  - Set by `planner` or `replan` nodes
  - Read by `agent` and `verifier` nodes

- **plan_iteration**: Number of times plan has been created/updated
  - Starts at 0, incremented by planner/replan
  - Can be used to limit replanning attempts

- **max_context_tokens**: Maximum context window size

- **final_answer**: Result from `attempt_completion` tool
  - Extracted by `verifier` node
  - Reset to None during replanning

- **is_verified**: Whether verifier approved the answer
  - Determines routing (END or replan)

- **todo_list**: Optional todo list for task progress tracking
  - Synced with TodoStateManager if todo tools are used

## Execution Flow Examples

### Example 1: Successful Task (No Replanning)

```
1. START
   ↓
2. planner: Creates plan
   ↓
3. agent: Reads plan, calls list_files tool
   ↓
4. tools: Executes list_files
   ↓
5. agent: Calls read_file tool
   ↓
6. tools: Executes read_file
   ↓
7. agent: Calls write_to_file tool
   ↓
8. tools: Executes write_to_file
   ↓
9. agent: Calls attempt_completion tool
   ↓
10. tools: Returns completion result
    ↓
11. verifier: Reviews work → VERIFIED
    ↓
12. END
```

### Example 2: Task with Replanning

```
1. START
   ↓
2. planner: Creates initial plan
   ↓
3. agent: Executes partial plan, calls attempt_completion
   ↓
4. tools: Returns completion result
   ↓
5. verifier: Reviews → NOT_VERIFIED (missing requirements)
   ↓
6. replan: Creates improved plan addressing feedback
   ↓
7. agent: Executes new plan with corrections
   ↓
8. tools: Executes tools
   ↓
9. agent: Calls attempt_completion
   ↓
10. tools: Returns completion result
    ↓
11. verifier: Reviews → VERIFIED
    ↓
12. END
```

### Example 3: Without Planning

```
1. START
   ↓
2. agent: Directly starts execution, calls tools
   ↓
3. tools: Executes tools
   ↓
4. agent: Continues, calls more tools
   ↓
5. tools: Executes tools
   ↓
6. agent: Calls attempt_completion
   ↓
7. tools: Returns completion result
   ↓
8. verifier: Reviews → VERIFIED or NOT_VERIFIED
   ↓
9. END (no retry if not verified)
```

## Creating an Agent

### Basic Usage

```python
from alfredo import Agent

# Create with default settings
agent = Agent(
    cwd=".",
    model_name="gpt-4.1-mini",
    verbose=True
)

# Run a task
result = agent.run("Create a Python script that prints 'Hello, World!'")

# Access results
print(result["final_answer"])
print(result["is_verified"])
```

### Configuration Options

```python
agent = Agent(
    cwd=".",                        # Working directory
    model_name="gpt-4.1-mini",      # Model to use
    max_context_tokens=100000,      # Context window size
    tools=None,                     # Custom tools (None = use all Alfredo tools)
    verbose=True,                   # Print progress
    recursion_limit=50,             # Max graph steps
    enable_planning=True,           # Use planner/replan nodes
    temperature=0.7,                # Model temperature (via **kwargs)
    base_url="...",                 # Custom API endpoint (via **kwargs)
)
```

### Model Selection

Alfredo uses LangChain's `init_chat_model` which supports:

```python
# OpenAI
agent = Agent(model_name="gpt-4.1-mini")
agent = Agent(model_name="gpt-4o")

# Anthropic
agent = Agent(model_name="anthropic/claude-3-5-sonnet-20241022")

# Custom endpoint
agent = Agent(
    model_name="custom-model",
    base_url="https://api.example.com/v1",
    api_key="your-key"
)
```

## Custom System Prompts

Each node has a default system prompt, but you can customize them:

### Setting Custom Prompts

```python
agent = Agent(cwd=".", model_name="gpt-4.1-mini")

# Customize planner
agent.set_planner_prompt("""
Create a concise, numbered implementation plan.
Focus on efficiency and simplicity.

Task: {task}
{tool_instructions}
""")

# Customize agent
agent.set_agent_prompt("""
Task: {task}
Plan: {plan}

Work methodically through each step.
{tool_instructions}
""")

# Customize verifier
agent.set_verifier_prompt("""
Task: {task}
Answer: {answer}

Trace: {trace_section}

Verify: Is this complete and correct?
{tool_instructions}
""")

# Customize replan
agent.set_replan_prompt("""
Task: {task}
Failed Plan: {previous_plan}
Feedback: {verification_feedback}

Create a better plan.
{tool_instructions}
""")
```

### Prompt Template Strategies

**Strategy 1: Plain Text (Auto-wrapping)**

Provide plain text without placeholders. The system automatically:
- Prepends dynamic variables (task, plan, etc.) at the beginning
- Appends tool-specific instructions at the end
- Keeps your custom content in the middle

```python
agent.set_planner_prompt("""
Your job is to create a detailed implementation plan.
Be thorough and specific in your planning.
""")
```

**Strategy 2: Explicit Placeholders (Validation)**

Provide a template with `{placeholder}` variables. The system:
- Validates that all required placeholders are present
- Formats the template with actual values
- Raises `ValueError` if required placeholders are missing

```python
agent.set_agent_prompt("""
# Task
{task}

# Plan
{plan}

# Instructions
Execute the plan step by step.

{tool_instructions}
""")
```

### Required Placeholders by Node

| Node | Required Placeholders |
|------|----------------------|
| **planner** | `{task}`, `{tool_instructions}` |
| **agent** | `{task}`, `{plan}`, `{tool_instructions}` |
| **verifier** | `{task}`, `{answer}`, `{trace_section}`, `{tool_instructions}` |
| **replan** | `{task}`, `{previous_plan}`, `{verification_feedback}`, `{tool_instructions}` |

**Note**: `{tool_instructions}` is **always required** to ensure tools with custom instructions work properly.

### Managing Prompts

```python
# Get current template
template = agent.get_prompt_template("planner")

# Preview all prompts
agent.display_system_prompts(
    task="Example task",
    plan="Example plan"
)

# Save prompts to file
agent.display_system_prompts(save_to_file=True)
# Saves to: alfredo/notes/system_prompts.md

# Reset all to defaults
agent.reset_prompts()
```

## Tool Integration

### Using Default Tools

```python
# Uses all Alfredo tools by default
agent = Agent(cwd=".", model_name="gpt-4.1-mini")
```

### Using Custom Tools

```python
from alfredo.integrations.langchain import create_langchain_tools

# Load specific Alfredo tools
tools = create_langchain_tools(
    cwd=".",
    tool_ids=["read_file", "write_to_file", "attempt_completion"]
)

agent = Agent(cwd=".", tools=tools)
```

### Using MCP Tools

```python
from alfredo.integrations.mcp import load_combined_tools_sync

# Load Alfredo + MCP tools
tools = load_combined_tools_sync(
    cwd=".",
    mcp_server_configs={
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "transport": "stdio"
        }
    }
)

agent = Agent(cwd=".", tools=tools)
```

### Important: attempt_completion Tool

The `attempt_completion` tool is **always required** and automatically added if missing. This tool is the **only way** for the agent to signal task completion and exit the ReAct loop.

## Execution Tracing

### Viewing Execution Trace

```python
# Run task
agent.run("Create a Python script")

# Display full trace
agent.display_trace()
```

**Trace Output Includes**:
- All messages exchanged (Human, AI, Tool, System)
- Tool calls with arguments
- Tool responses
- Verification results
- Metadata (plan iterations, verification status)

### Detecting MCP vs Alfredo Tools

```python
# In trace output, tools are marked:
# 🛠️  [Alfredo] read_file
# 🔬 [MCP] fs_read_file
```

## Recursion Limit

The recursion limit prevents infinite loops:

```python
agent = Agent(
    cwd=".",
    recursion_limit=50  # Default: 50 graph steps
)
```

**What counts as a step**:
- Each node execution (planner, agent, tools, verifier, replan)
- Loops back to agent count as additional steps

**When to adjust**:
- Increase for complex tasks requiring many tool calls
- Decrease for simple tasks to fail fast
- Monitor via `len(result["messages"])` to see step count

## Best Practices

### 1. Choose Planning Mode Based on Task Complexity

- **Enable planning** (default) for:
  - Multi-step tasks requiring coordination
  - Tasks with unclear steps that benefit from upfront planning
  - When verification and retry logic is important

- **Disable planning** for:
  - Simple, single-step tasks
  - Interactive tasks where planning adds overhead
  - When you want the agent to explore freely

### 2. Set Appropriate Recursion Limits

```python
# Simple task
agent = Agent(recursion_limit=20)

# Complex task
agent = Agent(recursion_limit=100)
```

### 3. Use Verbose Mode During Development

```python
agent = Agent(verbose=True)  # See progress updates
```

### 4. Customize Prompts for Domain-Specific Tasks

```python
agent.set_agent_prompt("""
You are a data scientist analyzing datasets.

Task: {task}
Plan: {plan}

Use pandas and statistical methods.
{tool_instructions}
""")
```

### 5. Monitor Plan Iterations

```python
result = agent.run("...")
print(f"Plan iterations: {result['plan_iteration']}")
# If high, consider improving planner prompt or task description
```

## Functional API Alternative

For one-off tasks without creating an agent instance:

```python
from alfredo.agentic.graph import run_agentic_task

result = run_agentic_task(
    task="Create a hello world script",
    cwd=".",
    model_name="gpt-4.1-mini",
    verbose=True,
    enable_planning=True,
    recursion_limit=50
)

print(result["final_answer"])
```

## Related Documentation

- **[Tools](tools.md)** - Available tools and creating custom tools
- **[AlfredoTool](alfredo-tools.md)** - Node-specific tool instructions
- **[MCP Integration](mcp-integration.md)** - Using MCP servers
- **[Prebuilt Agents](prebuilt-agents.md)** - ExplorationAgent and ReflexionAgent
