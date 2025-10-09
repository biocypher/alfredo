"""Todo list tool handlers: track task progress with a sequential checklist."""

import threading
from typing import Any, Optional

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec


class TodoStateManager:
    """Singleton manager for todo list state.

    This provides a simple in-memory storage for the todo list that can be
    accessed by both the handlers and the graph nodes.
    """

    _instance: Optional["TodoStateManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TodoStateManager":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the manager if not already initialized."""
        if not getattr(self, "_initialized", False):
            self._todo_list: Optional[str] = None
            self._initialized: bool = True

    def get_todo_list(self) -> Optional[str]:
        """Get the current todo list."""
        return self._todo_list

    def set_todo_list(self, content: Optional[str]) -> None:
        """Set the todo list content."""
        self._todo_list = content

    def clear(self) -> None:
        """Clear the todo list (mainly for testing)."""
        self._todo_list = None


class WriteTodoListHandler(BaseToolHandler):
    """Handler for writing/updating the todo list.

    This tool allows the agent to create or update a numbered sequential checklist
    for tracking task progress.
    """

    @property
    def tool_id(self) -> str:
        return "write_todo_list"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Write or update the todo list.

        Args:
            params: Must contain 'content' - the numbered checklist with checkboxes

        Returns:
            ToolResult with the saved checklist
        """
        try:
            self.validate_required_param(params, "content")
            content = params["content"].strip()

            # Store in singleton manager
            manager = TodoStateManager()
            manager.set_todo_list(content)

            return ToolResult.ok(f"Todo list updated:\n\n{content}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Error writing todo list: {e}")


class ReadTodoListHandler(BaseToolHandler):
    """Handler for reading the current todo list.

    This tool allows the agent to check the current state of the todo checklist.
    """

    @property
    def tool_id(self) -> str:
        return "read_todo_list"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Read the current todo list.

        Args:
            params: No parameters required

        Returns:
            ToolResult with the current checklist or a message if none exists
        """
        try:
            manager = TodoStateManager()
            content = manager.get_todo_list()

            if content is None:
                return ToolResult.ok("No todo list created yet.")

            return ToolResult.ok(content)

        except Exception as e:
            return ToolResult.err(f"Error reading todo list: {e}")


# Register tool specifications
_write_todo_spec = ToolSpec(
    id="write_todo_list",
    name="write_todo_list",
    description=(
        "Create or update the todo list with a numbered sequential checklist. "
        "Use this to track task progress. Tasks should be numbered (1, 2, 3...) "
        "and include checkboxes [ ] for incomplete or [x] for complete. "
        "\n\n"
        "**IMPORTANT**: Complete tasks sequentially - finish task 1, then task 2, then task 3, etc. "
        "You can revise the checklist at any time by adding new items, reordering, or modifying tasks "
        "as you discover new requirements."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="content",
            required=True,
            instruction=(
                "The todo list as a numbered checklist with checkboxes. "
                "Format: '1. [ ] Task description' for incomplete, '1. [x] Task description' for complete. "
                "Tasks should be numbered sequentially (1, 2, 3...) in the order they should be completed."
            ),
            usage="1. [x] First task (completed)\n2. [x] Second task (completed)\n3. [ ] Third task (in progress)\n4. [ ] Fourth task (pending)",
        ),
    ],
)

_read_todo_spec = ToolSpec(
    id="read_todo_list",
    name="read_todo_list",
    description="Read the current todo list to check task progress. Use this to see what tasks remain and what has been completed.",
    variant=ModelFamily.GENERIC,
    parameters=[],
)

registry.register_spec(_write_todo_spec)
registry.register_spec(_read_todo_spec)

registry.register_handler("write_todo_list", WriteTodoListHandler)
registry.register_handler("read_todo_list", ReadTodoListHandler)


# Node-specific system instructions for todo list tools
# These are used when creating AlfredoTools from todo tools
TODO_SYSTEM_INSTRUCTIONS = {
    "planner": """
# Todo List Tracking

After creating your implementation plan, call the `write_todo_list` tool to create a numbered sequential checklist.
The checklist should list all major steps in order (1, 2, 3...) with checkboxes to track completion.

Example format:
1. [ ] First task description
2. [ ] Second task description
3. [ ] Third task description

This will help track progress as the task is executed.
""",
    "agent": """
# Todo List Management

You have access to todo list tools for tracking your progress:
- `write_todo_list`: Create or update the numbered checklist
- `read_todo_list`: Check current progress

**Important Instructions:**
- Work through tasks **sequentially** - complete task 1, then task 2, then task 3, etc.
- After completing each task, update the checklist by calling `write_todo_list` with the completed task marked as [x]
- **You can revise the checklist at any time** - add new items, reorder tasks, or modify descriptions as you discover new requirements
- Use `read_todo_list` periodically to check your current progress

Example progression:
```
1. [x] First task (completed)
2. [x] Second task (completed)
3. [ ] Third task (currently working on this)
4. [ ] Fourth task (pending)
```
""",
}
