"""Command execution tool handler."""

import subprocess
from typing import Any

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec


class ExecuteCommandHandler(BaseToolHandler):
    """Handler for executing shell commands."""

    @property
    def tool_id(self) -> str:
        return "execute_command"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute a shell command and return the output.

        Args:
            params: Must contain 'command'. Optional: 'timeout' (in seconds)

        Returns:
            ToolResult with command output or error
        """
        try:
            self.validate_required_param(params, "command")
            command = params["command"]
            timeout = params.get("timeout", 120)  # Default 2 minute timeout

            # Parse timeout if it's a string
            if isinstance(timeout, str):
                try:
                    timeout = int(timeout)
                except ValueError:
                    timeout = 120

            # Execute command
            try:
                result = subprocess.run(  # noqa: S602
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(self.cwd),
                    timeout=timeout,
                )

                # Format output
                output_parts = []

                if result.stdout:
                    output_parts.append(f"STDOUT:\n{result.stdout}")

                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")

                if result.returncode != 0:
                    output_parts.append(f"\nCommand exited with code {result.returncode}")

                output = "\n\n".join(output_parts) if output_parts else "Command completed with no output"

                return ToolResult.ok(output)

            except subprocess.TimeoutExpired:
                return ToolResult.err(f"Command timed out after {timeout} seconds")

            except Exception as e:
                return ToolResult.err(f"Error executing command: {e}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Unexpected error: {e}")


# Register tool specification
_execute_command_spec = ToolSpec(
    id="execute_command",
    name="execute_command",
    description=(
        "Request to execute a CLI command on the system. "
        "Use this when you need to perform system operations or run specific commands. "
        "Commands are executed in the current working directory."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="command",
            required=True,
            instruction=(
                "The CLI command to execute. This should be valid for the current operating system. "
                "Ensure the command is properly formatted."
            ),
            usage="your command here",
        ),
        ToolParameter(
            name="timeout",
            required=False,
            instruction="Timeout in seconds. Default is 120 seconds (2 minutes).",
            usage="60",
        ),
    ],
)

registry.register_spec(_execute_command_spec)
registry.register_handler("execute_command", ExecuteCommandHandler)
