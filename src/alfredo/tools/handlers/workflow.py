"""Workflow control tool handlers: ask questions and signal completion."""

from typing import Any

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec


class AskFollowupHandler(BaseToolHandler):
    """Handler for requesting additional information from the user.

    This tool allows the agent to ask clarifying questions when it needs
    more information to proceed with a task.
    """

    @property
    def tool_id(self) -> str:
        return "ask_followup_question"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Ask the user a follow-up question.

        Args:
            params: Must contain 'question' - the question to ask

        Returns:
            ToolResult with a special marker indicating user input is needed
        """
        try:
            self.validate_required_param(params, "question")
            question = params["question"]

            # Return a special marker that the agent framework will recognize
            # as requiring user input
            return ToolResult.ok(f"[AWAITING_USER_RESPONSE]\n{question}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Unexpected error: {e}")


class AttemptCompletionHandler(BaseToolHandler):
    """Handler for signaling task completion.

    This tool allows the agent to signal that it believes the task is complete
    and provide a summary of what was accomplished.
    """

    @property
    def tool_id(self) -> str:
        return "attempt_completion"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Signal task completion with a result summary.

        Args:
            params: Optional 'result' - summary of what was accomplished
                   Optional 'command' - a command that was run as part of completion

        Returns:
            ToolResult with completion message
        """
        try:
            result = params.get("result", "Task completed successfully.")
            command = params.get("command")

            # Format completion message
            output_parts = ["[TASK_COMPLETE]", result]

            if command:
                output_parts.append(f"\nFinal command executed: {command}")

            return ToolResult.ok("\n".join(output_parts))

        except Exception as e:
            return ToolResult.err(f"Unexpected error: {e}")


# Register tool specifications
_ask_followup_spec = ToolSpec(
    id="ask_followup_question",
    name="ask_followup_question",
    description=(
        "Request to ask the user a follow-up question. "
        "Use this when you need additional information or clarification to proceed with the task. "
        "The user will be prompted with your question and can provide a response."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="question",
            required=True,
            instruction="The question to ask the user. Be clear and specific about what information you need.",
            usage="What is the target directory for the output files?",
        ),
    ],
)

_attempt_completion_spec = ToolSpec(
    id="attempt_completion",
    name="attempt_completion",
    description=(
        "COMPLETE THE TASK - Call this when you have finished the task. "
        "\n\n"
        "IMPORTANT: This is the ONLY way to complete the task and end execution. "
        "You MUST call this tool when: "
        "(1) You have completed all required steps, "
        "(2) The task requirements are satisfied, "
        "(3) You have verified your work. "
        "\n\n"
        "If you don't call this tool, the task will never complete."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="result",
            required=True,
            instruction=(
                "A comprehensive summary explaining: "
                "1. What you did to complete the task, "
                "2. What files were created/modified, "
                "3. Any commands you ran and their results, "
                "4. Confirmation that the task is complete. "
                "\n\n"
                "Example: 'I successfully completed the task by: "
                "1. Created file X with content Y, "
                "2. Tested with command Z - output was correct, "
                "3. Verified all requirements are met. "
                "The task is now complete.'"
            ),
            usage="I created hello.py with a print statement, ran it successfully, and verified the output.",
        ),
        ToolParameter(
            name="command",
            required=False,
            instruction=(
                "If you executed a final command as part of completing the task, include it here for reference."
            ),
            usage="python hello.py",
        ),
    ],
)

registry.register_spec(_ask_followup_spec)
registry.register_spec(_attempt_completion_spec)

registry.register_handler("ask_followup_question", AskFollowupHandler)
registry.register_handler("attempt_completion", AttemptCompletionHandler)
