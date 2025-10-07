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
        "Request to signal completion of the task. "
        "Use this when you believe you have successfully completed the user's request. "
        "Provide a clear summary of what was accomplished."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="result",
            required=False,
            instruction=(
                "A summary of what was accomplished and any important details the user should know. "
                "Be comprehensive but concise."
            ),
            usage="Successfully created 3 files and ran all tests. Everything passed.",
        ),
        ToolParameter(
            name="command",
            required=False,
            instruction=(
                "If you executed a command as part of completing the task, you can include it here "
                "for the user's reference."
            ),
            usage="npm test",
        ),
    ],
)

registry.register_spec(_ask_followup_spec)
registry.register_spec(_attempt_completion_spec)

registry.register_handler("ask_followup_question", AskFollowupHandler)
registry.register_handler("attempt_completion", AttemptCompletionHandler)
