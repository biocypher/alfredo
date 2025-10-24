"""Class-based interface for the agentic scaffold."""
# mypy: disable-error-code="no-any-unimported"

from typing import Any, Optional

from alfredo.agentic.graph import create_agentic_graph
from alfredo.agentic.prompt_templates import PromptTemplates
from alfredo.agentic.prompts import (
    get_agent_system_prompt,
    get_planning_prompt,
    get_replan_prompt,
    get_verification_prompt,
)
from alfredo.agentic.state import AgentState


class Agent:
    """Agentic scaffold with LangGraph-based plan-verify-replan loop.

    This is the recommended way to use Alfredo for autonomous task execution.
    The agent automatically plans, executes tools, and verifies completion.

    Example:
        >>> from alfredo import Agent
        >>> agent = Agent(cwd=".", model_name="gpt-4.1-mini", verbose=True)
        >>> agent.run("Create a hello world Python script")
        >>> agent.display_trace()
        >>> print(agent.results["final_answer"])
    """

    def __init__(
        self,
        cwd: str = ".",
        model_name: str = "gpt-4.1-mini",
        max_context_tokens: int = 100000,
        tools: Optional[list] = None,
        verbose: bool = True,
        recursion_limit: int = 50,
        enable_planning: bool = True,
        parse_reasoning: bool = False,
        codeact_mcp_functions: Optional[dict[str, dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the agentic agent.

        Args:
            cwd: Working directory for file operations (default: ".")
            model_name: Name of the model to use (default: "gpt-4.1-mini")
            max_context_tokens: Maximum context window size in tokens
            tools: Optional list of LangChain tools. If None, uses all Alfredo tools.
                Can include MCP tools loaded via alfredo.integrations.mcp
            verbose: Whether to print progress updates during execution
            recursion_limit: Maximum number of graph steps (default: 50)
            enable_planning: Whether to use the planner node for creating implementation plans.
                If False, agent starts directly without planning step (default: True)
            parse_reasoning: Whether to parse <think> tags from model responses and store in
                AIMessage.additional_kwargs['reasoning'] (default: False)
            codeact_mcp_functions: Optional MCP server configurations for generating
                importable Python wrapper modules. Format: {"name": {"url": "...", "headers": {...}}}
            **kwargs: Additional keyword arguments to pass to the model
                (e.g., temperature, base_url, api_key)
        """
        self.cwd = cwd
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.verbose = verbose
        self.recursion_limit = recursion_limit
        self.enable_planning = enable_planning
        self.parse_reasoning = parse_reasoning
        self.model_kwargs = kwargs

        # Normalize tools (wrap plain StructuredTools as AlfredoTools)
        self.tools: Optional[list[Any]]
        if tools is not None:
            from alfredo.tools.alfredo_tool import AlfredoTool

            self.tools = [t if isinstance(t, AlfredoTool) else AlfredoTool.from_langchain(t) for t in tools]
        else:
            self.tools = None

        # Generate MCP HTTP wrapper modules if configured
        if codeact_mcp_functions:
            self._generate_mcp_http_wrappers(codeact_mcp_functions)

        # Storage for custom prompt templates
        self.prompt_templates = PromptTemplates()

        # Create the LangGraph
        self.graph = create_agentic_graph(
            cwd=cwd,
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            tools=self.tools,
            recursion_limit=recursion_limit,
            enable_planning=enable_planning,
            parse_reasoning=parse_reasoning,
            prompt_templates=self.prompt_templates,
            **kwargs,
        )

        # Storage for execution results
        self._results: Optional[dict[str, Any]] = None

    def _generate_mcp_http_wrappers(self, codeact_mcp_functions: dict[str, dict[str, Any]]) -> None:  # noqa: C901
        """Generate MCP HTTP wrapper modules and add tool with system instructions.

        Args:
            codeact_mcp_functions: Dictionary mapping module names to server configurations
                Format: {"name": {"url": "...", "headers": {...}}}
        """
        from pathlib import Path

        from langchain_core.tools import StructuredTool

        from alfredo.integrations.mcp_http_wrapper import MCPWrapperGenerator
        from alfredo.tools.alfredo_tool import AlfredoTool

        # Initialize tools list if needed
        if self.tools is None:
            self.tools = []

        # Generate wrapper modules and collect instructions
        all_instructions = []

        for name, config in codeact_mcp_functions.items():
            if self.verbose:
                print(f"🔧 Generating MCP HTTP wrapper for '{name}'...")

            # Extract config
            server_url = config.get("url")
            if not server_url:
                if self.verbose:
                    print(f"⚠️  Skipping '{name}': No 'url' specified")
                continue

            headers = config.get("headers")

            try:
                # Create generator
                generator = MCPWrapperGenerator(
                    server_url=server_url,
                    name=name,
                    headers=headers,
                )

                # Initialize session
                if self.verbose:
                    print(f"   Initializing MCP session at {server_url}...")
                generator.initialize_session()

                # Fetch schema
                if self.verbose:
                    print("   Fetching tools via JSON-RPC tools/list...")
                tools_schema = generator.fetch_tools_schema()

                if self.verbose:
                    print(f"   Found {len(tools_schema)} tools")

                # Generate module in cwd
                module_path = Path(self.cwd) / f"{name}_mcp.py"
                if self.verbose:
                    print(f"   Generating module at {module_path}...")
                generator.generate_module(str(module_path))

                # Get instructions for system prompt
                instructions = generator.generate_system_instructions()
                all_instructions.append(instructions)

                if self.verbose:
                    print(f"✅ Generated wrapper for '{name}' ({len(tools_schema)} tools)\n")

            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to generate wrapper for '{name}': {e}\n")
                continue

        # Create synthetic tool with system instructions
        if all_instructions:
            combined_instructions = "\n\n".join(all_instructions)

            # Create a dummy StructuredTool (not actually callable, just for instruction delivery)
            def _dummy_mcp_http_info() -> str:
                """Synthetic tool for MCP HTTP module documentation."""
                return "This tool provides documentation for MCP HTTP wrapper modules."

            dummy_tool = StructuredTool.from_function(
                func=_dummy_mcp_http_info,
                name="mcp_http_modules_info",
                description="Documentation for MCP HTTP wrapper modules available as imports",
            )

            # Wrap as AlfredoTool with system instructions for agent node
            alfredo_tool = AlfredoTool.from_langchain(
                langchain_tool=dummy_tool,
                system_instructions={"agent": combined_instructions},
            )

            # Add to tools list
            self.tools.append(alfredo_tool)

            if self.verbose:
                print("📝 Added MCP HTTP module documentation to agent system prompt\n")

    @property
    def results(self) -> Optional[dict[str, Any]]:
        """Get the results from the last run.

        Returns:
            Final state dictionary with keys: messages, task, plan, plan_iteration,
            final_answer, is_verified. Returns None if run() hasn't been called yet.
        """
        return self._results

    def _is_mcp_tool(self, tool_name: str) -> bool:
        """Determine if a tool is an MCP tool based on its name.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is an MCP (or other external) tool, False if it's an Alfredo tool
        """
        # Check against the authoritative registry of Alfredo tools
        from alfredo.tools.registry import registry

        alfredo_tool_ids = registry.get_all_tool_ids()
        return tool_name not in alfredo_tool_ids

    def get_system_prompts(
        self,
        task: str = "Example task",
        plan: str = "Example plan",
        answer: str = "Example answer",
        verification_feedback: str = "Example feedback",
    ) -> dict[str, str]:
        """Get all system prompts used by different nodes in the agentic graph.

        Args:
            task: Example task text for formatting (default: "Example task")
            plan: Example plan text for formatting (default: "Example plan")
            answer: Example answer for verification prompt (default: "Example answer")
            verification_feedback: Example feedback for replan prompt (default: "Example feedback")

        Returns:
            Dictionary with keys: planner, agent, verifier, replan
        """
        # Get tools (create default set if none configured)
        if self.tools is None:
            from alfredo.integrations.langchain import create_alfredo_tools
            from alfredo.tools.handlers.todo import TODO_SYSTEM_INSTRUCTIONS

            tools = create_alfredo_tools(
                cwd=self.cwd,
                tool_configs={
                    "write_todo_list": TODO_SYSTEM_INSTRUCTIONS,
                    "read_todo_list": TODO_SYSTEM_INSTRUCTIONS,
                },
            )
        else:
            tools = self.tools

        return {
            "planner": get_planning_prompt(task=task, tools=tools),
            "agent": get_agent_system_prompt(task=task, plan=plan, tools=tools),
            "verifier": get_verification_prompt(task=task, answer=answer, execution_trace="", tools=tools),
            "replan": get_replan_prompt(
                task=task, previous_plan=plan, verification_feedback=verification_feedback, tools=tools
            ),
        }

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        """Get descriptions of all tools available to the agent.

        Returns:
            List of dictionaries with keys: name, description, parameters, tool_type, target_nodes
            where tool_type is either "alfredo" or "mcp", and target_nodes lists graph nodes
            that have custom instructions for this tool
        """
        from alfredo.tools.alfredo_tool import AlfredoTool

        if self.tools is None:
            # Import here to avoid circular import
            from alfredo.integrations.langchain import create_alfredo_tools
            from alfredo.tools.handlers.todo import TODO_SYSTEM_INSTRUCTIONS

            tools = create_alfredo_tools(
                cwd=self.cwd,
                tool_configs={
                    "write_todo_list": TODO_SYSTEM_INSTRUCTIONS,
                    "read_todo_list": TODO_SYSTEM_INSTRUCTIONS,
                },
            )
        else:
            tools = self.tools

        tool_descriptions = []

        for tool in tools:
            # Get underlying LangChain tool (for compatibility)
            if isinstance(tool, AlfredoTool):
                lc_tool = tool.to_langchain_tool()
                target_nodes = tool.get_target_nodes()
            else:
                lc_tool = tool
                target_nodes = []

            # Detect tool type (MCP tools often have prefixes like "bc_", "fs_", etc.)
            tool_name = getattr(lc_tool, "name", "unknown")
            tool_type = "mcp" if self._is_mcp_tool(tool_name) else "alfredo"

            # Extract basic info
            description = getattr(lc_tool, "description", "No description available")

            # Extract parameters from args_schema
            parameters = []
            if hasattr(lc_tool, "args_schema") and lc_tool.args_schema is not None:
                schema = lc_tool.args_schema
                if hasattr(schema, "model_fields"):
                    # Pydantic v2
                    for field_name, field_info in schema.model_fields.items():
                        param_info = {
                            "name": field_name,
                            "required": field_info.is_required(),
                            "description": field_info.description or "No description",
                        }
                        parameters.append(param_info)
                elif hasattr(schema, "__fields__"):
                    # Pydantic v1
                    for field_name, field_info in schema.__fields__.items():
                        param_info = {
                            "name": field_name,
                            "required": field_info.required,
                            "description": field_info.field_info.description or "No description",
                        }
                        parameters.append(param_info)

            tool_descriptions.append({
                "name": tool_name,
                "description": description,
                "parameters": parameters,
                "tool_type": tool_type,
                "target_nodes": target_nodes,
            })

        return tool_descriptions

    def _format_tool_section(self, tools: list[dict[str, Any]], section_title: str) -> list[str]:
        """Format a section of tools for display.

        Args:
            tools: List of tool dictionaries
            section_title: Title for the section (e.g., "ALFREDO TOOLS")

        Returns:
            List of formatted output lines
        """
        output_lines: list[str] = []
        if not tools:
            return output_lines

        output_lines.append("=" * 80)
        output_lines.append(section_title)
        output_lines.append("=" * 80)
        output_lines.append("")

        for tool in tools:
            output_lines.append(f"## {tool['name']}")
            output_lines.append("")
            output_lines.append(f"**Description:** {tool['description']}")
            output_lines.append("")

            # Show target nodes if present
            if tool.get("target_nodes"):
                nodes_str = ", ".join(tool["target_nodes"])
                output_lines.append(f"**Target Nodes:** {nodes_str}")
                output_lines.append("")

            if tool["parameters"]:
                output_lines.append("**Parameters:**")
                for param in tool["parameters"]:
                    required = "required" if param["required"] else "optional"
                    output_lines.append(f"  - `{param['name']}` ({required}): {param['description']}")
                output_lines.append("")
            else:
                output_lines.append("**Parameters:** None")
                output_lines.append("")

        return output_lines

    def display_tool_descriptions(self, save_to_file: bool = False) -> None:
        """Display formatted descriptions of all available tools.

        Args:
            save_to_file: If True, save output to alfredo/notes/tool_descriptions.md
        """
        tool_descriptions = self.get_tool_descriptions()

        # Group by tool type
        alfredo_tools = [t for t in tool_descriptions if t["tool_type"] == "alfredo"]
        mcp_tools = [t for t in tool_descriptions if t["tool_type"] == "mcp"]

        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("TOOL DESCRIPTIONS")
        output_lines.append("=" * 80)
        output_lines.append("")
        output_lines.append(f"Total Tools: {len(tool_descriptions)}")
        output_lines.append(f"- Alfredo Tools: {len(alfredo_tools)}")
        output_lines.append(f"- MCP Tools: {len(mcp_tools)}")
        output_lines.append("")

        # Display Alfredo tools
        output_lines.extend(self._format_tool_section(alfredo_tools, "ALFREDO TOOLS"))

        # Display MCP tools
        output_lines.extend(self._format_tool_section(mcp_tools, "MCP TOOLS"))

        output_text = "\n".join(output_lines)
        print(output_text)

        # Save to file if requested
        if save_to_file:
            from pathlib import Path

            # Determine notes directory (relative to project root or absolute)
            notes_dir = Path(self.cwd) / "alfredo" / "notes"
            if not notes_dir.exists():
                notes_dir = Path(self.cwd) / "notes"
            if not notes_dir.exists():
                notes_dir.mkdir(parents=True, exist_ok=True)

            output_path = notes_dir / "tool_descriptions.md"
            with open(output_path, "w") as f:
                f.write(output_text)
            print(f"\n💾 Tool descriptions saved to: {output_path}")

    def display_system_prompts(
        self,
        task: str = "Example task",
        plan: str = "Example plan",
        save_to_file: bool = False,
    ) -> None:
        """Display all system prompts used by different nodes in the agentic graph.

        Args:
            task: Example task text for formatting (default: "Example task")
            plan: Example plan text for formatting (default: "Example plan")
            save_to_file: If True, save output to alfredo/notes/system_prompts.md
        """
        prompts = self.get_system_prompts(task=task, plan=plan)
        tool_descriptions = self.get_tool_descriptions()

        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("AGENT SYSTEM PROMPTS")
        output_lines.append("=" * 80)
        output_lines.append("")
        output_lines.append("## Configuration")
        output_lines.append("")
        output_lines.append(f"- **Model:** {self.model_name}")
        output_lines.append(f"- **Working Directory:** {self.cwd}")
        output_lines.append(f"- **Max Context Tokens:** {self.max_context_tokens}")
        output_lines.append(f"- **Recursion Limit:** {self.recursion_limit}")
        output_lines.append(f"- **Tools Available:** {len(tool_descriptions)}")

        # Group tools by type
        alfredo_count = sum(1 for t in tool_descriptions if t["tool_type"] == "alfredo")
        mcp_count = sum(1 for t in tool_descriptions if t["tool_type"] == "mcp")
        output_lines.append(f"  - Alfredo: {alfredo_count}")
        output_lines.append(f"  - MCP: {mcp_count}")
        output_lines.append("")

        # Display each prompt
        for node_name, prompt_text in prompts.items():
            output_lines.append("=" * 80)
            output_lines.append(f"{node_name.upper()} NODE PROMPT")
            output_lines.append("=" * 80)
            output_lines.append("")
            output_lines.append(prompt_text)
            output_lines.append("")

        output_text = "\n".join(output_lines)
        print(output_text)

        # Save to file if requested
        if save_to_file:
            from pathlib import Path

            # Determine notes directory (relative to project root or absolute)
            notes_dir = Path(self.cwd) / "alfredo" / "notes"
            if not notes_dir.exists():
                notes_dir = Path(self.cwd) / "notes"
            if not notes_dir.exists():
                notes_dir.mkdir(parents=True, exist_ok=True)

            output_path = notes_dir / "system_prompts.md"
            with open(output_path, "w") as f:
                f.write(output_text)
            print(f"\n💾 System prompts saved to: {output_path}")

    def set_planner_prompt(self, template: str) -> None:
        """Set custom planner prompt.

        Note: If planning is disabled (enable_planning=False), this method will automatically
        set the agent prompt instead, since the planner node is not used.

        Args:
            template: Custom prompt string. Can be:
                - Plain text: Auto-prepended with task context, auto-appended with tool_instructions
                - Template: Must include {task} and {tool_instructions}
                  (or {task}, {plan}, {tool_instructions} if planning is disabled)

        Raises:
            ValueError: If template has placeholders but missing required ones

        Example:
            >>> agent.set_planner_prompt("Create a detailed step-by-step implementation plan.")
            >>> # Or with placeholders:
            >>> agent.set_planner_prompt("Task: {task}\\n\\nCreate a plan.\\n\\n{tool_instructions}")
        """
        # If planning is disabled, set agent prompt instead
        if not self.enable_planning:
            if self.verbose:
                print("INFO: Planning is disabled - setting agent prompt instead of planner prompt")
            self.set_agent_prompt(template)
            return

        self.prompt_templates.planner = template
        self._rebuild_graph()

    def set_agent_prompt(self, template: str) -> None:
        """Set custom agent prompt.

        If using placeholders, must include: {task}, {plan}, {tool_instructions}

        Args:
            template: Custom prompt string

        Raises:
            ValueError: If template has placeholders but missing required ones

        Example:
            >>> agent.set_agent_prompt("Execute the plan step by step.")
        """
        self.prompt_templates.agent = template
        self._rebuild_graph()

    def set_verifier_prompt(self, template: str) -> None:
        """Set custom verifier prompt.

        If using placeholders, must include: {task}, {answer}, {trace_section}, {tool_instructions}

        Args:
            template: Custom prompt string

        Raises:
            ValueError: If template has placeholders but missing required ones

        Example:
            >>> agent.set_verifier_prompt("Check if the task was completed correctly.")
        """
        self.prompt_templates.verifier = template
        self._rebuild_graph()

    def set_replan_prompt(self, template: str) -> None:
        """Set custom replan prompt.

        If using placeholders, must include: {task}, {previous_plan}, {verification_feedback}, {tool_instructions}

        Args:
            template: Custom prompt string

        Raises:
            ValueError: If template has placeholders but missing required ones

        Example:
            >>> agent.set_replan_prompt("Create an improved plan based on the feedback.")
        """
        self.prompt_templates.replan = template
        self._rebuild_graph()

    def reset_prompts(self) -> None:
        """Reset all prompts to defaults."""
        self.prompt_templates = PromptTemplates()
        self._rebuild_graph()

    def get_prompt_template(self, node_name: str) -> Optional[str]:
        """Get current custom template for a node.

        Args:
            node_name: Name of the node ("planner", "agent", "verifier", or "replan")

        Returns:
            The custom template string, or None if using default prompt

        Example:
            >>> template = agent.get_prompt_template("planner")
            >>> if template:
            ...     print(f"Custom planner template: {template}")
        """
        return getattr(self.prompt_templates, node_name, None)

    def _rebuild_graph(self) -> None:
        """Rebuild the graph with current settings (including prompt templates).

        This is called automatically when prompt templates are modified.
        """
        self.graph = create_agentic_graph(
            cwd=self.cwd,
            model_name=self.model_name,
            max_context_tokens=self.max_context_tokens,
            tools=self.tools,
            recursion_limit=self.recursion_limit,
            enable_planning=self.enable_planning,
            parse_reasoning=self.parse_reasoning,
            prompt_templates=self.prompt_templates,
            **self.model_kwargs,
        )

    def run(self, task: str) -> dict[str, Any]:
        """Run an agentic task from start to finish.

        Args:
            task: The task to accomplish

        Returns:
            Final state dictionary with results

        Raises:
            Exception: If execution fails
        """
        # Initial state
        initial_state: AgentState = {
            "messages": [],
            "task": task,
            "plan": "",
            "plan_iteration": 0,
            "max_context_tokens": self.max_context_tokens,
            "final_answer": None,
            "is_verified": False,
            "todo_list": None,
        }

        if self.verbose:
            print(f"🚀 Starting agentic task: {task}\n")

        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state, config={"recursion_limit": self.recursion_limit})
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Error during execution: {e}")
            raise
        else:
            if self.verbose:
                print("\n✅ Task completed!")
                print(f"\n📝 Final Answer:\n{final_state.get('final_answer', 'No answer provided')}")

            # Store results
            self._results = final_state
            return final_state  # type: ignore[no-any-return]

    def display_trace(self) -> None:  # noqa: C901
        """Display a formatted trace of all messages and tool calls.

        This provides a detailed view of the agent's execution, including:
        - All messages exchanged (HumanMessage, AIMessage, ToolMessage, etc.)
        - Tool calls made by the agent with their arguments
        - Tool responses
        - Reasoning content (if parse_reasoning was enabled)

        Raises:
            RuntimeError: If run() hasn't been called yet
        """
        if self._results is None:
            msg = "No execution results available. Call run() first."
            raise RuntimeError(msg)

        messages = self._results.get("messages", [])

        print("\n" + "=" * 80)
        print("EXECUTION TRACE")
        print("=" * 80)

        # Print summary stats
        print(f"\nTask: {self._results.get('task', 'N/A')}")
        print(f"Plan Iterations: {self._results.get('plan_iteration', 0)}")
        print(f"Total Messages: {len(messages)}")
        print(f"Verified: {self._results.get('is_verified', False)}")
        print(f"Final Answer: {self._results.get('final_answer', 'N/A')[:100]}...")

        print("\n" + "=" * 80)
        print("MESSAGES")
        print("=" * 80)

        # Print all messages
        for i, msg in enumerate(messages):
            print(f"\n--- Message {i + 1} ---")
            print(f"Type: {type(msg).__name__}")

            # Print reasoning content if present (AIMessage with parse_reasoning enabled)
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                reasoning = msg.additional_kwargs.get("reasoning")
                if reasoning:
                    print(f"Reasoning: {reasoning[:300]}..." if len(reasoning) > 300 else f"Reasoning: {reasoning}")

            # Print message content
            if hasattr(msg, "content"):
                content = str(msg.content)
                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "... (truncated)"
                print(f"Content: {content}")

            # Print tool calls if present (AIMessage)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"Tool Calls ({len(msg.tool_calls)}):")
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    # Detect MCP tools using consistent logic
                    prefix = "🔬 [MCP]" if self._is_mcp_tool(tool_name) else "🛠️  [Alfredo]"
                    print(f"  {prefix} {tool_name}")
                    if "args" in tc:
                        args_str = str(tc["args"])
                        if len(args_str) > 200:
                            args_str = args_str[:200] + "... (truncated)"
                        print(f"       Args: {args_str}")

            # Print tool call ID if present (ToolMessage)
            if hasattr(msg, "tool_call_id"):
                print(f"Tool Call ID: {msg.tool_call_id}")

            # Print name if present (ToolMessage)
            if hasattr(msg, "name"):
                print(f"Tool Name: {msg.name}")

        print("\n" + "=" * 80)
