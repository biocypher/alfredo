"""Directory exploration agent for generating markdown reports.

This module provides a specialized agent that explores directories and generates
comprehensive markdown reports about their contents, with smart file size handling
and data analysis capabilities.
"""

from pathlib import Path
from typing import Any, ClassVar, Optional

from alfredo.agentic.agent import Agent


class ExplorationAgent:
    """Agent for exploring directories and generating markdown reports.

    This agent intelligently explores directories, analyzing file contents based on type
    and size, and generates a comprehensive markdown report. It handles:
    - Text files (code, configs, docs) with size-aware reading
    - Data files (CSV, Excel, Parquet, HDF5, JSON) with pandas analysis
    - Binary files (images, archives) with metadata only

    Example:
        >>> agent = ExplorationAgent(cwd="./my_project")
        >>> report = agent.explore()
        >>> print(report)

        >>> # With context to steer exploration
        >>> agent = ExplorationAgent(
        ...     cwd="./data_pipeline",
        ...     context_prompt="Focus on data schemas and quality checks"
        ... )
        >>> report = agent.explore()
    """

    # File size thresholds for reading strategy
    SIZE_SMALL: ClassVar[int] = 10_000  # 10KB - read fully
    SIZE_MEDIUM: ClassVar[int] = 100_000  # 100KB - read with limit
    SIZE_LARGE: ClassVar[int] = 1_000_000  # 1MB - just peek

    # Line limits for different size categories
    LINES_MEDIUM: ClassVar[int] = 100  # Lines to read for medium files
    LINES_LARGE: ClassVar[int] = 50  # Lines to read for large files

    # Data file extensions that support analysis
    DATA_EXTENSIONS: ClassVar[set[str]] = {
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".h5",
        ".hdf5",
        ".json",
        ".jsonl",
        ".feather",
    }

    def __init__(
        self,
        cwd: str = ".",
        context_prompt: Optional[str] = None,
        model_name: str = "gpt-4.1-mini",
        max_file_size_bytes: int = 100_000,
        preview_kb: Optional[int] = None,
        preview_lines: Optional[int] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the exploration agent.

        Args:
            cwd: Working directory to explore (default: ".")
            context_prompt: Optional prompt to contextualize and steer the exploration
                (e.g., "Focus on data quality and validation logic")
            model_name: Name of the model to use (default: "gpt-4.1-mini")
            max_file_size_bytes: File size threshold for using limited reading (default: 100KB)
            preview_kb: Size in KB to preview for large files (default: 50KB if preview_lines not set)
            preview_lines: Number of lines to preview for large files (alternative to preview_kb)
            output_path: Path to save the report. If None, saves to notes/exploration_report.md
            verbose: Whether to print progress updates
            **kwargs: Additional keyword arguments to pass to the model
                (e.g., temperature, base_url, api_key)
        """
        self.cwd = Path(cwd).resolve()
        self.context_prompt = context_prompt
        self.model_name = model_name
        self.max_file_size_bytes = max_file_size_bytes

        # Support both preview_kb (bytes) and preview_lines (lines)
        # If neither is set, default to 50KB
        if preview_kb is None and preview_lines is None:
            preview_kb = 50

        self.preview_kb = preview_kb
        self.preview_lines = preview_lines
        self.preview_bytes = preview_kb * 1024 if preview_kb else None

        self.verbose = verbose
        self.model_kwargs = kwargs

        # Determine output path
        if output_path is None:
            notes_dir = self.cwd / "notes"
            notes_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = notes_dir / "exploration_report.md"
        else:
            self.output_path = Path(output_path)

        # Create the underlying agent with custom system prompt
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create and configure the underlying Agent instance with custom prompts.

        Returns:
            Configured Agent instance
        """
        agent = Agent(
            cwd=str(self.cwd),
            model_name=self.model_name,
            verbose=self.verbose,
            **self.model_kwargs,
        )

        # Set custom planner prompt to guide exploration
        planner_prompt = self._build_planner_prompt()
        agent.set_planner_prompt(planner_prompt)

        return agent

    def _build_planner_prompt(self) -> str:
        """Build custom planner prompt for directory exploration.

        Returns:
            Custom prompt string
        """
        base_prompt = f"""You are an expert at exploring and documenting directory structures.

Task: {{task}}

## Your Mission

Create a comprehensive directory exploration plan that will result in a detailed markdown report.

## Exploration Strategy

1. **Start with Directory Listing**
   - Use list_files with recursive=true to understand the full structure
   - Note total files, directories, and overall size

2. **Categorize Files**
   Group files by type:
   - Source code (.py, .r, .js, .ts, .java, .go, etc.)
   - Data files (.csv, .xlsx, .parquet, .h5, .json, etc.)
   - Configuration (.json, .yaml, .toml, .env, .ini)
   - Documentation (.md, .txt, .rst)
   - Binary/Other (images, archives, executables)

3. **Smart File Reading**
   For each file, check its size from list_files output:
   - Small (<{self.max_file_size_bytes} bytes): Read fully with read_file
   - Large (>{self.max_file_size_bytes} bytes): Use read_file with {"limit=" + str(self.preview_lines)} to peek

4. **Data File Analysis**
   For data files (CSV, Excel, Parquet, HDF5, JSON):
   a. First peek at the file structure with read_file (if text format)
   b. Write a Python analysis script to a temp file (e.g., /tmp/analyze_data.py)
   c. The script should:
      - Load the data with pandas (use appropriate reader: read_csv, read_excel, read_parquet, read_hdf, read_json)
      - Print shape, columns, dtypes
      - Show head() and describe()
      - Check for missing values
   d. Execute the script with execute_command: python /tmp/analyze_data.py
   e. Summarize and include the analysis results in the report

5. **Generate Markdown Report**
   Structure the final report as:
   ```markdown
   # Directory Exploration Report: {self.cwd.name}

   ## Overview
   - Total files: X
   - Total directories: Y
   - Total size: Z MB
   - Explored on: [date]

   ## Directory Structure
   [Hierarchical tree view]

   ## Files by Category

   ### Source Code
   [File listings with brief descriptions]

   ### Data Files
   [Detailed analysis for each with pandas output]

   ### Configuration
   [Key configs and settings]

   ### Documentation
   [List with summaries]

   ### Other Files
   [Binary files, etc.]

   ## Summary
   [Key insights and observations]
   ```

6. **Save Report**
   - Write the final report to: {self.output_path}
   - Call attempt_completion with the report content
"""

        # Add context-specific guidance if provided
        if self.context_prompt:
            base_prompt += f"""

## Additional Context

{self.context_prompt}

Use this context to focus your exploration and tailor the report accordingly.
"""

        base_prompt += """

{{tool_instructions}}

**IMPORTANT**:
- Call attempt_completion when the report is written to file
- Include the full markdown content in attempt_completion
"""

        return base_prompt

    def explore(self) -> str:
        """Explore the directory and generate a markdown report.

        Returns:
            The markdown report content as a string

        Raises:
            RuntimeError: If exploration fails
        """
        task = f"Explore the directory '{self.cwd}' and generate a comprehensive markdown exploration report."

        if self.verbose:
            print(f"\n🔍 Exploring directory: {self.cwd}")
            print(f"📝 Report will be saved to: {self.output_path}\n")

        # Run the agent
        result = self.agent.run(task)

        # Extract the final report from results
        final_answer = result.get("final_answer")

        if final_answer is None:
            msg = "Exploration failed: No final answer produced"
            raise RuntimeError(msg)

        if self.verbose:
            print("\n✅ Exploration complete!")
            print(f"📄 Report saved to: {self.output_path}")

        return final_answer  # type: ignore[no-any-return]

    def display_trace(self) -> None:
        """Display detailed execution trace of the exploration.

        Useful for debugging or understanding what the agent did.
        """
        self.agent.display_trace()
