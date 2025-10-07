"""File discovery tool handlers: list files and search content."""

import re
from pathlib import Path
from typing import Any

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec


class ListFilesHandler(BaseToolHandler):
    """Handler for listing directory contents."""

    @property
    def tool_id(self) -> str:
        return "list_files"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """List files and directories in the specified path.

        Args:
            params: Must contain 'path'. Optional: 'recursive' (bool)

        Returns:
            ToolResult with file listing or error
        """
        try:
            self.validate_required_param(params, "path")
            dir_path = self.resolve_path(params["path"])
            recursive = params.get("recursive", "false").lower() == "true"

            if not dir_path.exists():
                return ToolResult.err(f"Directory not found: {self.get_relative_path(dir_path)}")

            if not dir_path.is_dir():
                return ToolResult.err(f"Path is not a directory: {self.get_relative_path(dir_path)}")

            # List files
            try:
                items = self._list_recursive(dir_path) if recursive else self._list_top_level(dir_path)

                if not items:
                    return ToolResult.ok(f"Directory is empty: {self.get_relative_path(dir_path)}")

                output = "\n".join(items)
                return ToolResult.ok(f"Contents of {self.get_relative_path(dir_path)}:\n\n{output}")

            except Exception as e:
                return ToolResult.err(f"Error listing directory: {e}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Unexpected error: {e}")

    def _list_top_level(self, dir_path: Path) -> list[str]:
        """List only top-level contents."""
        items = []
        for item in sorted(dir_path.iterdir()):
            rel_path = self.get_relative_path(item)
            if item.is_dir():
                items.append(f"[DIR]  {rel_path}/")
            else:
                size = item.stat().st_size
                items.append(f"[FILE] {rel_path} ({size} bytes)")
        return items

    def _list_recursive(self, dir_path: Path) -> list[str]:
        """List all contents recursively."""
        items = []
        for item in sorted(dir_path.rglob("*")):
            rel_path = self.get_relative_path(item)
            if item.is_dir():
                items.append(f"[DIR]  {rel_path}/")
            else:
                size = item.stat().st_size
                items.append(f"[FILE] {rel_path} ({size} bytes)")
        return items


class SearchFilesHandler(BaseToolHandler):
    """Handler for searching file contents with regex."""

    @property
    def tool_id(self) -> str:
        return "search_files"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Search for a regex pattern across files in a directory.

        Args:
            params: Must contain 'path' and 'regex'. Optional: 'file_pattern' (glob)

        Returns:
            ToolResult with search results or error
        """
        try:
            self.validate_required_param(params, "path")
            self.validate_required_param(params, "regex")

            dir_path = self.resolve_path(params["path"])
            pattern = params["regex"]
            file_pattern = params.get("file_pattern", "*")

            if not dir_path.exists():
                return ToolResult.err(f"Directory not found: {self.get_relative_path(dir_path)}")

            if not dir_path.is_dir():
                return ToolResult.err(f"Path is not a directory: {self.get_relative_path(dir_path)}")

            # Compile regex
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return ToolResult.err(f"Invalid regex pattern: {e}")

            # Search files
            try:
                matches = self._search_directory(dir_path, regex, file_pattern)

                if not matches:
                    return ToolResult.ok(
                        f"No matches found for pattern '{pattern}' in {self.get_relative_path(dir_path)}"
                    )

                output = self._format_matches(matches)
                return ToolResult.ok(output)

            except Exception as e:
                return ToolResult.err(f"Error searching files: {e}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Unexpected error: {e}")

    def _search_directory(self, dir_path: Path, regex: re.Pattern, file_pattern: str) -> list[tuple[Path, int, str]]:
        """Search all matching files in directory.

        Returns:
            List of (file_path, line_number, line_content) tuples
        """
        matches = []

        for file_path in dir_path.rglob(file_pattern):
            if not file_path.is_file():
                continue

            # Skip binary files (basic check)
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            # Search line by line
            for line_num, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append((file_path, line_num, line))

        return matches

    def _format_matches(self, matches: list[tuple[Path, int, str]]) -> str:
        """Format search results for display."""
        lines = [f"Found {len(matches)} match(es):\n"]

        current_file = None
        for file_path, line_num, line in matches:
            rel_path = self.get_relative_path(file_path)

            if rel_path != current_file:
                if current_file is not None:
                    lines.append("")  # Blank line between files
                lines.append(f"{rel_path}:")
                current_file = rel_path

            lines.append(f"  {line_num}: {line.rstrip()}")

        return "\n".join(lines)


# Register tool specifications
_list_files_spec = ToolSpec(
    id="list_files",
    name="list_files",
    description=(
        "Request to list files and directories within the specified directory. "
        "If recursive is true, it will list all files and directories recursively. "
        "If recursive is false or not provided, it will only list the top-level contents."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="path",
            required=True,
            instruction="The path of the directory to list contents for (relative to the current working directory)",
            usage="path/to/directory",
        ),
        ToolParameter(
            name="recursive",
            required=False,
            instruction="Whether to list files recursively. Use 'true' for recursive listing, 'false' or omit for top-level only.",
            usage="true or false",
        ),
    ],
)

_search_files_spec = ToolSpec(
    id="search_files",
    name="search_files",
    description=(
        "Request to perform a regex search across files in a specified directory. "
        "This tool searches for patterns or specific content across multiple files, "
        "displaying each match with its line number and context."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="path",
            required=True,
            instruction="The path of the directory to search in (relative to the current working directory). This directory will be recursively searched.",
            usage="path/to/directory",
        ),
        ToolParameter(
            name="regex",
            required=True,
            instruction="The regular expression pattern to search for. Uses Python regex syntax.",
            usage="your pattern here",
        ),
        ToolParameter(
            name="file_pattern",
            required=False,
            instruction="Glob pattern to filter files (e.g., '*.py' for Python files). If not provided, searches all files.",
            usage="*.py",
        ),
    ],
)

registry.register_spec(_list_files_spec)
registry.register_spec(_search_files_spec)

registry.register_handler("list_files", ListFilesHandler)
registry.register_handler("search_files", SearchFilesHandler)
