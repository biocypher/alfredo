"""File operation tool handlers: read, write, and edit files."""

import re
from typing import Any

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec


class ReadFileHandler(BaseToolHandler):
    """Handler for reading file contents."""

    @property
    def tool_id(self) -> str:
        return "read_file"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Read and return the contents of a file.

        Args:
            params: Must contain 'path' - the file path to read
                   Optional: 'offset' - line number to start from (0-indexed)
                   Optional: 'limit' - maximum number of lines to read
                   Optional: 'limit_bytes' - maximum number of bytes to read

        Returns:
            ToolResult with file contents or error
        """
        try:
            self.validate_required_param(params, "path")
            file_path = self.resolve_path(params["path"])

            if not file_path.exists():
                return ToolResult.err(f"File not found: {self.get_relative_path(file_path)}")

            if not file_path.is_file():
                return ToolResult.err(f"Path is not a file: {self.get_relative_path(file_path)}")

            # Parse and validate parameters
            result = self._parse_and_validate_params(params)
            if isinstance(result, ToolResult):
                return result
            offset, limit, limit_bytes = result

            # Read file contents
            try:
                content = file_path.read_text(encoding="utf-8")

                # Handle byte-based limit
                if limit_bytes is not None:
                    return self._read_with_byte_limit(content, limit_bytes)

                # Handle line-based limit
                return self._read_with_line_limit(content, offset, limit, file_path)

            except UnicodeDecodeError:
                size = file_path.stat().st_size
                return ToolResult.err(f"File appears to be binary (size: {size} bytes). Cannot read as text.")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Error reading file: {e}")

    def _parse_and_validate_params(self, params: dict[str, Any]) -> ToolResult | tuple[int, int | None, int | None]:
        """Parse and validate offset, limit, and limit_bytes parameters.

        Returns:
            Either a ToolResult (error) or tuple of (offset, limit, limit_bytes)
        """
        limit = params.get("limit")
        limit_bytes = params.get("limit_bytes")

        # Check for mutually exclusive parameters
        if limit is not None and limit_bytes is not None:
            return ToolResult.err("Cannot use both 'limit' and 'limit_bytes' parameters")

        # Parse each parameter
        offset_result = self._parse_int_param(params.get("offset"), "offset", default=0, allow_zero=True)
        if isinstance(offset_result, ToolResult):
            return offset_result
        # offset_result is guaranteed to be int (default=0)
        offset = offset_result if offset_result is not None else 0

        limit_result = self._parse_int_param(limit, "limit", allow_zero=False) if limit else None
        if isinstance(limit_result, ToolResult):
            return limit_result

        limit_bytes_result = (
            self._parse_int_param(limit_bytes, "limit_bytes", allow_zero=False) if limit_bytes else None
        )
        if isinstance(limit_bytes_result, ToolResult):
            return limit_bytes_result

        return offset, limit_result, limit_bytes_result

    def _parse_int_param(
        self, value: Any, name: str, default: int | None = None, allow_zero: bool = False
    ) -> ToolResult | int | None:
        """Parse and validate an integer parameter.

        Args:
            value: Parameter value to parse
            name: Parameter name for error messages
            default: Default value if None
            allow_zero: Whether zero is allowed

        Returns:
            Either a ToolResult (error) or the parsed integer
        """
        if value is None:
            return default

        try:
            parsed = int(value)
        except ValueError:
            return ToolResult.err(f"Invalid {name} value: {value}")
        else:
            if allow_zero:
                if parsed < 0:
                    return ToolResult.err(f"{name.capitalize()} must be non-negative")
            elif parsed <= 0:
                return ToolResult.err(f"{name.capitalize()} must be positive")
            return parsed

    def _read_with_byte_limit(self, content: str, limit_bytes: int) -> ToolResult:
        """Read file with byte limit.

        Args:
            content: Full file content
            limit_bytes: Maximum bytes to read

        Returns:
            ToolResult with limited content
        """
        total_size = len(content.encode("utf-8"))

        if limit_bytes >= total_size:
            return ToolResult.ok(content)

        # Truncate at byte boundary, being careful with UTF-8
        byte_content = content.encode("utf-8")[:limit_bytes]
        try:
            result_content = byte_content.decode("utf-8")
        except UnicodeDecodeError:
            # Truncation split a multi-byte character
            result_content = byte_content[:-1].decode("utf-8", errors="ignore")

        # Add metadata
        actual_bytes = len(result_content.encode("utf-8"))
        metadata = f"[Showing first {actual_bytes} bytes of {total_size} total bytes]\n\n"
        return ToolResult.ok(metadata + result_content)

    def _read_with_line_limit(self, content: str, offset: int, limit: int | None, file_path: Any) -> ToolResult:
        """Read file with line-based offset and limit.

        Args:
            content: Full file content
            offset: Line offset (0-indexed)
            limit: Maximum lines to read
            file_path: Path object for error messages

        Returns:
            ToolResult with limited content
        """
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Validate offset
        if offset >= total_lines:
            return ToolResult.err(
                f"Offset {offset} exceeds total lines ({total_lines}) in file: {self.get_relative_path(file_path)}"
            )

        # Calculate line range
        end_line = (offset + limit) if limit else total_lines
        selected_lines = lines[offset:end_line]
        actual_end = min(end_line, total_lines)

        # Reconstruct content
        result_content = "".join(selected_lines)

        # Add metadata if partial read
        if offset > 0 or limit is not None:
            metadata = f"[Showing lines {offset + 1}-{actual_end} of {total_lines} total lines]\n\n"
            result_content = metadata + result_content

        return ToolResult.ok(result_content)


class WriteFileHandler(BaseToolHandler):
    """Handler for writing/creating files."""

    @property
    def tool_id(self) -> str:
        return "write_to_file"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Write content to a file (creates or overwrites).

        Args:
            params: Must contain 'path' and 'content'

        Returns:
            ToolResult indicating success or error
        """
        try:
            self.validate_required_param(params, "path")
            self.validate_required_param(params, "content")

            file_path = self.resolve_path(params["path"])
            content = params["content"]

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            file_path.write_text(content, encoding="utf-8")

            rel_path = self.get_relative_path(file_path)
            action = "Created" if not file_path.exists() else "Updated"
            return ToolResult.ok(f"{action} file: {rel_path}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Error writing file: {e}")


class ReplaceInFileHandler(BaseToolHandler):
    """Handler for applying search/replace diffs to files."""

    @property
    def tool_id(self) -> str:
        return "replace_in_file"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Apply SEARCH/REPLACE blocks to edit a file.

        Args:
            params: Must contain 'path' and 'diff'

        Returns:
            ToolResult indicating success or error
        """
        try:
            self.validate_required_param(params, "path")
            self.validate_required_param(params, "diff")

            file_path = self.resolve_path(params["path"])
            diff = params["diff"]

            if not file_path.exists():
                return ToolResult.err(f"File not found: {self.get_relative_path(file_path)}")

            # Read current content
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ToolResult.err("Cannot edit binary file")

            # Parse and apply SEARCH/REPLACE blocks
            try:
                new_content = self._apply_diff(content, diff)
            except ValueError as e:
                return ToolResult.err(f"Diff error: {e}")

            # Write back
            file_path.write_text(new_content, encoding="utf-8")

            rel_path = self.get_relative_path(file_path)
            return ToolResult.ok(f"Successfully updated file: {rel_path}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Error editing file: {e}")

    def _apply_diff(self, content: str, diff: str) -> str:
        """Parse and apply SEARCH/REPLACE blocks to content.

        The diff format is:
        ------- SEARCH
        [exact text to find]
        =======
        [text to replace with]
        +++++++ REPLACE

        Args:
            content: Original file content
            diff: Diff string with SEARCH/REPLACE blocks

        Returns:
            Modified content

        Raises:
            ValueError: If diff format is invalid or search text not found
        """
        # Parse SEARCH/REPLACE blocks
        blocks = self._parse_diff_blocks(diff)

        if not blocks:
            msg = "No valid SEARCH/REPLACE blocks found in diff"
            raise ValueError(msg)

        # Apply each block in order
        result = content
        for i, (search_text, replace_text) in enumerate(blocks):
            if search_text not in result:
                msg = f"Block {i + 1}: Search text not found in file:\n{search_text}"
                raise ValueError(msg)

            # Replace only the first occurrence
            result = result.replace(search_text, replace_text, 1)

        return result

    def _parse_diff_blocks(self, diff: str) -> list[tuple[str, str]]:
        """Parse SEARCH/REPLACE blocks from diff string.

        Args:
            diff: Diff string

        Returns:
            List of (search_text, replace_text) tuples
        """
        # Pattern to match SEARCH/REPLACE blocks
        # Looking for: ------- SEARCH\n...content...\n=======\n...content...\n+++++++ REPLACE
        pattern = r"-{7,}\s*SEARCH\s*\n(.*?)\n={7,}\s*\n(.*?)\n\+{7,}\s*REPLACE"

        matches = re.findall(pattern, diff, re.DOTALL)

        if not matches:
            # Try a simpler pattern without the trailing markers
            pattern = r"-{7,}\s*SEARCH\s*\n(.*?)\n={7,}\s*\n(.*?)(?=\n-{7,}\s*SEARCH|\Z)"
            matches = re.findall(pattern, diff, re.DOTALL)

        return [(search.rstrip(), replace.rstrip()) for search, replace in matches]


# Register tool specifications
_read_file_spec = ToolSpec(
    id="read_file",
    name="read_file",
    description=(
        "Request to read the contents of a file at the specified path. "
        "Use this when you need to examine the contents of an existing file. "
        "For large files, you can use limit_bytes to read only the first N bytes/KB, "
        "or use offset and limit to read a specific range of lines."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="path",
            required=True,
            instruction="The path of the file to read (relative to the current working directory)",
            usage="path/to/file.txt",
        ),
        ToolParameter(
            name="offset",
            required=False,
            instruction=(
                "Line number to start reading from (0-indexed). "
                "Use this to skip the first N lines of a file. "
                "Defaults to 0 (start from beginning). "
                "Cannot be used with limit_bytes."
            ),
            usage="100",
        ),
        ToolParameter(
            name="limit",
            required=False,
            instruction=(
                "Maximum number of lines to read from the offset position. "
                "Use this to peek at large files without reading the entire content. "
                "If not specified, reads until the end of the file. "
                "Cannot be used with limit_bytes."
            ),
            usage="50",
        ),
        ToolParameter(
            name="limit_bytes",
            required=False,
            instruction=(
                "Maximum number of bytes to read from the beginning of the file. "
                "Use this to preview large files by size (e.g., 50000 for 50KB). "
                "Handles UTF-8 encoding safely. "
                "Cannot be used with offset or limit."
            ),
            usage="51200",
        ),
    ],
)

_write_file_spec = ToolSpec(
    id="write_to_file",
    name="write_to_file",
    description=(
        "Request to write content to a file at the specified path. "
        "If the file exists, it will be overwritten. If it doesn't exist, it will be created. "
        "This tool will automatically create any directories needed."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="path",
            required=True,
            instruction="The path of the file to write to (relative to the current working directory)",
            usage="path/to/file.txt",
        ),
        ToolParameter(
            name="content",
            required=True,
            instruction=(
                "The content to write to the file. "
                "ALWAYS provide the COMPLETE intended content of the file, "
                "without any truncation or omissions."
            ),
            usage="Your file content here",
        ),
    ],
)

_replace_file_spec = ToolSpec(
    id="replace_in_file",
    name="replace_in_file",
    description=(
        "Request to replace sections of content in an existing file using SEARCH/REPLACE blocks. "
        "Use this when you need to make targeted changes to specific parts of a file."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="path",
            required=True,
            instruction="The path of the file to modify (relative to the current working directory)",
            usage="path/to/file.txt",
        ),
        ToolParameter(
            name="diff",
            required=True,
            instruction=(
                "One or more SEARCH/REPLACE blocks following this exact format:\n"
                "```\n"
                "------- SEARCH\n"
                "[exact content to find]\n"
                "=======\n"
                "[new content to replace with]\n"
                "+++++++ REPLACE\n"
                "```\n"
                "Critical rules:\n"
                "1. SEARCH content must match EXACTLY (character-for-character)\n"
                "2. Each block replaces only the FIRST occurrence\n"
                "3. To delete code, use an empty REPLACE section\n"
                "4. List multiple blocks in the order they appear in the file"
            ),
            usage="Search and replace blocks here",
        ),
    ],
)

# Register with the global registry
registry.register_spec(_read_file_spec)
registry.register_spec(_write_file_spec)
registry.register_spec(_replace_file_spec)

registry.register_handler("read_file", ReadFileHandler)
registry.register_handler("write_to_file", WriteFileHandler)
registry.register_handler("replace_in_file", ReplaceInFileHandler)
