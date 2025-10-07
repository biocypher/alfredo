"""Code analysis tool handlers: parse source code for definitions."""

import os
from pathlib import Path
from typing import Any

try:
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_python as ts_python
    import tree_sitter_typescript as ts_typescript
    from tree_sitter import Language, Parser
except ImportError:
    ts_python = None
    ts_javascript = None
    ts_typescript = None
    Parser = None
    Language = None

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec

# Language configuration: file extensions -> (tree-sitter language, node types to extract)
LANGUAGE_CONFIG: dict[str, tuple[Any, list[str]]] = {}

# Definition node types for each language
DEFINITION_NODE_TYPES: dict[str, list[str]] = {
    "python": ["class_definition", "function_definition"],
    "javascript": [
        "class_declaration",
        "function_declaration",
        "method_definition",
        "variable_declarator",
        "arrow_function",
    ],
    "typescript": [
        "class_declaration",
        "function_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    ],
}


def _initialize_languages() -> None:
    """Initialize tree-sitter languages if available."""
    global LANGUAGE_CONFIG
    if not Parser or not Language:
        return

    try:
        if ts_python:
            LANGUAGE_CONFIG.update({
                ".py": (Language(ts_python.language()), DEFINITION_NODE_TYPES["python"]),
            })
    except Exception:  # noqa: S110
        pass  # Silently skip if language initialization fails

    try:
        if ts_javascript:
            js_lang = Language(ts_javascript.language())
            LANGUAGE_CONFIG.update({
                ".js": (js_lang, DEFINITION_NODE_TYPES["javascript"]),
                ".jsx": (js_lang, DEFINITION_NODE_TYPES["javascript"]),
                ".mjs": (js_lang, DEFINITION_NODE_TYPES["javascript"]),
            })
    except Exception:  # noqa: S110
        pass  # Silently skip if language initialization fails

    try:
        if ts_typescript:
            ts_lang = Language(ts_typescript.language_typescript())
            tsx_lang = Language(ts_typescript.language_tsx())
            LANGUAGE_CONFIG.update({
                ".ts": (ts_lang, DEFINITION_NODE_TYPES["typescript"]),
                ".tsx": (tsx_lang, DEFINITION_NODE_TYPES["typescript"]),
            })
    except Exception:  # noqa: S110
        pass  # Silently skip if language initialization fails


# Initialize on module load
_initialize_languages()


class ListCodeDefinitionNamesHandler(BaseToolHandler):
    """Handler for listing code definition names in source files."""

    @property
    def tool_id(self) -> str:
        return "list_code_definition_names"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """List top-level code definitions (classes, functions, etc.) in a directory.

        Args:
            params: Must contain 'path' - the directory path to analyze

        Returns:
            ToolResult with formatted list of definitions or error
        """
        try:
            self.validate_required_param(params, "path")
            dir_path = self.resolve_path(params["path"])

            if not dir_path.exists():
                return ToolResult.err(f"Path not found: {self.get_relative_path(dir_path)}")

            if not dir_path.is_dir():
                return ToolResult.err(f"Path is not a directory: {self.get_relative_path(dir_path)}")

            # Check if tree-sitter is available
            if not Parser or not LANGUAGE_CONFIG:
                return ToolResult.err(
                    "Tree-sitter not available or no language parsers loaded. "
                    "Please install tree-sitter and language packages."
                )

            # Scan directory for source files
            definitions = self._scan_directory(dir_path)

            if not definitions:
                return ToolResult.ok("No definitions found in source files.")

            # Format output
            output = self._format_definitions(definitions, dir_path)
            return ToolResult.ok(output)

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Error analyzing code: {e}")

    def _scan_directory(self, dir_path: Path) -> dict[str, list[tuple[str, int]]]:
        """Scan directory for source files and extract definitions.

        Args:
            dir_path: Directory to scan

        Returns:
            Dictionary mapping file paths to list of (definition_name, line_number) tuples
        """
        definitions: dict[str, list[tuple[str, int]]] = {}

        for root, _dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext in LANGUAGE_CONFIG:
                    file_defs = self._parse_file(file_path, ext)
                    if file_defs:
                        rel_path = str(file_path.relative_to(dir_path))
                        definitions[rel_path] = file_defs

        return definitions

    def _parse_file(self, file_path: Path, ext: str) -> list[tuple[str, int]]:
        """Parse a single file and extract definitions.

        Args:
            file_path: Path to the file
            ext: File extension

        Returns:
            List of (definition_name, line_number) tuples
        """
        try:
            language, node_types = LANGUAGE_CONFIG[ext]

            # Read file
            source_code = file_path.read_bytes()

            # Parse with tree-sitter
            parser = Parser(language)
            tree = parser.parse(source_code)

            # Extract definitions by walking the tree
            definitions = self._find_definitions(tree.root_node, node_types)

            return sorted(set(definitions), key=lambda x: x[1])  # Sort by line number, remove duplicates

        except Exception:
            return []

    def _find_definitions(self, node: Any, node_types: list[str]) -> list[tuple[str, int]]:
        """Recursively find definition nodes in the syntax tree.

        Args:
            node: Current tree node
            node_types: List of node types to look for

        Returns:
            List of (definition_name, line_number) tuples
        """
        definitions = []

        if node.type in node_types:
            # Find the name of this definition
            name = self._extract_name(node)
            if name:
                line_num = node.start_point[0] + 1  # Convert to 1-indexed
                definitions.append((name, line_num))

        # Recursively search children
        for child in node.children:
            definitions.extend(self._find_definitions(child, node_types))

        return definitions

    def _extract_name(self, node: Any) -> str | None:
        """Extract the name from a definition node.

        Args:
            node: Definition node

        Returns:
            Name string or None if not found
        """
        # Look for identifier, type_identifier, or property_identifier children
        for child in node.children:
            if child.type in ["identifier", "type_identifier", "property_identifier"]:
                text = child.text
                return text.decode("utf-8") if isinstance(text, bytes) else str(text)

        return None

    def _format_definitions(self, definitions: dict[str, list[tuple[str, int]]], base_path: Path) -> str:
        """Format definitions into readable output.

        Args:
            definitions: Dictionary of file paths to definitions
            base_path: Base directory path for relative paths

        Returns:
            Formatted string output
        """
        lines = [f"Code definitions in {self.get_relative_path(base_path)}:\n"]

        for file_path, defs in sorted(definitions.items()):
            lines.append(f"\n{file_path}:")
            for name, line_num in defs:
                lines.append(f"  - {name} (line {line_num})")

        return "\n".join(lines)


# Register the tool
spec = ToolSpec(
    id="list_code_definition_names",
    name="list_code_definition_names",
    description=(
        "List definition names (classes, functions, methods, etc.) in source code files "
        "at the top level of the specified directory. This provides insights into codebase "
        "structure and important constructs. Supports Python, JavaScript, TypeScript."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="path",
            required=True,
            instruction="The directory path (relative to working directory) to analyze for code definitions",
            usage="src/",
        ),
    ],
)

registry.register_spec(spec)
registry.register_handler("list_code_definition_names", ListCodeDefinitionNamesHandler)
