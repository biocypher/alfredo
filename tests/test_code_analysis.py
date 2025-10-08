"""Tests for code analysis tool handler."""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from alfredo.tools.handlers.code_analysis import ListCodeDefinitionNamesHandler


@pytest.fixture
def temp_code_dir() -> Any:
    """Create a temporary directory with sample code files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        # Create sample Python file
        (temp_path / "sample.py").write_text(
            """
def hello_world():
    return "Hello, world!"

class MyClass:
    def method_one(self):
        pass

    def method_two(self):
        pass

async def async_function():
    pass
"""
        )

        # Create sample JavaScript file
        (temp_path / "sample.js").write_text(
            """
function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}

const myVar = 42;
"""
        )

        # Create sample TypeScript file
        (temp_path / "sample.ts").write_text(
            """
interface User {
    name: string;
    age: number;
}

type Status = 'active' | 'inactive';

class UserManager {
    getUser(id: number): User {
        return { name: 'Test', age: 30 };
    }
}

function processUser(user: User): void {
    console.log(user.name);
}
"""
        )

        yield temp_path


def test_list_code_definitions_python(temp_code_dir: Path) -> None:
    """Test listing code definitions in Python files."""
    handler = ListCodeDefinitionNamesHandler(cwd=str(temp_code_dir))

    result = handler.execute({"path": "."})
    assert result is not None
    assert result.success
    assert "sample.py" in result.output
    assert "hello_world" in result.output
    assert "MyClass" in result.output
    assert "async_function" in result.output


def test_list_code_definitions_javascript(temp_code_dir: Path) -> None:
    """Test listing code definitions in JavaScript files."""
    handler = ListCodeDefinitionNamesHandler(cwd=str(temp_code_dir))

    result = handler.execute({"path": "."})
    assert result is not None
    assert result.success
    assert "sample.js" in result.output
    assert "greet" in result.output
    assert "Calculator" in result.output


def test_list_code_definitions_typescript(temp_code_dir: Path) -> None:
    """Test listing code definitions in TypeScript files."""
    handler = ListCodeDefinitionNamesHandler(cwd=str(temp_code_dir))

    result = handler.execute({"path": "."})
    assert result is not None
    assert result.success
    assert "sample.ts" in result.output
    assert "User" in result.output
    assert "Status" in result.output
    assert "UserManager" in result.output
    assert "processUser" in result.output


def test_list_code_definitions_nonexistent_path() -> None:
    """Test error handling for nonexistent path."""
    handler = ListCodeDefinitionNamesHandler()

    result = handler.execute({"path": "/nonexistent/path"})
    assert result is not None
    assert not result.success
    assert result.error is not None
    assert "not found" in result.error.lower()


def test_list_code_definitions_missing_path() -> None:
    """Test error handling for missing path parameter."""
    handler = ListCodeDefinitionNamesHandler()

    result = handler.execute({})
    assert result is not None
    assert not result.success
    assert result.error is not None
    assert "required" in result.error.lower()


def test_list_code_definitions_file_not_directory(temp_code_dir: Path) -> None:
    """Test error handling when path is a file, not a directory."""
    handler = ListCodeDefinitionNamesHandler(cwd=str(temp_code_dir))

    result = handler.execute({"path": "sample.py"})
    assert result is not None
    assert not result.success
    assert result.error is not None
    assert "not a directory" in result.error.lower()


def test_list_code_definitions_empty_directory() -> None:
    """Test handling of directory with no source files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = ListCodeDefinitionNamesHandler(cwd=tmpdir)

        result = handler.execute({"path": "."})
        assert result is not None
        assert result.success
        assert "No definitions found" in result.output


def test_list_code_definitions_subdirectories(temp_code_dir: Path) -> None:
    """Test that subdirectories are scanned recursively."""
    # Create a subdirectory with a Python file
    subdir = temp_code_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.py").write_text(
        """
def nested_function():
    pass

class NestedClass:
    pass
"""
    )

    handler = ListCodeDefinitionNamesHandler(cwd=str(temp_code_dir))

    result = handler.execute({"path": "."})
    assert result is not None
    assert result.success
    assert "subdir" in result.output or "nested.py" in result.output
    assert "nested_function" in result.output
    assert "NestedClass" in result.output


def test_list_code_definitions_includes_line_numbers(temp_code_dir: Path) -> None:
    """Test that output includes line numbers."""
    handler = ListCodeDefinitionNamesHandler(cwd=str(temp_code_dir))

    result = handler.execute({"path": "."})
    assert result is not None
    assert result.success
    assert "line" in result.output.lower()
    # Check for line number format like "(line 2)" or similar
    assert "line " in result.output or "Line " in result.output
