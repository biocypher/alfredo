"""Tests for MCP HTTP wrapper generator."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from alfredo.integrations.mcp_http_wrapper import MCPWrapperGenerator


@pytest.fixture
def sample_tools_schema() -> list[dict]:
    """Sample tools schema for testing."""
    return [
        {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "search_files",
            "description": "Search for files matching a pattern",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search is case sensitive",
                    },
                },
                "required": ["pattern"],
            },
        },
    ]


@pytest.fixture
def generator() -> MCPWrapperGenerator:
    """Create a basic generator for testing."""
    return MCPWrapperGenerator(
        server_url="http://localhost:8000",
        name="testtools",
        headers={"Authorization": "Bearer test-token"},
    )


class TestMCPWrapperGenerator:
    """Test MCPWrapperGenerator class."""

    def test_init(self) -> None:
        """Test initialization."""
        gen = MCPWrapperGenerator(
            server_url="http://localhost:8000",
            name="mytools",
            headers={"X-API-Key": "key"},
        )

        assert gen.server_url == "http://localhost:8000"
        assert gen.name == "mytools"
        assert gen.headers == {"X-API-Key": "key"}
        assert gen.tools_schema == []

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is removed from URL."""
        gen = MCPWrapperGenerator(
            server_url="http://localhost:8000/",
            name="mytools",
        )

        assert gen.server_url == "http://localhost:8000"

    def test_init_without_headers(self) -> None:
        """Test initialization without headers."""
        gen = MCPWrapperGenerator(
            server_url="http://localhost:8000",
            name="mytools",
        )

        assert gen.headers == {}

    def test_type_mapping(self, generator: MCPWrapperGenerator) -> None:
        """Test JSON Schema to Python type mapping."""
        assert generator._map_json_type_to_python("string") == "str"
        assert generator._map_json_type_to_python("integer") == "int"
        assert generator._map_json_type_to_python("number") == "float"
        assert generator._map_json_type_to_python("boolean") == "bool"
        assert generator._map_json_type_to_python("array") == "List[Any]"
        assert generator._map_json_type_to_python("object") == "Dict[str, Any]"
        assert generator._map_json_type_to_python("null") == "None"
        assert generator._map_json_type_to_python("unknown") == "Any"

    def test_type_mapping_optional(self, generator: MCPWrapperGenerator) -> None:
        """Test optional type mapping."""
        assert generator._map_json_type_to_python("string", is_optional=True) == "Optional[str]"
        assert generator._map_json_type_to_python("integer", is_optional=True) == "Optional[int]"
        assert generator._map_json_type_to_python("array", is_optional=True) == "Optional[List[Any]]"

    @patch("requests.post")
    def test_fetch_tools_schema(
        self, mock_post: Mock, generator: MCPWrapperGenerator, sample_tools_schema: list[dict]
    ) -> None:
        """Test fetching tools schema via JSON-RPC."""
        # Mock session initialization (SSE format with session ID)
        init_response = Mock()
        init_response.headers = {"Mcp-Session-Id": "test-session-123", "Content-Type": "text/event-stream"}
        init_response.raise_for_status = Mock()
        init_response.iter_lines = Mock(return_value=['data: {"result": {"protocolVersion": "2024-11-05"}}'])

        # Mock tools/list response (SSE format)
        tools_response = Mock()
        tools_response.headers = {"Content-Type": "text/event-stream"}
        tools_response.raise_for_status = Mock()
        tools_response.iter_lines = Mock(
            return_value=[f'data: {{"result": {{"tools": {json.dumps(sample_tools_schema)}}}}}']
        )

        # Mock notification response
        notif_response = Mock()
        notif_response.raise_for_status = Mock()

        # Setup mock sequence: initialize, notification, tools/list
        mock_post.side_effect = [init_response, notif_response, tools_response]

        # Fetch schema (triggers session init automatically)
        result = generator.fetch_tools_schema()

        # Verify
        assert result == sample_tools_schema
        assert generator.tools_schema == sample_tools_schema
        assert generator.session_id == "test-session-123"
        assert mock_post.call_count == 3  # init + notification + tools/list

    @patch("requests.post")
    def test_fetch_tools_schema_http_error(self, mock_post: Mock, generator: MCPWrapperGenerator) -> None:
        """Test HTTP error handling during session initialization."""
        import requests

        # Session initialization fails
        mock_post.side_effect = requests.RequestException("Connection failed")

        with pytest.raises(requests.RequestException, match="Failed to initialize MCP session"):
            generator.fetch_tools_schema()

    @patch("requests.post")
    def test_fetch_tools_schema_invalid_json(self, mock_post: Mock, generator: MCPWrapperGenerator) -> None:
        """Test invalid JSON response handling."""
        # Mock session initialization (SSE format with session ID)
        init_response = Mock()
        init_response.headers = {"Mcp-Session-Id": "test-session-123", "Content-Type": "text/event-stream"}
        init_response.raise_for_status = Mock()
        init_response.iter_lines = Mock(return_value=['data: {"result": {"protocolVersion": "2024-11-05"}}'])

        # Mock notification response
        notif_response = Mock()
        notif_response.raise_for_status = Mock()

        # Mock tools/list with invalid SSE data
        tools_response = Mock()
        tools_response.headers = {"Content-Type": "text/event-stream"}
        tools_response.raise_for_status = Mock()
        tools_response.iter_lines = Mock(return_value=["data: invalid json"])

        mock_post.side_effect = [init_response, notif_response, tools_response]

        with pytest.raises(ValueError, match="Failed to parse SSE data"):
            generator.fetch_tools_schema()

    @patch("requests.post")
    def test_fetch_tools_schema_jsonrpc_error(self, mock_post: Mock, generator: MCPWrapperGenerator) -> None:
        """Test JSON-RPC error response handling."""
        # Mock session initialization (SSE format with session ID)
        init_response = Mock()
        init_response.headers = {"Mcp-Session-Id": "test-session-123", "Content-Type": "text/event-stream"}
        init_response.raise_for_status = Mock()
        init_response.iter_lines = Mock(return_value=['data: {"result": {"protocolVersion": "2024-11-05"}}'])

        # Mock notification response
        notif_response = Mock()
        notif_response.raise_for_status = Mock()

        # Mock tools/list with JSON-RPC error
        tools_response = Mock()
        tools_response.headers = {"Content-Type": "text/event-stream"}
        tools_response.raise_for_status = Mock()
        error_data = {"error": {"code": -32600, "message": "Invalid Request"}}
        tools_response.iter_lines = Mock(return_value=[f"data: {json.dumps(error_data)}"])

        mock_post.side_effect = [init_response, notif_response, tools_response]

        with pytest.raises(ValueError, match="JSON-RPC error: -32600"):
            generator.fetch_tools_schema()

    def test_generate_function_code_required_params(
        self, generator: MCPWrapperGenerator, sample_tools_schema: list[dict]
    ) -> None:
        """Test function code generation with required parameters."""
        generator.tools_schema = sample_tools_schema

        # Generate function for read_file (has required param)
        code = generator._generate_function_code(sample_tools_schema[0])

        assert "def read_file(path: str) -> Dict[str, Any]:" in code
        assert "Read a file from the filesystem" in code
        # Check for JSON-RPC format
        assert '"jsonrpc": "2.0"' in code
        assert '"method": "tools/call"' in code
        assert '"name": "read_file"' in code
        assert '"path": path,' in code
        assert "requests.post" in code
        assert "_ensure_session()" in code

    def test_generate_function_code_optional_params(
        self, generator: MCPWrapperGenerator, sample_tools_schema: list[dict]
    ) -> None:
        """Test function code generation with optional parameters."""
        generator.tools_schema = sample_tools_schema

        # Generate function for search_files (has optional params)
        code = generator._generate_function_code(sample_tools_schema[2])

        assert "def search_files(pattern: str, max_results: Optional[int] = None" in code
        assert "case_sensitive: Optional[bool] = None" in code
        assert "if max_results is not None:" in code
        assert "if case_sensitive is not None:" in code

    def test_generate_module_header(self, generator: MCPWrapperGenerator) -> None:
        """Test module header generation."""
        header = generator._generate_module_header()

        assert "Auto-generated MCP HTTP wrapper: testtools" in header
        assert "http://localhost:8000" in header
        assert "import json" in header
        assert "import requests" in header
        assert "from typing import Dict, Any, Optional, List" in header
        assert 'SERVER_URL = "http://localhost:8000"' in header
        assert '"Authorization": "Bearer test-token"' in header
        # Check for session management code
        assert "SESSION_ID: Optional[str] = None" in header
        assert "def _get_next_request_id()" in header
        assert "def _initialize_session()" in header
        assert "def _ensure_session()" in header
        assert '"jsonrpc": "2.0"' in header
        assert '"method": "initialize"' in header
        assert '"method": "notifications/initialized"' in header

    def test_generate_module(self, generator: MCPWrapperGenerator, sample_tools_schema: list[dict]) -> None:
        """Test complete module generation."""
        generator.tools_schema = sample_tools_schema

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            generator.generate_module(output_path)

            # Read generated module
            with open(output_path) as f:
                content = f.read()

            # Verify module structure
            assert "Auto-generated MCP HTTP wrapper: testtools" in content
            assert "def read_file(path: str)" in content
            assert "def write_file(path: str, content: str)" in content
            assert "def search_files(pattern: str" in content

        finally:
            Path(output_path).unlink()

    def test_generate_module_without_schema(self, generator: MCPWrapperGenerator) -> None:
        """Test module generation without fetching schema first."""
        with pytest.raises(ValueError, match="No tools schema available"):
            generator.generate_module("output.py")

    def test_get_module_info(self, generator: MCPWrapperGenerator, sample_tools_schema: list[dict]) -> None:
        """Test module info retrieval."""
        generator.tools_schema = sample_tools_schema

        info = generator.get_module_info()

        assert info["module_name"] == "testtools_mcp"
        assert info["server_url"] == "http://localhost:8000"
        assert len(info["functions"]) == 3

        # Check first function
        read_file_info = info["functions"][0]
        assert read_file_info["name"] == "read_file"
        assert "path: str" in read_file_info["signature"]
        assert read_file_info["description"] == "Read a file from the filesystem"

    def test_get_module_info_without_schema(self, generator: MCPWrapperGenerator) -> None:
        """Test module info retrieval without schema."""
        with pytest.raises(ValueError, match="No tools schema available"):
            generator.get_module_info()

    def test_generate_system_instructions(
        self, generator: MCPWrapperGenerator, sample_tools_schema: list[dict]
    ) -> None:
        """Test system instructions generation."""
        generator.tools_schema = sample_tools_schema

        instructions = generator.generate_system_instructions()

        assert "# MCP HTTP Module: testtools_mcp" in instructions
        assert "from testtools_mcp import" in instructions
        assert "read_file" in instructions
        assert "write_file" in instructions
        assert "search_files" in instructions
        assert "http://localhost:8000" in instructions

    def test_generate_system_instructions_without_schema(self, generator: MCPWrapperGenerator) -> None:
        """Test system instructions generation without schema."""
        with pytest.raises(ValueError, match="No tools schema available"):
            generator.generate_system_instructions()


class TestAgentIntegration:
    """Test integration with Agent class.

    Note: These tests require the full Alfredo tool registry to be initialized.
    They are skipped if the tool registry is not properly set up.
    """

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        """Create a temporary workspace."""
        return tmp_path / "workspace"

    @pytest.mark.skip(reason="Requires full Alfredo tool registry initialization")
    @patch("requests.post")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-for-testing"})
    def test_agent_with_codeact_mcp_functions(
        self, mock_post: Mock, workspace: Path, sample_tools_schema: list[dict]
    ) -> None:
        """Test Agent initialization with codeact_mcp_functions."""
        workspace.mkdir()

        # Mock session initialization
        init_response = Mock()
        init_response.headers = {"Mcp-Session-Id": "test-session-123"}
        init_response.raise_for_status = Mock()

        # Mock notification response
        notif_response = Mock()
        notif_response.raise_for_status = Mock()

        # Mock tools/list response (SSE format)
        tools_response = Mock()
        tools_response.headers = {"Content-Type": "text/event-stream"}
        tools_response.raise_for_status = Mock()
        tools_response.iter_lines = Mock(
            return_value=[f'data: {{"result": {{"tools": {json.dumps(sample_tools_schema)}}}}}']
        )

        mock_post.side_effect = [init_response, notif_response, tools_response]

        # Import here to avoid import errors if dependencies not installed
        from alfredo import Agent

        # Create agent with codeact_mcp_functions
        agent = Agent(
            cwd=str(workspace),
            model_name="gpt-4.1-mini",
            codeact_mcp_functions={
                "testtools": {
                    "url": "http://localhost:8000",
                    "headers": {"Authorization": "Bearer token"},
                }
            },
            verbose=False,
        )

        # Verify module was generated
        module_path = workspace / "testtools_mcp.py"
        assert module_path.exists()

        # Verify module content
        with open(module_path) as f:
            content = f.read()

        assert "def read_file" in content
        assert "def write_file" in content

        # Verify tool was added to agent
        assert agent.tools is not None
        # Should have Alfredo tools + MCP HTTP documentation tool
        tool_names = [getattr(t, "name", getattr(t.to_langchain_tool(), "name", "")) for t in agent.tools]
        assert "mcp_http_modules_info" in tool_names

    @pytest.mark.skip(reason="Requires full Alfredo tool registry initialization")
    @patch("requests.post")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-for-testing"})
    def test_agent_with_multiple_mcp_servers(
        self, mock_post: Mock, workspace: Path, sample_tools_schema: list[dict]
    ) -> None:
        """Test Agent with multiple MCP HTTP servers."""
        workspace.mkdir()

        # Mock responses for 2 servers (init + notification + tools/list for each)
        def create_mock_sequence():
            init_resp = Mock()
            init_resp.headers = {"Mcp-Session-Id": "test-session"}
            init_resp.raise_for_status = Mock()

            notif_resp = Mock()
            notif_resp.raise_for_status = Mock()

            tools_resp = Mock()
            tools_resp.headers = {"Content-Type": "text/event-stream"}
            tools_resp.raise_for_status = Mock()
            tools_resp.iter_lines = Mock(
                return_value=[f'data: {{"result": {{"tools": {json.dumps(sample_tools_schema)}}}}}'.encode()]
            )
            return [init_resp, notif_resp, tools_resp]

        # 2 servers * 3 requests each = 6 mock responses
        mock_post.side_effect = create_mock_sequence() + create_mock_sequence()

        from alfredo import Agent

        # Create agent with multiple configs
        _agent = Agent(
            cwd=str(workspace),
            model_name="gpt-4.1-mini",
            codeact_mcp_functions={
                "server1": {"url": "http://localhost:8000"},
                "server2": {"url": "http://localhost:8001"},
            },
            verbose=False,
        )

        # Verify both modules were generated
        assert (workspace / "server1_mcp.py").exists()
        assert (workspace / "server2_mcp.py").exists()

    @pytest.mark.skip(reason="Requires full Alfredo tool registry initialization")
    @patch("requests.post")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-for-testing"})
    def test_agent_with_invalid_server(self, mock_post: Mock, workspace: Path) -> None:
        """Test Agent handling of invalid MCP server."""
        workspace.mkdir()

        import requests

        # Mock connection failure during session initialization
        mock_post.side_effect = requests.RequestException("Connection refused")

        from alfredo import Agent

        # Should not raise, just skip the failing server
        _agent = Agent(
            cwd=str(workspace),
            model_name="gpt-4.1-mini",
            codeact_mcp_functions={
                "failing_server": {"url": "http://localhost:9999"},
            },
            verbose=False,
        )

        # Module should not be generated
        assert not (workspace / "failing_server_mcp.py").exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tool_with_no_parameters(self, generator: MCPWrapperGenerator) -> None:
        """Test tool with no parameters."""
        tool = {
            "name": "get_status",
            "description": "Get server status",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

        code = generator._generate_function_code(tool)

        assert "def get_status() -> Dict[str, Any]:" in code

    def test_tool_with_complex_types(self, generator: MCPWrapperGenerator) -> None:
        """Test tool with array and object types."""
        tool = {
            "name": "process_data",
            "description": "Process data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of items",
                    },
                    "config": {
                        "type": "object",
                        "description": "Configuration object",
                    },
                },
                "required": ["items"],
            },
        }

        code = generator._generate_function_code(tool)

        assert "items: List[Any]" in code
        assert "config: Optional[Dict[str, Any]] = None" in code

    def test_module_with_special_characters_in_name(self) -> None:
        """Test module name handling with special characters."""
        gen = MCPWrapperGenerator(
            server_url="http://localhost:8000",
            name="my-tools",  # Hyphens in name
        )

        # Name should be preserved as-is for now (would need schema to get module_info)
        assert gen.name == "my-tools"
        # In production, might want to sanitize: "my-tools" -> "my_tools"
