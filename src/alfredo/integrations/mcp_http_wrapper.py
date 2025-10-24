"""MCP HTTP Wrapper Generator.

This module provides functionality to generate Python wrapper modules for MCP HTTP servers.
The generated modules allow agents to import and use MCP tools as regular Python functions
instead of through a ReAct loop.

Example:
    >>> from alfredo.integrations.mcp_http_wrapper import MCPWrapperGenerator
    >>> generator = MCPWrapperGenerator(
    ...     server_url="http://localhost:8000",
    ...     name="codeact",
    ...     headers={"Authorization": "Bearer token"}
    ... )
    >>> generator.fetch_tools_schema()
    >>> generator.generate_module("./codeact_mcp.py")
    >>> info = generator.get_module_info()
"""

import json
from importlib.util import find_spec
from typing import Any, Optional

REQUESTS_AVAILABLE = find_spec("requests") is not None


class MCPWrapperGenerator:
    """Generate Python wrapper modules for MCP HTTP servers.

    This class fetches tool schemas from an MCP HTTP server and generates a Python
    module with typed wrapper functions that make HTTP calls to the server.

    Attributes:
        server_url: Base URL of the MCP HTTP server
        name: Name identifier for the module (e.g., "codeact" -> "codeact_mcp")
        headers: Optional HTTP headers for authentication
        tools_schema: Fetched tool schemas from the server
    """

    def __init__(
        self,
        server_url: str,
        name: str,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialize the MCP wrapper generator.

        Args:
            server_url: Base URL of the MCP HTTP server (e.g., "http://localhost:8000")
            name: Name identifier for the generated module
            headers: Optional HTTP headers (e.g., {"Authorization": "Bearer token"})

        Raises:
            ImportError: If requests library is not installed
        """
        if not REQUESTS_AVAILABLE:
            msg = "requests library is not installed. Install it with: uv add requests"
            raise ImportError(msg)

        if not isinstance(server_url, str):
            msg = f"server_url must be a string, got {type(server_url)}"
            raise TypeError(msg)

        self.server_url = server_url.rstrip("/")
        self.name = name
        self.headers = headers or {}
        self.tools_schema: list[dict[str, Any]] = []
        self.session_id: Optional[str] = None
        self._request_id_counter = 0

    def _get_next_request_id(self) -> int:
        """Get next unique request ID for JSON-RPC calls.

        Returns:
            Integer request ID
        """
        self._request_id_counter += 1
        return self._request_id_counter

    def initialize_session(self) -> str:
        """Initialize MCP session and get session ID.

        Makes a JSON-RPC initialize request to the MCP server and stores the session ID.

        Returns:
            Session ID string

        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If session initialization fails
        """
        import requests

        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "alfredo", "version": "1.0.0"},
            },
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **self.headers,
        }

        try:
            response = requests.post(
                self.server_url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            # Extract session ID from response headers
            session_id = response.headers.get("Mcp-Session-Id")

            # Parse response to check for errors
            # Some servers return SSE, others return plain JSON
            content_type = response.headers.get("Content-Type", "")
            result = self._parse_sse_response(response) if "text/event-stream" in content_type else response.json()

            # Check for JSON-RPC errors in response
            if "error" in result:
                error = result["error"]
                msg = f"MCP initialize error: {error.get('message', 'Unknown error')}"
                raise ValueError(msg)

            # Store session ID (may be None for servers that don't use session-based auth)
            self.session_id = session_id

            # Send initialized notification if we have a session ID
            if session_id:
                notif_payload = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
                notif_headers = {
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": session_id,
                    **self.headers,
                }
                requests.post(self.server_url, json=notif_payload, headers=notif_headers, timeout=5)
                return session_id
            else:
                return ""

        except requests.RequestException as e:
            msg = f"Failed to initialize MCP session at {self.server_url}: {e}"
            raise requests.RequestException(msg) from e

    def _ensure_session(self) -> None:
        """Ensure we have a valid session, initialize if needed."""
        if self.session_id is None:
            self.initialize_session()

    def _parse_sse_response(self, response: Any) -> dict[str, Any]:
        """Parse Server-Sent Events (SSE) response.

        Args:
            response: requests Response object

        Returns:
            Parsed JSON-RPC result

        Raises:
            ValueError: If SSE parsing fails
        """
        # Parse SSE format: event: message\ndata: {...}\n\n
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                try:
                    parsed: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError as e:
                    msg = f"Failed to parse SSE data: {e}"
                    raise ValueError(msg) from e
                else:
                    return parsed

        msg = "No data found in SSE response"
        raise ValueError(msg)

    def fetch_tools_schema(self) -> list[dict[str, Any]]:
        """Fetch tool schemas from the MCP server using JSON-RPC tools/list method.

        Makes a JSON-RPC request to list available tools and parses the response.

        Returns:
            List of tool schema dictionaries

        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response is not valid JSON or has unexpected format
        """
        import requests

        # Ensure we have a session
        self._ensure_session()

        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/list",
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **self.headers,
        }

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        try:
            response = requests.post(
                self.server_url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            # Check content type and parse accordingly
            content_type = response.headers.get("Content-Type", "")
            result = self._parse_sse_response(response) if "text/event-stream" in content_type else response.json()

            # Check for JSON-RPC errors
            if "error" in result:
                error = result["error"]
                msg = f"JSON-RPC error: {error.get('code')} - {error.get('message')}"
                raise ValueError(msg)

            if "result" not in result:
                msg = "Invalid JSON-RPC response: missing 'result' field"
                raise ValueError(msg)

            # Extract tools from result
            tools = result["result"].get("tools", [])

            if not isinstance(tools, list):
                msg = f"Expected list of tools, got {type(tools)}"
                raise TypeError(msg)
            else:
                self.tools_schema = tools
                return tools

        except requests.RequestException as e:
            msg = f"Failed to fetch tools schema from {self.server_url}: {e}"
            raise requests.RequestException(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from {self.server_url}: {e}"
            raise ValueError(msg) from e

    def _map_json_type_to_python(self, json_type: str, is_optional: bool = False) -> str:
        """Map JSON Schema types to Python type hints.

        Args:
            json_type: JSON Schema type (e.g., "string", "integer", "array")
            is_optional: Whether the parameter is optional

        Returns:
            Python type hint string (e.g., "str", "Optional[int]", "List[Any]")
        """
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
            "null": "None",
        }

        python_type = type_mapping.get(json_type, "Any")

        if is_optional:
            return f"Optional[{python_type}]"

        return python_type

    def _generate_function_code(self, tool: dict[str, Any]) -> str:
        """Generate Python function code for a single tool.

        Args:
            tool: Tool schema dictionary with name, description, and inputSchema

        Returns:
            Complete Python function code as string
        """
        tool_name = tool.get("name", "unknown_tool")
        description = tool.get("description", "No description available")
        input_schema = tool.get("inputSchema", {})

        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))

        # Generate function signature (module-level functions, no self parameter)
        params: list[str] = []

        # Build parameter list with type hints
        param_docs = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "No description")
            is_optional = param_name not in required_params

            python_type = self._map_json_type_to_python(param_type, is_optional)

            if is_optional:
                params.append(f"{param_name}: {python_type} = None")
            else:
                params.append(f"{param_name}: {python_type}")

            param_docs.append(f"        {param_name}: {param_desc}")

        params_str = ", ".join(params)

        # Generate docstring
        param_docs_str = "\n".join(param_docs) if param_docs else "        None"

        docstring = f'''    """{description}

    Args:
{param_docs_str}

    Returns:
        Tool result as dictionary

    Raises:
        RuntimeError: If HTTP request fails
    """'''

        # Generate function body
        # Separate required and optional args
        required_args = [
            f'            "{param_name}": {param_name},' for param_name in properties if param_name in required_params
        ]

        optional_args = [
            f'    if {param_name} is not None:\n        arguments["{param_name}"] = {param_name}'
            for param_name in properties
            if param_name not in required_params
        ]

        required_args_str = "\n".join(required_args) if required_args else ""
        optional_args_str = "\n\n" + "\n".join(optional_args) if optional_args else ""

        function_code = f'''
def {tool_name}({params_str}) -> Dict[str, Any]:
{docstring}
    # Ensure session is active
    _ensure_session()

    # Build arguments
    arguments = {{
{required_args_str}
    }}
{optional_args_str}

    # Create JSON-RPC payload
    payload = {{
        "jsonrpc": "2.0",
        "id": _get_next_request_id(),
        "method": "tools/call",
        "params": {{
            "name": "{tool_name}",
            "arguments": arguments
        }}
    }}

    # Set headers
    headers = {{
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        **HEADERS
    }}
    if SESSION_ID:
        headers["Mcp-Session-Id"] = SESSION_ID

    try:
        response = requests.post(
            SERVER_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        # Check content type and parse accordingly
        content_type = response.headers.get("Content-Type", "")

        if "text/event-stream" in content_type:
            # Parse SSE response
            result = _parse_sse_response(response)
        else:
            # Parse regular JSON response
            result = response.json()

        # Check for JSON-RPC errors
        if "error" in result:
            error = result["error"]
            raise RuntimeError(f"JSON-RPC error {{error.get('code')}}: {{error.get('message')}}")

        # Extract and parse text content from MCP response
        if "result" in result:
            content = result["result"].get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if first_item.get("type") == "text":
                    text = first_item.get("text", "")
                    if text:
                        # Parse JSON text and return as dictionary
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError as e:
                            raise RuntimeError(f"Failed to parse text content as JSON: {{e}}")

            raise RuntimeError("No text content found in MCP response")

        raise RuntimeError("Invalid MCP response: missing 'result' field")

    except requests.RequestException as e:
        # Handle session expiry
        if e.response is not None and e.response.status_code == 404:
            # Session expired, re-initialize and retry
            _initialize_session()
            return {tool_name}({", ".join([f"{pname}={pname}" for pname in [p.split(":")[0].strip() for p in params if p]])})
        raise RuntimeError(f"MCP tool '{tool_name}' failed: {{e}}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from MCP tool '{tool_name}': {{e}}")
'''

        return function_code

    def _generate_module_header(self) -> str:
        """Generate module header with docstring, imports, and session management.

        Returns:
            Module header code as string
        """
        headers_str = json.dumps(self.headers, indent=4) if self.headers else "{}"

        header = f'''"""Auto-generated MCP HTTP wrapper: {self.name}

This module provides Python functions for calling MCP tools via JSON-RPC over HTTP.
Generated from: {self.server_url}

Uses the Model Context Protocol (MCP) with JSON-RPC 2.0.

DO NOT EDIT THIS FILE MANUALLY - it is auto-generated.
"""

import json
from typing import Dict, Any, Optional, List

try:
    import requests
except ImportError:
    raise ImportError("requests library is required. Install with: pip install requests")


SERVER_URL = "{self.server_url}"
HEADERS = {headers_str}
SESSION_ID: Optional[str] = None
_request_id_counter = 0


def _get_next_request_id() -> int:
    """Get next unique request ID for JSON-RPC calls."""
    global _request_id_counter
    _request_id_counter += 1
    return _request_id_counter


def _initialize_session() -> str:
    """Initialize MCP session and get session ID.

    Returns:
        Session ID string

    Raises:
        RuntimeError: If session initialization fails
    """
    global SESSION_ID

    payload = {{
        "jsonrpc": "2.0",
        "id": _get_next_request_id(),
        "method": "initialize",
        "params": {{
            "protocolVersion": "2024-11-05",
            "capabilities": {{}},
            "clientInfo": {{"name": "alfredo", "version": "1.0.0"}}
        }}
    }}

    headers = {{
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        **HEADERS
    }}

    try:
        response = requests.post(SERVER_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        # Extract session ID from response headers
        session_id = response.headers.get("Mcp-Session-Id")

        # Parse response to check for errors
        # Some servers return SSE, others return plain JSON
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            # Parse SSE response (e.g., remote servers)
            result = _parse_sse_response(response)
        else:
            # Parse regular JSON response (e.g., local servers)
            result = response.json()

        # Check for JSON-RPC errors in response
        if "error" in result:
            error = result["error"]
            raise RuntimeError(f"MCP initialize error: {{error.get('message', 'Unknown error')}}")

        # Store session ID (may be None for servers that don't use session-based auth)
        SESSION_ID = session_id

        # Send initialized notification if we have a session ID
        if session_id:
            notif_payload = {{
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }}
            notif_headers = {{
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Mcp-Session-Id": session_id,
                **HEADERS
            }}
            requests.post(SERVER_URL, json=notif_payload, headers=notif_headers, timeout=5)

        return session_id or ""

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to initialize MCP session: {{e}}")


def _ensure_session() -> None:
    """Ensure we have a valid session, initialize if needed."""
    if SESSION_ID is None:
        _initialize_session()


def _parse_sse_response(response) -> Dict[str, Any]:
    """Parse Server-Sent Events (SSE) response.

    Args:
        response: requests Response object

    Returns:
        Parsed JSON-RPC result

    Raises:
        RuntimeError: If SSE parsing fails
    """
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data_str = line[6:]  # Remove "data: " prefix
            try:
                return json.loads(data_str)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse SSE data: {{e}}")

    raise RuntimeError("No data found in SSE response")


'''

        return header

    def generate_module(self, output_path: str) -> None:
        """Generate the complete Python module file.

        Args:
            output_path: Path where the module file will be written

        Raises:
            ValueError: If tools_schema is empty (call fetch_tools_schema first)
            IOError: If writing to the file fails
        """
        if not self.tools_schema:
            msg = "No tools schema available. Call fetch_tools_schema() first."
            raise ValueError(msg)

        # Generate header
        module_code = self._generate_module_header()

        # Generate function for each tool
        for tool in self.tools_schema:
            function_code = self._generate_function_code(tool)
            module_code += function_code
            module_code += "\n"

        # Write to file
        try:
            with open(output_path, "w") as f:
                f.write(module_code)
        except OSError as e:
            msg = f"Failed to write module to {output_path}: {e}"
            raise OSError(msg) from e

    def get_module_info(self) -> dict[str, Any]:
        """Get module information for prompt injection.

        Returns:
            Dictionary with module name, functions, and documentation
        """
        if not self.tools_schema:
            msg = "No tools schema available. Call fetch_tools_schema() first."
            raise ValueError(msg)

        module_name = f"{self.name}_mcp"
        functions = []

        for tool in self.tools_schema:
            tool_name = tool.get("name", "unknown_tool")
            description = tool.get("description", "No description")
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required_params = set(input_schema.get("required", []))

            # Build signature
            params = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "string")
                is_optional = param_name not in required_params
                python_type = self._map_json_type_to_python(param_type, is_optional)
                params.append(f"{param_name}: {python_type}")

            signature = f"{tool_name}({', '.join(params)}) -> Dict[str, Any]"

            functions.append({
                "name": tool_name,
                "signature": signature,
                "description": description,
            })

        return {
            "module_name": module_name,
            "functions": functions,
            "server_url": self.server_url,
        }

    def generate_system_instructions(self) -> str:
        """Generate system prompt instructions for the agent.

        Returns:
            Formatted instructions string for injection into system prompt
        """
        module_info = self.get_module_info()
        module_name = module_info["module_name"]
        functions = module_info["functions"]

        instructions = f"""# MCP HTTP Module: {module_name}

The following Python module is available in your working directory for use in scripts:

## Import Statement
```python
from {module_name} import {", ".join([f["name"] for f in functions[:3]])}{"..." if len(functions) > 3 else ""}
```

## Available Functions

"""

        for func in functions:
            instructions += f"- `{func['signature']}` - {func['description']}\n"

        instructions += f"""
These functions make HTTP calls to the MCP server at {self.server_url}.
Use them in Python scripts executed via the execute_command tool.
"""

        return instructions
