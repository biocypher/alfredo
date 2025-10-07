"""Tests for web tool handlers."""

from unittest.mock import Mock, patch

import pytest

from alfredo.agent import Agent


@pytest.fixture
def mock_response() -> Mock:
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.text = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Welcome</h1>
        <p>This is a test page.</p>
        <a href="/link">A link</a>
    </body>
    </html>
    """
    response.raise_for_status = Mock()
    return response


def test_web_fetch_success(mock_response: Mock) -> None:
    """Test successful web content fetching."""
    with patch("alfredo.tools.handlers.web.requests.get", return_value=mock_response):
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://example.com</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert result.success
        assert "Welcome" in result.output
        assert "test page" in result.output
        assert "example.com" in result.output  # URL should be in output


def test_web_fetch_http_upgrade(mock_response: Mock) -> None:
    """Test HTTP URL is upgraded to HTTPS."""
    with patch("alfredo.tools.handlers.web.requests.get", return_value=mock_response) as mock_get:
        agent = Agent()

        text = """
        <web_fetch>
        <url>http://example.com</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert result.success

        # Check that the URL was upgraded to HTTPS
        assert mock_get.called
        called_url = mock_get.call_args[0][0]
        assert called_url.startswith("https://")


def test_web_fetch_invalid_url() -> None:
    """Test error handling for invalid URL."""
    agent = Agent()

    text = """
    <web_fetch>
    <url>not-a-valid-url</url>
    </web_fetch>
    """

    result = agent.execute_from_text(text)
    assert result is not None
    assert not result.success
    assert result.error is not None
    assert "invalid" in result.error.lower()


def test_web_fetch_missing_url() -> None:
    """Test error handling for missing URL parameter."""
    agent = Agent()

    text = """
    <web_fetch>
    </web_fetch>
    """

    result = agent.execute_from_text(text)
    assert result is not None
    assert not result.success
    assert result.error is not None
    assert "required" in result.error.lower()


def test_web_fetch_timeout() -> None:
    """Test error handling for request timeout."""
    import requests

    with patch("alfredo.tools.handlers.web.requests.get", side_effect=requests.exceptions.Timeout()):
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://example.com</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert not result.success
        assert result.error is not None
        assert "timed out" in result.error.lower()


def test_web_fetch_connection_error() -> None:
    """Test error handling for connection errors."""
    import requests

    with patch(
        "alfredo.tools.handlers.web.requests.get",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://example.com</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert not result.success
        assert result.error is not None
        assert "failed" in result.error.lower() or "error" in result.error.lower()


def test_web_fetch_non_html_text_content() -> None:
    """Test handling of non-HTML text content (JSON, plain text)."""
    response = Mock()
    response.status_code = 200
    response.headers = {"content-type": "application/json"}
    response.text = '{"key": "value", "message": "Hello, world!"}'
    response.raise_for_status = Mock()

    with patch("alfredo.tools.handlers.web.requests.get", return_value=response):
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://api.example.com/data</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert result.success
        # Should return JSON as-is
        assert "key" in result.output
        assert "value" in result.output


def test_web_fetch_binary_content() -> None:
    """Test error handling for binary content."""
    response = Mock()
    response.status_code = 200
    response.headers = {"content-type": "image/png"}
    response.text = ""
    response.raise_for_status = Mock()

    with patch("alfredo.tools.handlers.web.requests.get", return_value=response):
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://example.com/image.png</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert not result.success
        assert result.error is not None
        assert "non-text" in result.error.lower() or "cannot process" in result.error.lower()


def test_web_fetch_html_with_scripts() -> None:
    """Test that script and style tags are removed from HTML."""
    response = Mock()
    response.status_code = 200
    response.headers = {"content-type": "text/html"}
    response.text = """
    <html>
    <head>
        <style>body { color: red; }</style>
        <script>console.log('test');</script>
    </head>
    <body>
        <h1>Clean Content</h1>
        <p>Visible text</p>
        <script>alert('remove me');</script>
    </body>
    </html>
    """
    response.raise_for_status = Mock()

    with patch("alfredo.tools.handlers.web.requests.get", return_value=response):
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://example.com</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert result.success
        assert "Clean Content" in result.output
        assert "Visible text" in result.output
        # Scripts and styles should not appear in output
        assert "console.log" not in result.output
        assert "alert" not in result.output


def test_web_fetch_redirects() -> None:
    """Test that redirects are followed."""
    response = Mock()
    response.status_code = 200
    response.headers = {"content-type": "text/html"}
    response.text = "<html><body><h1>Final page</h1></body></html>"
    response.raise_for_status = Mock()

    with patch("alfredo.tools.handlers.web.requests.get", return_value=response) as mock_get:
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://example.com/redirect</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert result.success

        # Verify that allow_redirects was set to True
        assert mock_get.called
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("allow_redirects") is True


def test_web_fetch_user_agent() -> None:
    """Test that a user agent is set in the request."""
    response = Mock()
    response.status_code = 200
    response.headers = {"content-type": "text/html"}
    response.text = "<html><body>Test</body></html>"
    response.raise_for_status = Mock()

    with patch("alfredo.tools.handlers.web.requests.get", return_value=response) as mock_get:
        agent = Agent()

        text = """
        <web_fetch>
        <url>https://example.com</url>
        </web_fetch>
        """

        result = agent.execute_from_text(text)
        assert result is not None
        assert result.success

        # Verify that User-Agent header was set
        assert mock_get.called
        call_kwargs = mock_get.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "User-Agent" in headers
        assert "Alfredo" in headers["User-Agent"]
