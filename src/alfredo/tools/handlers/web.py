"""Web-related tool handlers: fetch and process web content."""

from typing import Any
from urllib.parse import urlparse

try:
    import html2text
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None  # type: ignore[assignment]
    BeautifulSoup = None  # type: ignore[assignment,misc]
    html2text = None

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec


class WebFetchHandler(BaseToolHandler):
    """Handler for fetching web content and converting to markdown."""

    @property
    def tool_id(self) -> str:
        return "web_fetch"

    def execute(self, params: dict[str, Any]) -> ToolResult:  # noqa: C901
        """Fetch content from a URL and convert HTML to markdown.

        Args:
            params: Must contain 'url' - the URL to fetch

        Returns:
            ToolResult with markdown content or error
        """
        try:
            self.validate_required_param(params, "url")
            url = params["url"]

            # Check if required libraries are available
            if not requests or not html2text:
                return ToolResult.err(
                    "Required libraries not available. Please install: requests, beautifulsoup4, html2text"
                )

            # Validate URL format
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    return ToolResult.err(f"Invalid URL format: {url}")
            except Exception:
                return ToolResult.err(f"Invalid URL format: {url}")

            # Auto-upgrade HTTP to HTTPS
            if parsed.scheme == "http":
                url = url.replace("http://", "https://", 1)

            # Fetch URL content
            try:
                response = requests.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; AlfredoBot/1.0)",
                    },
                    timeout=30,
                    allow_redirects=True,
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                return ToolResult.err(f"Request timed out after 30 seconds: {url}")
            except requests.exceptions.TooManyRedirects:
                return ToolResult.err(f"Too many redirects: {url}")
            except requests.exceptions.RequestException as e:
                return ToolResult.err(f"Failed to fetch URL: {e}")

            # Check content type
            content_type = response.headers.get("content-type", "").lower()

            # Handle non-HTML content
            if "html" not in content_type:
                if "text" in content_type or "json" in content_type or "xml" in content_type:
                    # Return text content as-is
                    return ToolResult.ok(response.text)
                return ToolResult.err(
                    f"URL returned non-text content (content-type: {content_type}). "
                    f"Cannot process this type of content."
                )

            # Convert HTML to markdown
            markdown_content = self._html_to_markdown(response.text, url)
            return ToolResult.ok(markdown_content)

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Error fetching web content: {e}")

    def _html_to_markdown(self, html_content: str, url: str) -> str:
        """Convert HTML content to markdown format.

        Args:
            html_content: Raw HTML content
            url: Original URL (for context)

        Returns:
            Markdown-formatted content
        """
        # Use html2text to convert
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        h.single_line_break = True

        # Clean HTML with BeautifulSoup first (optional, but helps with malformed HTML)
        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html_content, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                html_content = str(soup)
            except Exception:  # noqa: S110
                pass  # If parsing fails, use original HTML

        # Convert to markdown
        markdown = h.handle(html_content)

        # Add URL reference at the top
        result = f"# Content from {url}\n\n{markdown}"

        return result


# Register the tool
spec = ToolSpec(
    id="web_fetch",
    name="web_fetch",
    description=(
        "Fetch content from a specified URL and convert HTML to markdown. "
        "The URL must be a fully-formed valid URL. HTTP URLs are automatically upgraded to HTTPS. "
        "This tool is read-only and does not modify any files. "
        "Use this when you need to retrieve and analyze web content."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="url",
            required=True,
            instruction="The URL to fetch content from (must be a valid URL with protocol)",
            usage="https://example.com/docs",
        ),
    ],
)

registry.register_spec(spec)
registry.register_handler("web_fetch", WebFetchHandler)
