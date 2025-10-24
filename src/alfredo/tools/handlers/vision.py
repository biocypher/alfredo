"""Vision tool handlers: analyze images with vision models."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, ClassVar

from alfredo.tools.base import BaseToolHandler, ToolResult, ToolValidationError
from alfredo.tools.registry import registry
from alfredo.tools.specs import ModelFamily, ToolParameter, ToolSpec


class AnalyzeImageHandler(BaseToolHandler):
    """Handler for analyzing images with vision models."""

    # Supported image formats
    SUPPORTED_FORMATS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    def __init__(self, cwd: str | Path | None = None, model_name: str | None = None):
        """Initialize the handler.

        Args:
            cwd: Current working directory
            model_name: Optional model name to use for vision analysis.
                       Falls back to ALFREDO_VISION_MODEL env var, then gpt-4o-mini.
        """
        super().__init__(cwd)  # type: ignore[arg-type]
        # Priority: explicit param > env var > default
        self.model_name = model_name or os.getenv("ALFREDO_VISION_MODEL", "gpt-4o-mini")

    @property
    def tool_id(self) -> str:
        return "analyze_image"

    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Analyze an image with a vision model.

        Args:
            params: Must contain 'path' - the image file path
                   Optional: 'prompt' - question/instruction for the model
                   Optional: 'model' - vision model to use (overrides default)

        Returns:
            ToolResult with image analysis or error
        """
        try:
            self.validate_required_param(params, "path")
            image_path = self.resolve_path(params["path"])
            prompt = params.get("prompt", "Describe this image in detail.")
            model_name = params.get("model", self.model_name)

            # Validate file exists
            if not image_path.exists():
                return ToolResult.err(f"Image file not found: {self.get_relative_path(image_path)}")

            if not image_path.is_file():
                return ToolResult.err(f"Path is not a file: {self.get_relative_path(image_path)}")

            # Validate file format
            file_ext = image_path.suffix.lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                return ToolResult.err(
                    f"Unsupported image format: {file_ext}. "
                    f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
                )

            # Read and encode image
            try:
                image_data = self._encode_image(image_path)
            except Exception as e:
                return ToolResult.err(f"Failed to read image: {e}")

            # Determine MIME type
            mime_type = self._get_mime_type(image_path)

            # Analyze image with vision model
            try:
                analysis = self._analyze_with_model(image_data, mime_type, prompt, model_name)
                return ToolResult.ok(analysis)
            except Exception as e:
                return ToolResult.err(f"Vision model analysis failed: {e}")

        except ToolValidationError as e:
            return ToolResult.err(str(e))
        except Exception as e:
            return ToolResult.err(f"Error analyzing image: {e}")

    def _encode_image(self, image_path: Path) -> str:
        """Read and encode image as base64.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        """Determine MIME type for image.

        Args:
            image_path: Path to image file

        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type and mime_type.startswith("image/"):
            return mime_type

        # Fallback based on extension
        ext_to_mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return ext_to_mime.get(image_path.suffix.lower(), "image/jpeg")

    def _analyze_with_model(self, image_data: str, mime_type: str, prompt: str, model_name: str) -> str:
        """Send image to vision model for analysis.

        Args:
            image_data: Base64-encoded image
            mime_type: Image MIME type
            prompt: User prompt/question
            model_name: Model to use

        Returns:
            Model's analysis text

        Raises:
            ImportError: If langchain not available
            Exception: If model call fails
        """
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as e:
            msg = "LangChain is required for vision analysis. Install with: uv add langchain-core"
            raise ImportError(msg) from e

        # Try to initialize the model
        try:
            from langchain.chat_models import init_chat_model

            llm = init_chat_model(model_name)
        except Exception as e:
            msg = (
                f"Failed to initialize vision model '{model_name}'. "
                f"Make sure you have the required API key set. Error: {e}"
            )
            raise RuntimeError(msg) from e

        # Create message with image content
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
            ]
        )

        # Invoke model
        try:
            response = llm.invoke([message])
        except Exception as e:
            msg = f"Model invocation failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return str(response.content)


# Register the tool
spec = ToolSpec(
    id="analyze_image",
    name="analyze_image",
    description=(
        "Analyze an image file using a vision model (like GPT-4 Vision or Claude 3.5 Sonnet). "
        "Provide the path to a local image file and an optional prompt describing what you want to know. "
        "The model will analyze the image and return a description or answer to your question. "
        "Supports formats: JPG, PNG, GIF, WebP, BMP."
    ),
    variant=ModelFamily.GENERIC,
    parameters=[
        ToolParameter(
            name="path",
            required=True,
            instruction="Path to the image file to analyze (relative to current working directory)",
            usage="path/to/image.png",
        ),
        ToolParameter(
            name="prompt",
            required=False,
            instruction=(
                "Question or instruction for analyzing the image. "
                "Examples: 'Describe this image', 'What text is visible?', 'Is this a diagram or photo?'. "
                "Defaults to 'Describe this image in detail.'"
            ),
            usage="What does this screenshot show?",
        ),
        ToolParameter(
            name="model",
            required=False,
            instruction=(
                "Vision model to use for analysis. "
                "Examples: 'gpt-4o', 'gpt-4o-mini', 'anthropic:claude-3-5-sonnet-latest'. "
                "Defaults to 'gpt-4o-mini'."
            ),
            usage="gpt-4o",
        ),
    ],
)

registry.register_spec(spec)
registry.register_handler("analyze_image", AnalyzeImageHandler)
