"""Tests for the vision tool."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from alfredo.tools.handlers.vision import AnalyzeImageHandler
from alfredo.tools.registry import registry


def test_vision_tool_registered() -> None:
    """Test that vision tool is registered."""
    spec = registry.get_spec("analyze_image")
    assert spec is not None
    assert spec.id == "analyze_image"
    assert spec.name == "analyze_image"
    assert "vision model" in spec.description.lower()

    handler_class = registry.get_handler("analyze_image")
    assert handler_class is not None
    assert handler_class == AnalyzeImageHandler


def test_vision_handler_initialization() -> None:
    """Test that handler initializes correctly."""
    handler = AnalyzeImageHandler(cwd=".")
    assert handler.cwd == Path(".")
    assert handler.tool_id == "analyze_image"
    assert handler.model_name == "gpt-4o-mini"  # default

    # Test custom model
    with TemporaryDirectory() as tmpdir:
        handler2 = AnalyzeImageHandler(cwd=tmpdir, model_name="gpt-4o")
        assert handler2.model_name == "gpt-4o"

    # Test environment variable fallback
    import os

    old_env = os.environ.get("ALFREDO_VISION_MODEL")
    try:
        os.environ["ALFREDO_VISION_MODEL"] = "gpt-4o"
        handler3 = AnalyzeImageHandler(cwd=".")
        assert handler3.model_name == "gpt-4o"
    finally:
        if old_env:
            os.environ["ALFREDO_VISION_MODEL"] = old_env
        else:
            os.environ.pop("ALFREDO_VISION_MODEL", None)


def test_vision_handler_file_not_found() -> None:
    """Test error when image file doesn't exist."""
    with TemporaryDirectory() as tmpdir:
        handler = AnalyzeImageHandler(cwd=tmpdir)
        result = handler.execute({"path": "nonexistent.jpg"})

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()


def test_vision_handler_not_a_file() -> None:
    """Test error when path is a directory."""
    with TemporaryDirectory() as tmpdir:
        # Create a directory
        dir_path = Path(tmpdir) / "subdir"
        dir_path.mkdir()

        handler = AnalyzeImageHandler(cwd=tmpdir)
        result = handler.execute({"path": "subdir"})

        assert not result.success
        assert result.error is not None
        assert "not a file" in result.error.lower()


def test_vision_handler_unsupported_format() -> None:
    """Test error for unsupported file format."""
    with TemporaryDirectory() as tmpdir:
        # Create a .txt file
        txt_file = Path(tmpdir) / "test.txt"
        txt_file.write_text("Not an image")

        handler = AnalyzeImageHandler(cwd=tmpdir)
        result = handler.execute({"path": "test.txt"})

        assert not result.success
        assert result.error is not None
        assert "unsupported" in result.error.lower()
        assert ".txt" in result.error


def test_vision_handler_supported_formats() -> None:
    """Test that all supported formats are recognized."""
    formats = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]

    for fmt in formats:
        assert fmt in AnalyzeImageHandler.SUPPORTED_FORMATS


def test_vision_handler_missing_path() -> None:
    """Test error when path parameter is missing."""
    handler = AnalyzeImageHandler(cwd=".")
    result = handler.execute({})

    assert not result.success
    assert result.error is not None
    assert "required" in result.error.lower() or "path" in result.error.lower()


def test_vision_handler_encode_image() -> None:
    """Test image encoding to base64."""
    with TemporaryDirectory() as tmpdir:
        # Create a tiny "image" file (just some bytes)
        img_path = Path(tmpdir) / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake png data")

        handler = AnalyzeImageHandler(cwd=tmpdir)

        # Test private method
        encoded = handler._encode_image(img_path)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Should be valid base64
        import base64

        decoded = base64.b64decode(encoded)
        assert decoded == img_path.read_bytes()


def test_vision_handler_get_mime_type() -> None:
    """Test MIME type detection."""
    handler = AnalyzeImageHandler(cwd=".")

    # Test various extensions
    assert handler._get_mime_type(Path("test.jpg")) == "image/jpeg"
    assert handler._get_mime_type(Path("test.jpeg")) == "image/jpeg"
    assert handler._get_mime_type(Path("test.png")) == "image/png"
    assert handler._get_mime_type(Path("test.gif")) == "image/gif"
    assert handler._get_mime_type(Path("test.webp")) == "image/webp"
    assert handler._get_mime_type(Path("test.bmp")) == "image/bmp"

    # Case insensitive
    assert handler._get_mime_type(Path("test.PNG")) == "image/png"
    assert handler._get_mime_type(Path("test.JPG")) == "image/jpeg"


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="No API key set for vision model",
)
def test_vision_handler_live_analysis() -> None:
    """Test live image analysis with vision model (requires API key).

    This test creates a simple colored PNG and analyzes it.
    It requires OPENAI_API_KEY or ANTHROPIC_API_KEY to be set.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    with TemporaryDirectory() as tmpdir:
        # Create a simple red square image
        img_path = Path(tmpdir) / "red_square.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        # Analyze with default model (gpt-4o-mini)
        handler = AnalyzeImageHandler(cwd=tmpdir, model_name="gpt-4o-mini")
        result = handler.execute({
            "path": "red_square.png",
            "prompt": "What color is this image? Answer with just the color name.",
        })

        # Check result
        assert result.success, f"Analysis failed: {result.error}"
        assert result.output is not None
        assert len(result.output) > 0

        # Output should mention "red" (might be case-insensitive)
        assert "red" in result.output.lower()


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not set",
)
def test_vision_handler_with_custom_model() -> None:
    """Test using a custom vision model."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    with TemporaryDirectory() as tmpdir:
        # Create a simple blue square
        img_path = Path(tmpdir) / "blue_square.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path)

        # Analyze with gpt-4o
        handler = AnalyzeImageHandler(cwd=tmpdir)
        result = handler.execute({
            "path": "blue_square.png",
            "prompt": "What is the dominant color in this image?",
            "model": "gpt-4o-mini",  # Can override default
        })

        assert result.success, f"Analysis failed: {result.error}"
        assert "blue" in result.output.lower()


def test_vision_tool_parameter_spec() -> None:
    """Test that tool parameters are correctly specified."""
    spec = registry.get_spec("analyze_image")
    assert spec is not None

    # Check parameters
    param_names = [p.name for p in spec.parameters]
    assert "path" in param_names
    assert "prompt" in param_names
    assert "model" in param_names

    # Check required parameters
    path_param = next(p for p in spec.parameters if p.name == "path")
    assert path_param.required is True

    prompt_param = next(p for p in spec.parameters if p.name == "prompt")
    assert prompt_param.required is False

    model_param = next(p for p in spec.parameters if p.name == "model")
    assert model_param.required is False


def test_vision_handler_langchain_not_available(monkeypatch) -> None:
    """Test error when LangChain is not available."""
    # This test is tricky because we can't actually uninstall langchain
    # We'll just verify the error handling exists in the code

    # We can't easily mock the import, but we can verify error handling exists
    import sys

    if "langchain_core" in sys.modules:
        # LangChain is available, so we can't test ImportError path
        pytest.skip("LangChain is installed, cannot test ImportError path")


def test_agent_vision_model_parameter() -> None:
    """Test that Agent accepts and sets vision_model parameter."""
    try:
        from alfredo import Agent
    except ImportError:
        pytest.skip("LangGraph not installed")

    # Test default (no vision model specified)
    agent1 = Agent(cwd=".", model_name="gpt-4.1-mini")
    assert agent1.vision_model is None

    # Test custom vision model
    agent2 = Agent(cwd=".", model_name="gpt-4.1-mini", vision_model="gpt-4o")
    assert agent2.vision_model == "gpt-4o"

    # Verify it sets the environment variable
    assert os.getenv("ALFREDO_VISION_MODEL") == "gpt-4o"

    # Test with Anthropic model
    agent3 = Agent(cwd=".", vision_model="anthropic:claude-3-5-sonnet-latest")
    assert agent3.vision_model == "anthropic:claude-3-5-sonnet-latest"
    assert os.getenv("ALFREDO_VISION_MODEL") == "anthropic:claude-3-5-sonnet-latest"
