"""MCP server for image aesthetic scoring."""

import base64
import io
import json
import logging
import logging.handlers
import sys
from pathlib import Path

import torch
from transformers import CLIPProcessor
from PIL import Image
from mcp.server.fastmcp import FastMCP
from huggingface_hub import hf_hub_download

from .model import AestheticScorer

# Logging: file + stderr, never stdout
LOG_DIR = Path("/tmp/agor-mcp-logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "aesthetic-scorer.log"

file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stderr_handler])
logger = logging.getLogger(__name__)
logger.info("Aesthetic Scorer MCP server starting")

# Global model and processor (lazy loaded on first tool call)
_model: AestheticScorer | None = None
_processor: CLIPProcessor | None = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"

mcp = FastMCP("aesthetic-scorer")


def _load_model():
    """Load the aesthetic scorer model and processor (lazy, called on first use)."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return

    logger.info("Loading aesthetic scorer model...")
    logger.info(f"Using device: {_device}")

    # Load CLIP processor from rsinema repo
    _processor = CLIPProcessor.from_pretrained("rsinema/aesthetic-scorer")

    # Download and load the fine-tuned model weights from Hugging Face
    logger.info("Downloading fine-tuned model from HuggingFace...")
    model_path = hf_hub_download(repo_id="rsinema/aesthetic-scorer", filename="model.pt")
    logger.info(f"Loading fine-tuned model from {model_path}")

    state_dict = torch.load(model_path, map_location=_device, weights_only=False)

    from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPVisionConfig
    config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
    backbone = CLIPVisionTransformer(config)

    _model = AestheticScorer(backbone)
    _model.load_state_dict(state_dict)
    _model = _model.to(_device)
    _model.eval()

    logger.info("Fine-tuned model loaded successfully")


def _score(image: Image.Image) -> dict[str, float]:
    """Score a PIL image across 7 aesthetic dimensions."""
    _load_model()

    inputs = _processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_device)

    with torch.no_grad():
        aesthetic, quality, composition, light, color, dof, content = _model(pixel_values)

    def to_score(tensor):
        return round(torch.sigmoid(tensor).item() * 5.0, 2)

    return {
        "aesthetic": to_score(aesthetic),
        "quality": to_score(quality),
        "composition": to_score(composition),
        "lighting": to_score(light),
        "color_harmony": to_score(color),
        "depth_of_field": to_score(dof),
        "content": to_score(content),
    }


def _format_scores(scores: dict[str, float]) -> str:
    return (
        "Aesthetic Scores (0-5 scale):\n\n"
        f"Overall Aesthetic: {scores['aesthetic']}\n"
        f"Technical Quality: {scores['quality']}\n"
        f"Composition: {scores['composition']}\n"
        f"Lighting: {scores['lighting']}\n"
        f"Color Harmony: {scores['color_harmony']}\n"
        f"Depth of Field: {scores['depth_of_field']}\n"
        f"Content: {scores['content']}\n"
    )


@mcp.tool()
def score_image(image_path: str) -> str:
    """Analyze an image and provide aesthetic scores across 7 dimensions: overall aesthetic, technical quality, composition, lighting, color harmony, depth of field, and content. Scores range from 0-5 for each dimension."""
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(path).convert("RGB")
    scores = _score(image)
    return _format_scores(scores)


@mcp.tool()
def score_image_base64(base64_data: str) -> str:
    """Analyze an image from base64 encoded data and provide aesthetic scores across 7 dimensions: overall aesthetic, technical quality, composition, lighting, color harmony, depth of field, and content. Scores range from 0-5."""
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scores = _score(image)
    return _format_scores(scores)


async def main():
    """Run the server."""
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options()
        )


if __name__ == "__main__":
    mcp.run()
