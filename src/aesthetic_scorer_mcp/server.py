"""MCP server for image aesthetic scoring."""

import asyncio
import base64
from pathlib import Path
from typing import Any
import logging

import torch
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent

from .model import AestheticScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and processor
model: AestheticScorer | None = None
processor: CLIPProcessor | None = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the aesthetic scorer model and processor."""
    global model, processor

    if model is not None and processor is not None:
        return

    logger.info("Loading aesthetic scorer model...")
    logger.info(f"Using device: {device}")

    try:
        # Load CLIP processor and vision model
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        # Load the aesthetic scorer
        model = AestheticScorer(backbone)

        # Try to load fine-tuned weights from Hugging Face
        # For now, we'll use the base model structure
        # Users can download model.pt separately and load it
        logger.warning("Using base CLIP model. Download model.pt from https://huggingface.co/rsinema/aesthetic-scorer for fine-tuned weights")

        model = model.to(device)
        model.eval()

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def score_image(image_path: str | Path) -> dict[str, float]:
    """
    Score an image for aesthetic qualities.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with aesthetic scores (0-5 scale)
    """
    if model is None or processor is None:
        load_model()

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Get predictions
    with torch.no_grad():
        aesthetic, quality, composition, light, color, dof, content = model(pixel_values)

    # Convert to scores (multiply by 5 to get 0-5 range, applying sigmoid if needed)
    def to_score(tensor):
        # Apply sigmoid to normalize to 0-1, then scale to 0-5
        score = torch.sigmoid(tensor).item() * 5.0
        return round(score, 2)

    return {
        "aesthetic": to_score(aesthetic),
        "quality": to_score(quality),
        "composition": to_score(composition),
        "lighting": to_score(light),
        "color_harmony": to_score(color),
        "depth_of_field": to_score(dof),
        "content": to_score(content),
    }


def score_image_from_base64(base64_data: str) -> dict[str, float]:
    """
    Score an image from base64 encoded data.

    Args:
        base64_data: Base64 encoded image data

    Returns:
        Dictionary with aesthetic scores (0-5 scale)
    """
    import io

    if model is None or processor is None:
        load_model()

    # Decode base64 and load image
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Get predictions
    with torch.no_grad():
        aesthetic, quality, composition, light, color, dof, content = model(pixel_values)

    # Convert to scores
    def to_score(tensor):
        score = torch.sigmoid(tensor).item() * 5.0
        return round(score, 2)

    return {
        "aesthetic": to_score(aesthetic),
        "quality": to_score(quality),
        "composition": to_score(composition),
        "lighting": to_score(light),
        "color_harmony": to_score(color),
        "depth_of_field": to_score(dof),
        "content": to_score(content),
    }


# Create MCP server
server = Server("aesthetic-scorer")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="score_image",
            description=(
                "Analyze an image and provide aesthetic scores across 7 dimensions: "
                "overall aesthetic, technical quality, composition, lighting, color harmony, "
                "depth of field, and content. Scores range from 0-5 for each dimension."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file to analyze",
                    },
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="score_image_base64",
            description=(
                "Analyze an image from base64 encoded data and provide aesthetic scores "
                "across 7 dimensions: overall aesthetic, technical quality, composition, "
                "lighting, color harmony, depth of field, and content. Scores range from 0-5."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "base64_data": {
                        "type": "string",
                        "description": "Base64 encoded image data",
                    },
                },
                "required": ["base64_data"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "score_image":
            image_path = arguments.get("image_path")
            if not image_path:
                return [TextContent(type="text", text="Error: image_path is required")]

            scores = score_image(image_path)

            # Format the output
            output = "Aesthetic Scores (0-5 scale):\n\n"
            output += f"Overall Aesthetic: {scores['aesthetic']}\n"
            output += f"Technical Quality: {scores['quality']}\n"
            output += f"Composition: {scores['composition']}\n"
            output += f"Lighting: {scores['lighting']}\n"
            output += f"Color Harmony: {scores['color_harmony']}\n"
            output += f"Depth of Field: {scores['depth_of_field']}\n"
            output += f"Content: {scores['content']}\n"

            return [TextContent(type="text", text=output)]

        elif name == "score_image_base64":
            base64_data = arguments.get("base64_data")
            if not base64_data:
                return [TextContent(type="text", text="Error: base64_data is required")]

            scores = score_image_from_base64(base64_data)

            # Format the output
            output = "Aesthetic Scores (0-5 scale):\n\n"
            output += f"Overall Aesthetic: {scores['aesthetic']}\n"
            output += f"Technical Quality: {scores['quality']}\n"
            output += f"Composition: {scores['composition']}\n"
            output += f"Lighting: {scores['lighting']}\n"
            output += f"Color Harmony: {scores['color_harmony']}\n"
            output += f"Depth of Field: {scores['depth_of_field']}\n"
            output += f"Content: {scores['content']}\n"

            return [TextContent(type="text", text=output)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the server."""
    from mcp.server.stdio import stdio_server

    # Pre-load the model
    load_model()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
