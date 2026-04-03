# Aesthetic Scorer MCP Server

An MCP (Model Context Protocol) server that provides image aesthetic scoring using the [rsinema/aesthetic-scorer](https://huggingface.co/rsinema/aesthetic-scorer) model from Hugging Face.

## Features

- **7 Aesthetic Dimensions**: Analyzes images across multiple quality metrics
  - Overall Aesthetic Score - General visual appeal
  - Technical Quality - Image sharpness, noise, artifacts
  - Composition - Rule of thirds, balance, framing
  - Lighting - Exposure, shadows, highlights
  - Color Harmony - Color balance and palette
  - Depth of Field - Focus and bokeh quality
  - Content Score - Subject matter and interest
- **Flexible Input**: Score images from file paths or base64-encoded data
- **CLIP-based**: Built on OpenAI's CLIP ViT-B/32 visual encoder
- **0-5 Scale**: All scores normalized to an intuitive 0-5 range (0 = poor, 5 = excellent)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/maroun2/aesthetic-mcp.git
cd aesthetic-mcp

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode (required)
pip install -e .
```

### Optional: Download Fine-tuned Weights

The server works with the base CLIP model out of the box. For better results with the fine-tuned model:

1. Download `model.pt` from [https://huggingface.co/rsinema/aesthetic-scorer](https://huggingface.co/rsinema/aesthetic-scorer)
2. Place it in the project root or update the model loading path in `src/aesthetic_scorer_mcp/server.py`

## Usage

### Running the Server

After installing the package with `pip install -e .`, run:

```bash
# Activate your virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the server
python -m aesthetic_scorer_mcp.server
```

The server will wait for JSON-RPC messages over stdio (standard input/output).

### Configuring with Claude Desktop

Add this to your Claude Desktop MCP settings configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "aesthetic-scorer": {
      "command": "/path/to/aesthetic-mcp/venv/bin/python",
      "args": ["-m", "aesthetic_scorer_mcp.server"]
    }
  }
}
```

Replace `/path/to/aesthetic-mcp/venv/bin/python` with your actual virtual environment Python path.

**Note**: Ensure you've installed the package with `pip install -e .` in your virtual environment before configuring Claude Desktop.

After configuration, restart Claude Desktop to load the MCP server.

### Available Tools

#### 1. `score_image`

Score an image from a file path.

**Parameters:**
- `image_path` (string, required): Path to the image file

**Example:**
```json
{
  "image_path": "/path/to/image.jpg"
}
```

**Output:**
```
Aesthetic Scores (0-5 scale):

Overall Aesthetic: 4.23
Technical Quality: 4.56
Composition: 3.89
Lighting: 4.12
Color Harmony: 4.34
Depth of Field: 3.67
Content: 4.01
```

#### 2. `score_image_base64`

Score an image from base64-encoded data.

**Parameters:**
- `base64_data` (string, required): Base64 encoded image data

**Example:**
```json
{
  "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAA..."
}
```

## Using with Claude Desktop

Once configured, you can ask Claude to analyze images:

**Score a single image:**
```
Can you score the aesthetic quality of /path/to/my/image.jpg?
```

**Compare multiple images:**
```
Compare the aesthetic scores of image1.jpg and image2.jpg
```

**Analyze specific dimensions:**
```
What's the composition score for this photo?
```

## Testing the Server

### Quick Test with Echo Technique

You can test the MCP server using the echo technique before integrating it with Claude Desktop. Make sure you have:
1. Activated your virtual environment
2. Installed the package with `pip install -e .`

**List available tools:**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python3 -m aesthetic_scorer_mcp.server
```

**Score an image:**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"score_image","arguments":{"image_path":"/absolute/path/to/your/image.jpg"}}}' | python3 -m aesthetic_scorer_mcp.server
```

**Important notes:**
- Use absolute paths for image files (e.g., `/home/user/pictures/photo.jpg`)
- Each echo command starts a fresh server instance and loads the model (takes a few seconds)
- The server will output logs to stderr and JSON-RPC responses to stdout
- Model loading messages (INFO logs) are normal and indicate the server is starting

### Example with Real Image

```bash
# Activate virtual environment
source venv/bin/activate

# Test with your image (replace with actual path)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"score_image","arguments":{"image_path":"/home/user/photos/landscape.jpg"}}}' | python -m aesthetic_scorer_mcp.server 2>/dev/null | jq
```

The `2>/dev/null` suppresses model loading logs, and `jq` formats the JSON output (install with `apt install jq` or `brew install jq`).

## Command Line Usage

You can also use the scorer directly in Python:

```python
from aesthetic_scorer_mcp.server import score_image

# Score an image
scores = score_image('my_photo.jpg')
print(scores)
# Output: {'aesthetic': 4.23, 'quality': 4.56, 'composition': 3.89, ...}
```

## Model Details

- **Base Model**: CLIP ViT-B/32 (Visual Transformer)
- **Fine-tuned on**: PARA dataset for aesthetic scoring
- **Architecture**: CLIP visual encoder + 7 separate prediction heads
- **Score Range**: 0-5 for each dimension
- **Input**: RGB images (any size, automatically preprocessed)

## Development

### Project Structure

```
aesthetic-mcp/
├── src/
│   └── aesthetic_scorer_mcp/
│       ├── __init__.py
│       ├── server.py       # MCP server implementation
│       └── model.py        # AestheticScorer model definition
├── pyproject.toml          # Project configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore
```

## Technical Notes

### GPU Support

The server automatically detects and uses CUDA if available. For CPU-only inference:
- Model loading will automatically fall back to CPU
- Inference will be slower but fully functional

### Performance

- **First run**: Model download and initialization (~350MB)
- **Subsequent runs**: Fast inference (milliseconds per image on GPU)
- **Memory**: ~1GB RAM/VRAM for model weights

## Improving Accuracy

Currently using the base CLIP model. For better results with fine-tuned weights:

1. Download the fine-tuned model:
   ```bash
   cd aesthetic-mcp
   wget https://huggingface.co/rsinema/aesthetic-scorer/resolve/main/model.pt
   ```

2. The model will automatically be loaded if `model.pt` is present in the project root

## License

MIT License - See the [original model repository](https://huggingface.co/rsinema/aesthetic-scorer) for model-specific licensing.

## Credits

- Model: [rsinema/aesthetic-scorer](https://huggingface.co/rsinema/aesthetic-scorer)
- Base Architecture: OpenAI CLIP
- Training Dataset: PARA (Aesthetics dataset)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Troubleshooting

### Model fails to load
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.10+ required)
- Verify PyTorch installation for your system

### Out of memory errors
- Reduce image sizes before scoring
- Use CPU instead of GPU for inference
- Close other applications to free up memory

### Import errors or "Module not found" errors
- Ensure you've installed the package: `pip install -e .`
- Verify virtual environment is activated
- Reinstall the package: `pip install -e . --force-reinstall`
