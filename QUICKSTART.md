# Quick Start Guide

Get up and running with the Aesthetic Scorer MCP Server in minutes!

## Installation

```bash
# Navigate to project directory
cd /home/maron/projects/aesthetic-mcp

# Activate virtual environment (already created)
source venv/bin/activate

# Verify installation
python test_scorer.py
```

## Usage with Claude Desktop

### 1. Configure Claude Desktop

Edit your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

Add this configuration:

```json
{
  "mcpServers": {
    "aesthetic-scorer": {
      "command": "/home/maron/projects/aesthetic-mcp/venv/bin/python",
      "args": ["-m", "aesthetic_scorer_mcp.server"],
      "cwd": "/home/maron/projects/aesthetic-mcp",
      "env": {
        "PYTHONPATH": "/home/maron/projects/aesthetic-mcp/src"
      }
    }
  }
}
```

### 2. Restart Claude Desktop

Close and reopen Claude Desktop to load the MCP server.

### 3. Use the Tools

In Claude Desktop, you can now:

**Score an image from a file:**
```
Can you score the aesthetic quality of /path/to/my/image.jpg?
```

**Score multiple images:**
```
Compare the aesthetic scores of image1.jpg and image2.jpg
```

## Available Tools

### `score_image`
Analyzes an image file and returns 7 aesthetic scores.

**Input:**
- `image_path`: Path to image file (JPG, PNG, etc.)

**Output:**
```
Aesthetic Scores (0-5 scale):

Overall Aesthetic: 2.68
Technical Quality: 2.72
Composition: 1.53
Lighting: 3.48
Color Harmony: 1.87
Depth of Field: 3.41
Content: 4.06
```

### `score_image_base64`
Analyzes an image from base64-encoded data.

**Input:**
- `base64_data`: Base64-encoded image data

## What Gets Scored

The model analyzes 7 aesthetic dimensions:

1. **Overall Aesthetic** - General visual appeal
2. **Technical Quality** - Image sharpness, noise, artifacts
3. **Composition** - Rule of thirds, balance, framing
4. **Lighting** - Exposure, shadows, highlights
5. **Color Harmony** - Color balance and palette
6. **Depth of Field** - Focus and bokeh quality
7. **Content** - Subject matter and interest

Each score ranges from **0 (poor) to 5 (excellent)**.

## Command Line Usage

You can also use the scorer directly in Python:

```python
from aesthetic_scorer_mcp.server import score_image

# Score an image
scores = score_image('my_photo.jpg')
print(scores)
# {'aesthetic': 4.23, 'quality': 4.56, ...}
```

## Improving Accuracy

Currently using the base CLIP model. For better results:

1. Download the fine-tuned model:
   ```bash
   wget https://huggingface.co/rsinema/aesthetic-scorer/resolve/main/model.pt
   ```

2. Update `src/aesthetic_scorer_mcp/server.py` to load the weights (see README.md)

## Troubleshooting

**Model loading slowly?**
- First run downloads CLIP model (~600MB)
- Subsequent runs load from cache (~2-3 seconds)

**Out of memory?**
- Using CPU by default (GPU auto-detected if available)
- Resize large images before scoring

**Import errors?**
- Ensure virtual environment is activated
- Check PYTHONPATH includes `src` directory

## Testing

Run the test suite:
```bash
python test_scorer.py
```

See `TESTING.md` for detailed test results and more examples.

## Support

- GitHub Issues: (create a repository for this)
- Model: https://huggingface.co/rsinema/aesthetic-scorer
- MCP Protocol: https://modelcontextprotocol.io
