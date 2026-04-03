# Aesthetic Scorer MCP Server - Development Rules

## Testing the Server

### Quick Test - List Available Tools

To test if the server is working and see available tools:

```bash
(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'; echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}') | venv/bin/python -m aesthetic_scorer_mcp
```

This initializes the MCP server and then lists all available tools. The model will NOT load during this test (lazy loading).

### Prerequisites

Ensure the virtual environment is set up:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Development Guidelines

- Always test changes using the command above before committing
- The server uses stdin/stdout for JSON-RPC communication
- Model: `rsinema/aesthetic-scorer` (ViT-based aesthetic scoring model)
