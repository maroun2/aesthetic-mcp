"""Entry point for running the aesthetic scorer MCP server."""

from .server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
