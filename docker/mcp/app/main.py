from fastmcp import FastMCP
from starlette.responses import JSONResponse

from tools import register_all_tools

mcp = FastMCP("nier_analyzer")
register_all_tools(mcp)

@mcp.custom_route("/", methods=["GET"])
async def manifest(_request):
    """Expose MCP manifest for LangFlow discovery."""
    return JSONResponse(
        {
            "name": "nier_analyzer",
            "version": "0.1.0",
            "protocol": {
                "version": "1.0",
                "capabilities": {
                    "tools": True,
                    "resources": False,
                    "instructions": False,
                },
            },
            "transport": {
                "type": "sse",
                "url": "/sse",
            },
        }
    )


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=9999)
