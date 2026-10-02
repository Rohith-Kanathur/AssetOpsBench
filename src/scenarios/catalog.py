"""Read MCP tool schemas for generation prompts."""

from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcphub import DEFAULT_SERVERS

_REPO_ROOT = Path(__file__).resolve().parents[2]


async def get_tool_descriptions() -> dict[str, str]:
    descriptions = {}
    for name, command in DEFAULT_SERVERS.items():
        try:
            params = StdioServerParameters(
                command=command[0], args=command[1:], cwd=str(_REPO_ROOT),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
            lines = []
            for tool in result.tools:
                schema = tool.inputSchema or {}
                required = set(schema.get("required", []))
                parameters = ", ".join(
                    f"{key}: {value.get('type', 'any')}{'' if key in required else '?'}"
                    for key, value in schema.get("properties", {}).items()
                )
                lines.append(f"  - {tool.name}({parameters}): {tool.description or ''}")
            descriptions[name] = "\n".join(lines)
        except Exception as exc:
            descriptions[name] = f"  (unavailable: {exc})"
    return descriptions
