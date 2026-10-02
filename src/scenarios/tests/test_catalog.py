"""Generation discovers MCP schemas without running a scenario agent."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from mcp import Tool
import pytest

from scenarios import catalog


@pytest.mark.anyio
async def test_discovers_schemas_without_calling_tools(monkeypatch):
    commands = []
    methods = []

    @asynccontextmanager
    async def stdio(params):
        commands.append(params)
        yield "read", "write"

    class Session:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            methods.append("initialize")

        async def list_tools(self):
            methods.append("list_tools")
            return SimpleNamespace(tools=[Tool(
                name="history", description="Read past measurements.",
                inputSchema={"type": "object", "properties": {
                    "asset_id": {"type": "string"}, "limit": {"type": "integer"},
                }, "required": ["asset_id"]},
            )])

        async def call_tool(self, *args):
            pytest.fail("Generation catalog must never execute a tool")

    monkeypatch.setattr(catalog, "DEFAULT_SERVERS", {"iot": ["uv", "run", "iot-mcp-server"]})
    monkeypatch.setattr(catalog, "stdio_client", stdio)
    monkeypatch.setattr(catalog, "ClientSession", Session)
    assert await catalog.get_tool_descriptions() == {
        "iot": "  - history(asset_id: string, limit: integer?): Read past measurements.",
    }
    assert methods == ["initialize", "list_tools"]
    assert (commands[0].command, commands[0].args) == ("uv", ["run", "iot-mcp-server"])


@pytest.mark.anyio
async def test_unavailable_server_is_recorded_for_generation(monkeypatch):
    @asynccontextmanager
    async def unavailable(params):
        raise RuntimeError("Server unavailable")
        yield

    monkeypatch.setattr(catalog, "DEFAULT_SERVERS", {"iot": ["uv", "run", "iot-mcp-server"]})
    monkeypatch.setattr(catalog, "stdio_client", unavailable)
    assert await catalog.get_tool_descriptions() == {"iot": "  (unavailable: Server unavailable)"}
