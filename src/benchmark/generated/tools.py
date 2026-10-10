"""Serve the existing domain MCPs over HTTP inside a private tool container."""

import importlib
import subprocess
import sys
import time

SERVERS = ("iot", "fmsr", "tsfm", "wo", "vibration", "utilities")


def main():
    if len(sys.argv) > 1:
        from mcp.server.transport_security import TransportSecuritySettings

        name, port = sys.argv[1:]
        mcp = importlib.import_module(f"servers.{name}.main").mcp
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(port)
        mcp.settings.transport_security = TransportSecuritySettings(
            allowed_hosts=["tools:*", "127.0.0.1:*", "localhost:*"],
            allowed_origins=["http://tools:*", "http://127.0.0.1:*", "http://localhost:*"],
        )
        mcp.run(transport="streamable-http")
        return
    processes = [subprocess.Popen([sys.executable, __file__, name, str(8100 + i)])
                 for i, name in enumerate(SERVERS)]
    try:
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError("A domain MCP server stopped; inspect tools.log")
    finally:
        for process in processes:
            process.terminate()


if __name__ == "__main__":
    main()
