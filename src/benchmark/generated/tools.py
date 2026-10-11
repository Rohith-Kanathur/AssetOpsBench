"""Serve the existing domain MCPs over HTTP inside a private tool container."""

import asyncio
import importlib
import signal
import subprocess
import sys
import threading
import time

SERVERS = ("iot", "fmsr", "tsfm", "wo", "vibration", "utilities")


def configured_server(name, port):
    from mcp.server.transport_security import TransportSecuritySettings

    mcp = importlib.import_module(f"servers.{name}.main").mcp
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(port)
    mcp.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=["tools:*", "127.0.0.1:*", "localhost:*"],
        allowed_origins=["http://tools:*", "http://127.0.0.1:*", "http://localhost:*"],
    )
    return mcp


def compact_servers():
    """Share imports, retaining each endpoint's own event loop and lifespan."""
    import uvicorn

    return [uvicorn.Server(uvicorn.Config(
        configured_server(name, 8100 + i).streamable_http_app(),
        host="0.0.0.0", port=8100 + i, log_level="warning", lifespan="on"))
        for i, name in enumerate(SERVERS)]


def run_compact():
    servers = compact_servers()
    stopped = threading.Event()
    shutdown_requested = threading.Event()
    errors = []

    def serve(server):
        try:
            asyncio.run(server.serve())
        except BaseException as exc:
            errors.append(exc)
        finally:
            stopped.set()

    def stop(*_):
        shutdown_requested.set()
        stopped.set()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    threads = [threading.Thread(target=serve, args=(server,), daemon=True,
                                name=f"mcp-{name}") for server, name in zip(servers, SERVERS)]
    try:
        for thread in threads:
            thread.start()
        stopped.wait()
    finally:
        for server in servers:
            server.should_exit = True
        for thread in threads:
            thread.join(timeout=10)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if errors:
        raise RuntimeError("A domain MCP server stopped") from errors[0]
    if not shutdown_requested.is_set():
        raise RuntimeError("A domain MCP server stopped unexpectedly")


def main():
    if len(sys.argv) > 2:
        mcp = configured_server(*sys.argv[1:])
        mcp.run(transport="streamable-http")
        return
    if "--separate-processes" not in sys.argv:
        run_compact()
        return
    # Retain the original launcher for infrastructure parity checks.
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
