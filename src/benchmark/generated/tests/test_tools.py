"""A compact launcher retains the independent public MCP endpoints."""
from types import SimpleNamespace

from benchmark.generated import tools


def test_compact_launcher_preserves_each_endpoint_and_lifespan(monkeypatch):
    configured, apps = [], []

    def configure(name, port):
        configured.append((name, port))
        app = object()
        apps.append(app)
        return SimpleNamespace(streamable_http_app=lambda: app)

    monkeypatch.setattr(tools, 'configured_server', configure)
    servers = tools.compact_servers()
    assert configured == [(name, 8100 + i) for i, name in enumerate(tools.SERVERS)]
    assert [s.config.app for s in servers] == apps
    assert [s.config.port for s in servers] == list(range(8100, 8106))
    assert all(s.config.lifespan == 'on' for s in servers)
