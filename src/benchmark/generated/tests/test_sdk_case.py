"""SDK execution isolates native Claude configuration and preserves API auth."""

import json
from unittest.mock import MagicMock

import pytest

from benchmark.generated import cli


def test_multiple_models_share_a_runner():
    assert cli.runner_models({"codex": ["gpt-6-astra", "gpt-6-sol"],
                              "claude-code": ["claude-opus-5-5", "zai/glm-5.3"]},
                             "general-execution") == [
        ("codex", "gpt-6-astra"), ("codex", "gpt-6-sol"),
        ("claude-code", "claude-opus-5-5"), ("claude-code", "zai/glm-5.3")]
    assert cli.runner_models({"openai-agent": "zai/glm-5.3"}, "mcp-only") == [
        ("openai-agent", "zai/glm-5.3")]


@pytest.mark.parametrize("runners", [
    {}, {"codex": []}, {"codex": ["model", "model"]}, {"codex": ["model", None]},
    {"codex": " "}, {"openai-agent": "model"}, {"codex": {"model": "name"}},
])
def test_invalid_runner_matrix(runners):
    with pytest.raises(ValueError):
        cli.runner_models(runners, "general-execution")


def test_coding_zai_forwards_key_without_copying_subscription(tmp_path, monkeypatch):
    cli.private_json(tmp_path / "compose.json", {"services": {}})
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "personal-oauth")
    monkeypatch.setattr(cli, "prepare_auth", lambda *a: pytest.fail("Copied subscription auth"))

    def execute(case, *arguments, **kwargs):
        assert kwargs["env"]["ZAI_API_KEY"] == "test-zai-key"
        assert "ZAI_API_KEY" in arguments and "test-zai-key" not in arguments
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in arguments
        assert not any((case / "auth").iterdir())
        assert "test-zai-key" not in (case / "compose.json").read_text()
        cli.private_json(case / "native/result.json", {"status": "completed", "answer": "Done"})

    monkeypatch.setattr(cli.sandbox, "compose", execute)
    result = cli.coding_case(tmp_path, "claude-code", "zai/glm-5.3", 10,
                             {"ZAI_API_KEY": "test-zai-key"})
    assert result["status"] == "completed"
    assert not (tmp_path / "auth").exists()


@pytest.mark.parametrize("model,credentials,needs_login", [
    ("claude-opus-5-5", {}, True),
    ("claude-opus-5-5", {"ANTHROPIC_API_KEY": "test-api-key"}, False),
    ("tokenrouter/claude-model", {"TOKENROUTER_API_KEY": "test-router-key"}, False),
    ("claude-opus-5-5", {"CLAUDE_CODE_OAUTH_TOKEN": "test-oauth-token"}, False),
])
def test_sdk_claude_uses_clean_home_and_cleans_credentials(tmp_path, monkeypatch, model, credentials, needs_login):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "TOKENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SAFE_MODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SIMPLE", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/personal/config")
    (tmp_path / "workspace").mkdir()
    prepared = []

    def prepare(path, harness):
        prepared.append(harness)
        cli.private_json(path / ".claude/.credentials.json", {"accessToken": "test-private-oauth"})

    def launch(command, *, cwd, env, **kwargs):
        assert cwd == tmp_path / "workspace"
        assert env["HOME"] == str(tmp_path / "auth")
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "auth/.claude")
        assert "CLAUDE_CODE_SAFE_MODE" not in env and "CLAUDE_CODE_SIMPLE" not in env
        assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
        assert (tmp_path / "auth").is_dir()
        for name, value in credentials.items():
            assert env[name] == value
        cli.private_json(tmp_path / "native/result.json", {"status": "completed", "answer": "Ready"})
        return MagicMock(returncode=0)

    monkeypatch.setattr(cli, "prepare_auth", prepare)
    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    record = cli.sdk_case(tmp_path, "claude-agent", model, {}, 10, credentials)
    assert record["answer"] == "Ready"
    assert prepared == (["claude"] if needs_login else [])
    assert not (tmp_path / "auth").exists()


def test_sdk_claude_removes_auth_if_launch_fails(tmp_path, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "prepare_auth", lambda path, _: cli.private_json(path / ".claude/.credentials.json", {}))
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("launch failed")))
    with pytest.raises(OSError, match="launch failed"):
        cli.sdk_case(tmp_path, "claude-agent", "claude-opus-5-5", {}, 10, {})
    assert not (tmp_path / "auth").exists()
