"""Optional native execution adapters preserve explicit authentication."""

import json

import pytest

from benchmark.generated import cli


def test_multiple_models_share_a_runner():
    assert cli.runner_models({"codex": ["gpt-6-astra", "gpt-6-sol"],
                              "claude-code": ["claude-opus-5-5", "zai/glm-5.3"]},
                             "general-execution") == [
        ("codex", "gpt-6-astra"), ("codex", "gpt-6-sol"),
        ("claude-code", "claude-opus-5-5"), ("claude-code", "zai/glm-5.3")]


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
