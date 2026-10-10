"""The paper harness preserves inputs, isolates credentials and records actual usage."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchmark.generated import cli, sandbox
from benchmark.generated.auth import private_json
from benchmark.generated.stirrup_worker import UsageRecorder


def test_snapshot_preserves_human_reference_and_shares_only_declared_inputs(tmp_path):
    source, target, case = (tmp_path / name for name in ("source", "run", "case"))
    for directory in ("environment/src", "database", "inputs/data"):
        (source / directory).mkdir(parents=True)
    (source / "inputs/data/history.csv").write_text("value\n42\n")
    (source / "environment/src/private-tool.py").write_text("# tool implementation")
    question = [{"id": 9, "type": "iot", "text": "Read the data", "characteristic_form": "Expected answer"}]
    private_json(source / "scenarios.json", question)
    private_json(source / "manifest.json", {
        "files": sandbox.file_hashes(source, ("environment", "database", "inputs", "scenarios.json")),
        "validation_evidence_is_agent_visible": False})
    sandbox.import_snapshot(source, target)
    sandbox.import_snapshot(source, target)  # unchanged resume
    assert (target / "scenarios.json").read_bytes() == (source / "scenarios.json").read_bytes()
    sandbox.prepare_case(target, case, question[0])
    assert (case / "workspace/data/history.csv").read_text() == "value\n42\n"
    assert not (case / "workspace/scenarios.json").exists()
    assert not (case / "workspace/src/private-tool.py").exists()
    assert sandbox.artifact_inventory(case, question[0]) == []
    (case / "workspace/result.txt").write_text("42")
    assert [row["path"] for row in sandbox.artifact_inventory(case, question[0])] == ["result.txt"]
    (source / "inputs/data/history.csv").write_text("value\n99\n")
    with pytest.raises(ValueError, match="manifest"):
        sandbox.import_snapshot(source, tmp_path / "other")


def test_stirrup_worker_receives_only_execution_credentials(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "workspace").mkdir()
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "generator-only-secret")
    monkeypatch.setenv("COUCHDB_PASSWORD", "database-only-secret")
    monkeypatch.setattr(cli, "prepare_auth", lambda *a: pytest.fail("Unexpected native auth"))

    def launch(command, *, env, cwd, **kwargs):
        assert "benchmark.generated.stirrup_worker" in command
        assert "--workspace" in command
        assert env["LITELLM_API_KEY"] == "execution-secret"
        assert "SEMANTIC_SCHOLAR_API_KEY" not in env and "COUCHDB_PASSWORD" not in env
        assert "execution-secret" not in " ".join(command)
        assert cwd == tmp_path / "workspace"
        private_json(tmp_path / "native/result.json", {"status": "completed", "answer": "42"})
        return MagicMock(returncode=0)

    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    assert cli.stirrup_case(tmp_path, "litellm_proxy/model", {}, 10,
                           {"LITELLM_API_KEY": "execution-secret"}, {"max_turns": 20})["answer"] == "42"


@pytest.mark.anyio
async def test_usage_records_real_responses_and_unknown_cost(tmp_path):
    recorder = UsageRecorder(tmp_path / "api-usage.json")
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="response", model="actual-model", usage=SimpleNamespace(model_dump=lambda: {
            "prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 60}}))

    await recorder.wrap(create)(model="requested-model", messages=[], api_key="never-save-this")
    assert len(calls) == 1  # no extra cache warm-up or grading call
    summary = recorder.summary()
    assert summary["cache_read_tokens"] == 60
    assert summary["cost_usd"] is None
    assert "never-save-this" not in recorder.path.read_text()
    assert recorder.calls[0]["response_model"] == "actual-model"


def test_router_credits_are_preferred_without_changing_explicit_routes(monkeypatch):
    for key in cli.KEYS:
        monkeypatch.delenv(key, raising=False)
    credentials = cli.evaluation_credentials({"AI_GATEWAY_API_KEY": "gateway", "OPENAI_API_KEY": "personal"})
    assert credentials["LITELLM_BASE_URL"] == "https://ai-gateway.vercel.sh/v1"
    assert cli.default_runners(credentials) == {"stirrup": "litellm_proxy/openai/gpt-5.6-luna"}
    credentials.update(TOKENROUTER_API_KEY="router", TOKENROUTER_BASE_URL="https://router.example/v1")
    assert cli.default_runners(credentials) == {"stirrup": "tokenrouter/openai/gpt-5.6-luna"}
    assert cli.runner_models({"stirrup": ["openai/model", "anthropic/model"]}, "general-execution") == [
        ("stirrup", "openai/model"), ("stirrup", "anthropic/model")]
