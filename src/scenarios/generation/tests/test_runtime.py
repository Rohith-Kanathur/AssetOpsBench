import json

import pytest

from scenarios.generation.harnesses import codex_command
from scenarios.generation.runtime import configure


def test_container_does_not_mount_reference_checkout_or_docker_socket(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    (target / "baseline.json").write_text("{}")
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    path = configure(target, auth, kaggle)
    data = json.loads(path.read_text())
    mounts = data["services"]["agent"]["volumes"]
    assert len(mounts) == 3
    assert all("docker.sock" not in mount for mount in mounts)
    assert mounts[1].endswith("auth.json:ro")
    assert mounts[2].endswith("/run/kaggle-auth:ro")
    assert "ports" not in data["services"]["database"]
    original = path.read_text()
    configure(target, auth, kaggle)
    assert path.read_text() == original


def test_native_tools_and_mcp_are_available():
    command = codex_command()
    assert 'web_search="live"' in command
    assert "--ignore-user-config" in command
    assert not any("shell_tool=false" in arg for arg in command)
    assert 'mcp_servers.fmsr.args=["-m","servers.fmsr.main"]' in command
    assert command[command.index("--model") + 1] == "gpt-6-astra"
    assert 'model_reasoning_effort="xhigh"' in command
    assert 'service_tier="fast"' in command
    assert 'mcp_servers.research.args=["-m","scenarios.generation.research"]' in command


def test_explicit_model_settings_override_defaults():
    command = codex_command("gpt-6-sol", "high", "default")
    assert command[command.index("--model") + 1] == "gpt-6-sol"
    assert 'model_reasoning_effort="high"' in command
    assert 'service_tier="default"' in command


def test_existing_database_is_not_reseeded(tmp_path, monkeypatch):
    from scenarios.generation import runtime

    calls = []
    monkeypatch.setattr(runtime, "ensure_image", lambda: None)
    monkeypatch.setattr(runtime, "compose", lambda *args, **kwargs: calls.append(args))
    runtime.start(tmp_path)
    runtime.start(tmp_path)
    seed_calls = [args for args in calls if "couchdb.init_data" in args]
    assert len(seed_calls) == 1
    assert "--reset" not in seed_calls[0]


def test_failed_initialization_is_not_marked_complete(tmp_path, monkeypatch):
    import subprocess
    import pytest
    from scenarios.generation import runtime

    def fail_seed(*args, **kwargs):
        if "couchdb.init_data" in args:
            raise subprocess.CalledProcessError(1, "seed")

    monkeypatch.setattr(runtime, "ensure_image", lambda: None)
    monkeypatch.setattr(runtime, "compose", fail_seed)
    with pytest.raises(subprocess.CalledProcessError):
        runtime.start(tmp_path)
    assert not (tmp_path / "initialized.json").exists()


def test_only_research_key_is_loaded_from_dotenv(tmp_path, monkeypatch):
    from scenarios.generation.runtime import research_environment

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_TEST_SECRET", raising=False)
    path = tmp_path / ".env"
    path.write_text("SEMANTIC_SCHOLAR_API_KEY=private-test-key\nUNRELATED_TEST_SECRET=other\n")
    environment = research_environment(path)
    assert environment["SEMANTIC_SCHOLAR_API_KEY"] == "private-test-key"
    assert "UNRELATED_TEST_SECRET" not in environment
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "exported-key")
    assert research_environment(path)["SEMANTIC_SCHOLAR_API_KEY"] == "exported-key"


def test_run_repairs_failed_checks_before_marking_complete(tmp_path, monkeypatch):
    from scenarios.generation import runtime

    (tmp_path / "workspace/output").mkdir(parents=True)
    (tmp_path / "workspace/request.json").write_text('{"asset_class":"AHU","generation_mode":"general-execution"}')
    monkeypatch.setattr(runtime, "start", lambda _: None)
    calls = []
    monkeypatch.setattr(runtime, "compose", lambda *args, **kwargs: calls.append((args, kwargs)))
    results = iter([{"errors": ["Missing operator tasks"]}, {"errors": []}])
    monkeypatch.setattr(runtime, "check", lambda _: next(results))
    runtime.run(tmp_path, env_file=tmp_path / "absent.env")
    assert len(calls) == 2
    assert "output/review.json" in calls[1][1]["input"]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "complete"
    assert json.loads((tmp_path / "logs/run-1.json").read_text())["validation_status"] == "failed"
    assert json.loads((tmp_path / "logs/review-1.json").read_text())["errors"] == ["Missing operator tasks"]
    assert json.loads((tmp_path / "logs/run-2.json").read_text())["validation_status"] == "passed"


def test_unresolved_checks_leave_run_incomplete(tmp_path, monkeypatch):
    import pytest
    from scenarios.generation import runtime

    (tmp_path / "workspace/output").mkdir(parents=True)
    (tmp_path / "workspace/request.json").write_text('{"asset_class":"AHU","generation_mode":"general-execution"}')
    monkeypatch.setattr(runtime, "start", lambda _: None)
    monkeypatch.setattr(runtime, "compose", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "check", lambda _: {"errors": ["Unresolved evidence"]})
    with pytest.raises(ValueError, match="incomplete"):
        runtime.run(tmp_path, env_file=tmp_path / "absent.env")
    assert len(list((tmp_path / "logs").glob("codex-*.jsonl"))) == 3
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "incomplete"


def test_nonzero_checker_exit_cannot_report_success(tmp_path, monkeypatch):
    import subprocess
    from scenarios.generation import runtime

    (tmp_path / "workspace/output").mkdir(parents=True)
    monkeypatch.setattr(runtime, "start", lambda _: None)

    def fail(*args, **kwargs):
        kwargs["stdout"].write('{"errors":[]}')
        raise subprocess.CalledProcessError(2, "checker")

    monkeypatch.setattr(runtime, "compose", fail)
    assert runtime.check(tmp_path)["errors"] == ["Checker exited with status 2; see review.stderr"]


def test_interrupted_process_records_failure(tmp_path, monkeypatch):
    import pytest
    from scenarios.generation import runtime

    (tmp_path / "workspace/output").mkdir(parents=True)
    (tmp_path / "workspace/request.json").write_text('{"asset_class":"AHU","generation_mode":"general-execution"}')
    monkeypatch.setattr(runtime, "start", lambda _: None)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "compose", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runtime.run(tmp_path, env_file=tmp_path / "absent.env")
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    assert json.loads((tmp_path / "logs/run-1.json").read_text())["process_status"] == "failed"


def test_generation_cannot_change_the_frozen_mode_or_budget(tmp_path, monkeypatch):
    import pytest
    from scenarios.generation import runtime

    (tmp_path / "workspace/output").mkdir(parents=True)
    request = {"asset_class": "AHU", "generation_mode": "general-execution",
               "scenario_count": 10}
    path = tmp_path / "workspace/request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(runtime, "start", lambda _: None)

    def change_request(*args, **kwargs):
        path.write_text(json.dumps({**request, "scenario_count": 101}))

    monkeypatch.setattr(runtime, "compose", change_request)
    checked = []
    monkeypatch.setattr(runtime, "check", lambda _: checked.append(True) or {"errors": []})
    with pytest.raises(ValueError, match="changed the requested"):
        runtime.run(tmp_path, env_file=tmp_path / "absent.env")
    assert json.loads((tmp_path / "request.json").read_text()) == request
    assert checked == []
    with pytest.raises(ValueError, match="differs from the original"):
        runtime.run(tmp_path, env_file=tmp_path / "absent.env")


def test_agent_can_append_research_but_cannot_write_native_logs(tmp_path):
    from scenarios.generation.runtime import execution_arguments

    args = execution_arguments(tmp_path)
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "--volume"]
    assert f"{tmp_path / 'logs'}:/run-logs:ro" in mounts
    assert f"{tmp_path / 'logs/research.jsonl'}:/run-logs/research.jsonl" in mounts
    assert all(mount.endswith(":ro") or mount.endswith(":/run-logs/research.jsonl") for mount in mounts)
    (tmp_path / "logs/research.jsonl").write_text('{"query":"existing"}\n')
    execution_arguments(tmp_path)
    assert (tmp_path / "logs/research.jsonl").read_text() == '{"query":"existing"}\n'


def test_runtime_saves_native_events_without_agent_authored_log_files(tmp_path, monkeypatch):
    from scenarios.generation import runtime

    (tmp_path / "workspace/output").mkdir(parents=True)
    (tmp_path / "workspace/request.json").write_text('{"asset_class":"AHU","generation_mode":"general-execution"}')
    monkeypatch.setattr(runtime, "start", lambda _: None)
    events = '{"type":"item.completed","item":{"type":"command_execution","exit_code":0}}\n'

    def execute(*args, **kwargs):
        kwargs["stdout"].write(events)
        kwargs["stderr"].write("Native diagnostic\n")

    monkeypatch.setattr(runtime, "compose", execute)
    monkeypatch.setattr(runtime, "check", lambda _: {"errors": [], "scenario_execution": "not_verified"})
    runtime.run(tmp_path, env_file=tmp_path / "absent.env")
    assert (tmp_path / "logs/codex-1.jsonl").read_text() == events
    assert (tmp_path / "logs/codex-1.stderr").read_text() == "Native diagnostic\n"
    assert json.loads((tmp_path / "logs/review-1.json").read_text())["scenario_execution"] == "not_verified"
    assert not (tmp_path / "workspace/output/tool_checks.json").exists()
