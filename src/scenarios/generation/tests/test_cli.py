import json

import pytest

from scenarios.generation import cli


def prepare_stub(repository, destination, ref):
    (destination / "workspace").mkdir(parents=True)


@pytest.mark.parametrize("flag,value,key,expected", [
    ("--count", "3", "scenario_count", 3),
    ("--plan", '{"iot":1,"multiagent":1}', "scenario_plan",
     {"iot":1,"multiagent":1}),
])
def test_one_command_accepts_an_asset_and_preserves_generation_settings(tmp_path, monkeypatch, flag, value, key, expected):
    target = tmp_path / "new"
    calls = []
    monkeypatch.setattr(cli, "prepare", prepare_stub)
    monkeypatch.setattr(cli, "audit_baseline", lambda _: [])
    monkeypatch.setattr(cli.runtime, "configure", lambda *args: None)
    monkeypatch.setattr(cli.runtime, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main(["run", str(target), "--asset", "AHU", flag, value,
              "--repo", str(tmp_path / "source"), "--reasoning", "high", "--tier", "default"])
    request = json.loads((target / "workspace/request.json").read_text())
    assert request == {"asset_class": "AHU", "generation_mode": "general-execution",
                       "environment_policy": "extend", key: expected}
    assert calls[0][0][1] == "gpt-6-astra"
    assert calls[0][1] == {"harness": "codex", "reasoning_effort": "high", "service_tier": "default",
                           "env_file": tmp_path / "source/.env"}


@pytest.mark.parametrize("flag,value", [("--temperature", "0"), ("--mode", "mcp-only"), ("--counts", '{"positive":1}')])
def test_removed_options_fail_before_preparing_run(tmp_path, monkeypatch, capsys, flag, value):
    monkeypatch.setattr(cli, "prepare", lambda *a, **k: pytest.fail("Must not prepare a run"))
    target = tmp_path / "new"
    with pytest.raises(SystemExit) as error:
        cli.main(["run", str(target), "--asset", "Chiller", flag, value])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not target.exists()


def test_existing_directory_is_not_overwritten(tmp_path):
    marker = tmp_path / "existing-data"
    marker.write_text("keep")
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path), "--asset", "Transformer"])
    assert marker.read_text() == "keep"


def test_bad_scope_does_not_prepare_a_workspace(tmp_path):
    target = tmp_path / "new"
    with pytest.raises(SystemExit):
        cli.main(["run", str(target), "--asset", "Chiller", "--count", "-1"])
    assert not target.exists()


def test_followup_retains_saved_scope(tmp_path, monkeypatch):
    (tmp_path / "workspace").mkdir()
    request = {"asset_class": "AHU", "generation_mode": "general-execution", "scenario_count": 3}
    (tmp_path / "workspace/request.json").write_text(json.dumps(request))
    monkeypatch.setattr(cli.runtime, "configure", lambda *args: None)
    monkeypatch.setattr(cli.runtime, "run", lambda *args, **kwargs: None)
    cli.main(["run", str(tmp_path), "--followup", "Check sources."])
    assert json.loads((tmp_path / "workspace/request.json").read_text()) == request
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path), "--asset", "Transformer", "--followup", "Check."])


def test_budget_cannot_change_during_followup(tmp_path):
    (tmp_path / "workspace").mkdir()
    request = {"asset_class": "AHU", "generation_mode": "general-execution", "scenario_count": 1}
    (tmp_path / "workspace/request.json").write_text(json.dumps(request))
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path), "--followup", "Continue", "--count", "2"])
    assert json.loads((tmp_path / "workspace/request.json").read_text()) == request


def test_new_run_without_count_uses_twenty_five(tmp_path, monkeypatch):
    target = tmp_path / "new"
    monkeypatch.setattr(cli, "prepare", prepare_stub)
    monkeypatch.setattr(cli, "audit_baseline", lambda _: [])
    monkeypatch.setattr(cli.runtime, "configure", lambda *args: None)
    monkeypatch.setattr(cli.runtime, "run", lambda *args, **kwargs: None)
    cli.main(["run", str(target), "--asset", "Transformer"])
    request = json.loads((target / "workspace/request.json").read_text())
    assert request["scenario_count"] == 25
