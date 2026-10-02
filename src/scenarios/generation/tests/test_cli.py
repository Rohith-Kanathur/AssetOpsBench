import json

import pytest

from scenarios.generation import cli


def prepare_stub(repository, destination, ref):
    (destination / "workspace").mkdir(parents=True)


def test_one_command_accepts_an_asset_and_preserves_generation_settings(tmp_path, monkeypatch):
    target = tmp_path / "new"
    calls = []
    monkeypatch.setattr(cli, "prepare", prepare_stub)
    monkeypatch.setattr(cli, "audit_baseline", lambda _: [])
    monkeypatch.setattr(cli.runtime, "configure", lambda *args: None)
    monkeypatch.setattr(cli.runtime, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main(["run", str(target), "--asset-class", "AHU", "--domains", "IoT", "WO", "--count", "2"])
    request = json.loads((target / "workspace/request.json").read_text())
    assert request == {"asset_class": "AHU", "count": 2, "domains": ["IoT", "WO"]}
    assert calls[0][0][1] == "gpt-6-astra"
    assert calls[0][1] == {"harness": "codex", "reasoning_effort": "xhigh", "service_tier": "fast"}


def test_existing_directory_is_not_overwritten(tmp_path):
    marker = tmp_path / "existing-data"
    marker.write_text("keep")
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path), "--asset-class", "Transformer"])
    assert marker.read_text() == "keep"


def test_bad_scope_does_not_prepare_a_workspace(tmp_path):
    target = tmp_path / "new"
    with pytest.raises(SystemExit):
        cli.main(["run", str(target), "--asset-class", "Chiller", "--count", "1"])
    assert not target.exists()


def test_followup_retains_saved_scope(tmp_path, monkeypatch):
    (tmp_path / "workspace").mkdir()
    request = {"asset_class": "AHU", "count": 2, "domains": ["IoT", "WO"]}
    (tmp_path / "workspace/request.json").write_text(json.dumps(request))
    monkeypatch.setattr(cli.runtime, "configure", lambda *args: None)
    monkeypatch.setattr(cli.runtime, "run", lambda *args, **kwargs: None)
    cli.main(["run", str(tmp_path), "--followup", "Check sources."])
    assert json.loads((tmp_path / "workspace/request.json").read_text()) == request
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path), "--asset-class", "Transformer", "--followup", "Check."])
