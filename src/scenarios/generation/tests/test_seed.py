import json
from types import SimpleNamespace

import pytest

from scenarios.generation import cli, runtime, seed
from scenarios.generation.tests.test_cli import prepare_stub


def prepared_snapshot(root):
    files = {
        "database/iot.json": [{"_id": "reading-1", "value": 42}],
        "inputs/data/history.json": [{"event": "inspection"}],
        "scenarios.json": [{"text": "Private human question", "characteristic_form": "Private answer"}],
        "environment/src/private.py": "Original snapshot implementation",
    }
    hashes = {}
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        hashes[name] = seed.digest(path)
    (root / "manifest.json").write_text(json.dumps({"files": hashes}))
    return root


def test_seed_copies_data_without_reference_answers_or_source_code(tmp_path):
    source = prepared_snapshot(tmp_path / "source")
    destination = tmp_path / "generation"
    (destination / "workspace").mkdir(parents=True)
    description = seed.describe(source)
    seed.prepare(source, destination, description)
    assert seed.verify(destination)["sha256"] == description["sha256"]
    files = {str(p.relative_to(destination / "workspace")) for p in (destination / "workspace").rglob("*") if p.is_file()}
    assert files == {"data/seed-database/iot.json", "data/history.json", "data/seed-manifest.json"}
    (source / "database/iot.json").write_text("changed source")
    assert seed.verify(destination)["sha256"] == description["sha256"]
    (destination / "workspace/data/history.json").write_text("changed copy")
    with pytest.raises(ValueError, match="Prepared seed changed"):
        seed.verify(destination)


@pytest.mark.parametrize("change", ["tamper", "extra", "symlink", "reserved"])
def test_bad_seed_rejected_before_generation_directory_exists(tmp_path, change):
    source = prepared_snapshot(tmp_path / "source")
    manifest = json.loads((source / "manifest.json").read_text())
    if change == "tamper":
        (source / "database/iot.json").write_text("[]")
    elif change == "extra":
        (source / "inputs/unlisted.txt").write_text("unlisted")
    elif change == "symlink":
        (source / "inputs/link").symlink_to(source / "scenarios.json")
    else:
        path = source / "inputs/request.json"
        path.write_text("{}")
        manifest["files"]["inputs/request.json"] = seed.digest(path)
        (source / "manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "new"
    with pytest.raises(SystemExit):
        cli.main(["run", str(destination), "--asset", "Chiller", "--seed", str(source)])
    assert not destination.exists()


def test_cli_seed_is_saved_and_cannot_change_on_followup(tmp_path, monkeypatch):
    source = prepared_snapshot(tmp_path / "source")
    destination = tmp_path / "new"
    monkeypatch.setattr(cli, "prepare", prepare_stub)
    monkeypatch.setattr(cli, "audit_baseline", lambda _: [])
    monkeypatch.setattr(runtime, "configure", lambda *a: None)
    monkeypatch.setattr(runtime, "run", lambda *a, **kw: None)
    cli.main(["run", str(destination), "--asset", "Chiller", "--seed", str(source), "--environment", "existing"])
    request = json.loads((destination / "workspace/request.json").read_text())
    assert request["seed"] == {"sha256": seed.describe(source)["sha256"]}
    cli.main(["run", str(destination), "--followup", "Continue"])
    changed = prepared_snapshot(tmp_path / "changed")
    path = changed / "database/iot.json"
    path.write_text("[]")
    manifest = json.loads((changed / "manifest.json").read_text())
    manifest["files"]["database/iot.json"] = seed.digest(path)
    (changed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        cli.main(["run", str(destination), "--followup", "Continue", "--seed", str(changed)])
    assert json.loads((destination / "workspace/request.json").read_text()) == request


def test_runtime_restores_seed_once_and_baselines_its_inputs(tmp_path, monkeypatch):
    source = prepared_snapshot(tmp_path / "source")
    destination = tmp_path / "generation"
    (destination / "workspace").mkdir(parents=True)
    seed.prepare(source, destination, seed.describe(source))
    (destination / "workspace/request.json").write_text('{"environment_policy":"existing"}')
    (destination / "baseline.json").write_text('{"files":{}}')
    calls = []
    monkeypatch.setattr(runtime, "ensure_image", lambda: None)
    monkeypatch.setattr(runtime, "compose", lambda *a, **kw: calls.append(a) or SimpleNamespace(stdout='{"iot":{"records":1}}'))
    runtime.start(destination)
    runtime.start(destination)
    assert len([call for call in calls if "scenarios.generation.seed" in call]) == 1
    assert not any("couchdb.init_data" in call for call in calls)
    baseline = json.loads((destination / "environment-baseline.json").read_text())
    assert baseline["files"]["data/history.json"] == seed.digest(destination / "workspace/data/history.json")


def test_failed_seed_import_is_retryable_and_never_falls_back(tmp_path, monkeypatch):
    source = prepared_snapshot(tmp_path / "source")
    destination = tmp_path / "generation"
    (destination / "workspace").mkdir(parents=True)
    seed.prepare(source, destination, seed.describe(source))
    monkeypatch.setattr(runtime, "ensure_image", lambda: None)
    calls = []
    def compose(*args, **kwargs):
        calls.append(args)
        if "scenarios.generation.seed" in args:
            raise RuntimeError("Import failed")
    monkeypatch.setattr(runtime, "compose", compose)
    with pytest.raises(RuntimeError, match="Import failed"):
        runtime.start(destination)
    assert not (destination / "initialized.json").exists()
    assert not any("couchdb.init_data" in call for call in calls)


def test_missing_seed_cannot_fall_back_to_repository_data(tmp_path, monkeypatch):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/request.json").write_text('{"seed":{"sha256":"expected"}}')
    monkeypatch.setattr(runtime, "ensure_image", lambda: pytest.fail("Must fail before starting containers"))
    with pytest.raises(ValueError, match="Prepared seed is missing"):
        runtime.start(tmp_path)


def test_restore_replaces_partial_import_and_checks_bulk_errors(tmp_path, monkeypatch):
    import requests
    (tmp_path / "iot.json").write_text('[{"_id":"reading-1","value":42}]')
    calls = []
    fail = False
    class Session:
        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            result = ["_users", "partial"] if url.endswith("/_all_dbs") else [{"error": "conflict"}] if fail and method == "POST" else {}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: result)
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setenv("COUCHDB_URL", "http://database:5984")
    monkeypatch.setenv("COUCHDB_USERNAME", "private")
    monkeypatch.setenv("COUCHDB_PASSWORD", "private")
    seed.restore(tmp_path)
    assert any(method == "DELETE" and url.endswith("/partial") for method, url, _ in calls)
    assert not any(method == "DELETE" and url.endswith("/_users") for method, url, _ in calls)
    assert calls[-1][2]["json"]["docs"][0]["value"] == 42
    fail = True
    with pytest.raises(ValueError, match="Failed to restore"):
        seed.restore(tmp_path)
