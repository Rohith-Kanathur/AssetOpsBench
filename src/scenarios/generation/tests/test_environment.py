import hashlib
import json

import pytest

from scenarios.generation import cli, environment, runtime
from scenarios.generation.data_grounding import validate_data_sources
from scenarios.generation.review import check_contract
from scenarios.generation.tests.test_cli import prepare_stub
from scenarios.generation.tests.test_review import one_contract, replace_artifact


def test_existing_environment_policy_is_retained_on_followup(tmp_path, monkeypatch):
    destination = tmp_path / "run"
    monkeypatch.setattr(cli, "prepare", prepare_stub)
    monkeypatch.setattr(cli, "audit_baseline", lambda _: [])
    monkeypatch.setattr(runtime, "configure", lambda *args: None)
    monkeypatch.setattr(runtime, "run", lambda *args, **kwargs: None)
    cli.main(["run", str(destination), "--asset", "Chiller", "--environment", "existing"])
    request = json.loads((destination / "workspace/request.json").read_text())
    assert request["environment_policy"] == "existing"
    assert request["generation_mode"] == "general-execution"
    cli.main(["run", str(destination), "--followup", "Continue"])
    with pytest.raises(SystemExit):
        cli.main(["run", str(destination), "--followup", "Continue", "--environment", "extend"])


@pytest.mark.parametrize("selected", environment.POLICIES)
def test_prompts_select_one_preparation_policy(tmp_path, selected):
    (tmp_path / "request.json").write_text(json.dumps({"environment_policy": selected}))
    (tmp_path / "profile.md").write_text("Research the asset.\n{{environment_guidance}}")
    (tmp_path / "generate.md").write_text("Generate scenarios.")
    environment.write_guidance(tmp_path)
    text = (tmp_path / "profile.md").read_text()
    assert "{{environment_guidance}}" not in text
    assert f"Environment policy: {selected}" in text
    if selected == "existing":
        assert "Continue academic research" in text
        assert "Actively look for real asset records" not in text


def test_existing_source_is_read_only_and_baseline_is_harness_owned(tmp_path):
    run = tmp_path / "run"
    (run / "workspace/src").mkdir(parents=True)
    (run / "baseline.json").write_text('{"files":{}}')
    (run / "workspace/request.json").write_text('{"environment_policy":"existing"}')
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    compose = json.loads(runtime.configure(run, auth, tmp_path / "missing").read_text())
    agent = compose["services"]["agent"]
    assert f"{run / 'workspace/src'}:/workspace/src:ro" in agent["volumes"]
    assert agent["environment"][environment.BASELINE_ENV] == "/opt/generation/environment-baseline.json"
    (run / "environment-baseline.json").write_text('{"files":{},"databases":{}}')
    args = runtime.execution_arguments(run)
    assert (run / "tooling/environment-baseline.json").read_text() == (run / "environment-baseline.json").read_text()
    assert f"{run / 'tooling'}:/opt/generation:ro" in args


def test_source_audit_detects_changed_added_and_removed_implementations(tmp_path):
    path = tmp_path / "src/servers/iot/main.py"
    path.parent.mkdir(parents=True)
    path.write_text("original")
    baseline = {"files": {"src/servers/iot/main.py": hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert environment.audit_source(tmp_path, baseline) == []
    path.write_text("changed")
    assert len(environment.audit_source(tmp_path, baseline)) == 1
    path.unlink()
    path.with_name("new.py").write_text("added")
    assert len(environment.audit_source(tmp_path, baseline)) == 2


def test_input_database_changes_are_reported():
    baseline = {"databases": {"iot": {"sha256": "original", "records": 2}}}
    assert environment.audit_database(baseline, baseline["databases"]) == []
    assert environment.audit_database(baseline, {"iot": {"sha256": "changed", "records": 2}})
    assert environment.audit_database(baseline, {**baseline["databases"], "new_data": {}})
    assert environment.audit_database(baseline, {})


def test_input_audit_still_runs_when_scenario_budget_is_incomplete(tmp_path, monkeypatch, capsys):
    from scenarios.generation import review

    monkeypatch.setattr(review, "check_contract", lambda *args: {
        "environment_policy": "existing", "errors": ["quota shortfall"]})
    monkeypatch.setattr(environment, "load_baseline", lambda: {"databases": {"iot": {}}})
    monkeypatch.setattr(environment, "database_state", lambda: {})
    assert review.main(["--workspace", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == ["quota shortfall", "Existing environment input database changed: iot"]


def test_database_snapshot_ignores_revisions_indexes_and_native_outputs(monkeypatch):
    import requests
    from types import SimpleNamespace

    monkeypatch.setenv("COUCHDB_URL", "http://database:5984")
    monkeypatch.setenv("COUCHDB_USERNAME", "test")
    monkeypatch.setenv("COUCHDB_PASSWORD", "private-test")
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        data = ["_users", "iot", "tsfm_runs", "forecast_result"] if url.endswith("_all_dbs") else {
            "rows": [{"id": "r1", "doc": {"_id": "r1", "_rev": "2-revision", "value": 4}},
                     {"id": "_design/index", "doc": {"_id": "_design/index"}}]}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    monkeypatch.setattr(requests, "get", get)
    state = environment.database_state()
    assert set(state) == {"iot"}
    assert state["iot"]["records"] == 1
    assert state["iot"]["sha256"] == hashlib.sha256(b'[{"_id":"r1","value":4}]').hexdigest()
    assert len(calls) == 2


def test_existing_input_baseline_is_captured_once_after_normal_initialization(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/request.json").write_text('{"environment_policy":"existing"}')
    (tmp_path / "baseline.json").write_text('{"files":{},"baseline":"revision"}')
    monkeypatch.setattr(runtime, "ensure_image", lambda: None)
    calls = []

    def compose(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='{"iot":{"sha256":"initial","records":3}}')

    monkeypatch.setattr(runtime, "compose", compose)
    runtime.start(tmp_path)
    runtime.start(tmp_path)
    assert len([c for c in calls if "couchdb.init_data" in c]) == 1
    assert len([c for c in calls if "scenarios.generation.environment" in c]) == 1
    baseline = json.loads((tmp_path / "environment-baseline.json").read_text())
    assert baseline["databases"]["iot"]["records"] == 3


def test_fixture_data_is_accepted_only_when_bound_to_existing_baseline(tmp_path, monkeypatch):
    one_contract(tmp_path)
    fixture = tmp_path / "src/couchdb/scenarios_data/shared/input.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"value":4}')
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    name = str(fixture.relative_to(tmp_path))
    baseline = tmp_path / "protected-baseline.json"
    baseline.write_text(json.dumps({"files": {name: digest}}))
    monkeypatch.setenv(environment.BASELINE_ENV, str(baseline))
    request = json.loads((tmp_path / "request.json").read_text())
    request["environment_policy"] = "existing"
    (tmp_path / "request.json").write_text(json.dumps(request))
    sources = json.loads((tmp_path / "output/sources.json").read_text())
    sources[0].update(kind="fixture", files=[{"path": name, "sha256": digest}])
    replace_artifact(tmp_path, "sources", sources)
    assert check_contract(tmp_path)["errors"] == []
    fixture.write_text('{"value":99}')
    assert any("source changed" in e for e in check_contract(tmp_path)["errors"])
    assert validate_data_sources(sources)[0]


def test_existing_policy_rejects_new_observed_roots_and_synthetic_histories():
    source = {"id": "new", "role": "data", "kind": "observed", "files": {"download.csv": "new"}}
    assert validate_data_sources([source], {"fixture.json": "original"})[0]
    source.update(kind="synthetic")
    assert any("lossless" in e for e in validate_data_sources([source], {})[0])
