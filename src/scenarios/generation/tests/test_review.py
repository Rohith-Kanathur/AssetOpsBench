import hashlib
import json

import pytest

from scenarios.generation.review import check_contract, check_grounding, local_file
from scenarios.generation.contracts import RESEARCH


def profile_fixture():
    return {"asset_class": "Transformer", "description": "Recorded asset evidence",
            "operator_tasks": ["Inspect the asset"], "manager_tasks": ["Plan maintenance"],
            "assets": [{"site": "S1", "asset_id": "T1", "sensors": ["oil_temperature"],
                        "source_ids": ["fixture"], "data_source_ids": ["fixture"]}],
            "failure_modes": [], "sensor_mapping": [], "gaps": [{"id": "missing-channel"}],
            "available_capabilities": {"iot": ["iot.asset_detail"]},
            "research": {key: {"status": "gap", "summary": "Not available in this fixture",
                               "source_ids": []} for key in RESEARCH}}


def save_contract(workspace, request, scenarios, allocation, tools=("iot.asset_detail",)):
    (workspace / "output").mkdir()
    for name in ("README.md", "requirements.txt", "environment.md"):
        (workspace / "output" / name).write_text("fixture")
    evidence = workspace / "output/response.json"
    evidence.write_text("{}\n")
    for scenario in scenarios:
        scenario.update(text=f"Inspect T1 at S1 for issue {scenario['id']}", category="Inspection",
                        characteristic_form="Read " + " and ".join(tools) + " and inspect the recorded evidence", source_ids=["fixture"],
                        grounding={"scope": "asset", "site": "S1", "asset_id": "T1", "data_source_ids": ["fixture"]})
    values = {
        "profile": profile_fixture(), "allocation": allocation, "scenarios": scenarios,
        "sources": [{"id": "fixture", "kind": "observed", "role": "data", "url": "repository:HEAD:fixture.json",
                     "files": [{"path": "output/response.json",
                                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}],
    }
    for name, value in values.items():
        (workspace / f"output/{name}.json").write_text(json.dumps(value))
    (workspace / "request.json").write_text(json.dumps(request))


def test_evidence_cannot_escape_workspace(tmp_path):
    work = tmp_path / "workspace"
    work.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("private")
    (work / "link").symlink_to(secret)
    for name in ("../secret", "link", "missing"):
        with pytest.raises(ValueError):
            local_file(work, name)


def test_missing_sensor_and_empty_window_are_recorded_without_classifying_scenario():
    def invoke(tool, args):
        if tool == "iot.asset_detail":
            return {"result": {"asset_id": "T1", "site_name": "S1"}}
        if tool == "iot.measured_sensors":
            return {"result": {"sensors": ["oil_temperature"]}}
        return {"result": {"total_records": 0}}
    scenario = {"id": 1, "type": "iot",
                "grounding": {"asset_id": "T1", "site": "S1",
                              "sensors": ["imaginary", "oil_temperature"]}}
    errors, evidence = check_grounding([scenario], invoke)
    assert errors == []
    assert evidence[1]["result"]["sensors"] == ["oil_temperature"]
    assert evidence[-1]["result"]["total_records"] == 0
    assert len(evidence) == 4


def test_cross_asset_workorder_is_rejected():
    def invoke(tool, args):
        if tool == "iot.asset_detail":
            return {"asset_id": "T1", "site_name": "S1"}
        return {"work_order": {"assetnum": "T2"}}
    errors, evidence = check_grounding([{"id": 1, "type": "wo",
        "grounding": {"asset_id": "T1", "site": "S1", "workorder_ids": ["W1"]}}], invoke)
    assert errors == ["Scenario 1: work order W1 belongs to another asset"]


@pytest.mark.parametrize("budget_key", ["scenario_count", "scenario_plan"])
def test_contract_accepts_budget_and_matching_allocation(tmp_path, budget_key):
    allocation = {"iot": 2}
    budget = allocation["iot"] if budget_key == "scenario_count" else allocation
    scenarios = [{"id": 1, "type": "iot"},
                 {"id": 2, "type": "iot"}]
    save_contract(tmp_path, {budget_key: budget}, scenarios, allocation)
    report = check_contract(tmp_path)
    assert report["errors"] == []
    assert report["scenario_count"] == 2
    assert report["counts"] == {"iot": 2}
    assert "positive" not in report and "negative" not in report


def test_contract_rejects_domain_budget_mismatch(tmp_path):
    allocation = {"iot": 1}
    save_contract(tmp_path, {"scenario_plan": allocation},
                  [{"id": 1, "type": "wo"}], allocation)
    assert check_contract(tmp_path)["errors"]


@pytest.mark.parametrize("budget_request", [
    {"scenario_plan": {"iot": 1}},
    {"scenario_count": 2},
])
def test_contract_rejects_allocation_that_changes_requested_budget(tmp_path, budget_request):
    allocation = {"wo": 1}
    save_contract(tmp_path, budget_request, [{"id": 1, "type": "wo"}], allocation)
    assert check_contract(tmp_path)["errors"]


def test_contract_preserves_source_and_checksum_checks(tmp_path):
    allocation = {"iot": 1}
    save_contract(tmp_path, {"scenario_count": allocation["iot"]},
                  [{"id": 1, "type": "iot"}], allocation)
    scenarios = json.loads((tmp_path / "output/scenarios.json").read_text())
    scenarios[0]["source_ids"] = ["unknown"]
    (tmp_path / "output/scenarios.json").write_text(json.dumps(scenarios))
    (tmp_path / "output/response.json").write_text('{"changed": true}\n')
    report = check_contract(tmp_path)
    assert "Scenario 1: unresolved source IDs" in report["errors"]
    assert "Source checksum mismatch: output/response.json" in report["errors"]
    assert report["scenario_count"] == 1


@pytest.mark.parametrize("tools, valid", [
    (("iot.asset_detail", "iot.installed_sensors"), False),
    (("iot.asset_detail", "utilities.current_time"), False),
    (("iot.asset_detail", "fmsr.list_failure_modes"), True),
])
def test_multiagent_rubric_requires_two_domain_servers(tmp_path, tools, valid):
    allocation = {"multiagent": 1}
    save_contract(tmp_path, {"scenario_plan": allocation},
                  [{"id": 1, "type": "multiagent"}], allocation, tools)
    errors = check_contract(tmp_path)["errors"]
    if valid:
        assert errors == []
    else:
        assert errors == ["Scenario 1: multiagent rubric must span at least two domain servers"]


@pytest.mark.parametrize("domain", ["vibration", "Vibration"])
def test_vibration_grounding_uses_vibration_sensor_tool(domain):
    def invoke(tool, args):
        if tool == "iot.asset_detail":
            return {"asset_id": "A1", "site_name": "S1"}
        assert tool == "vibration.list_vibration_sensors"
        return {"sensors": ["acceleration"]}
    errors, evidence = check_grounding([{"id": 1, "type": domain,
        "grounding": {"asset_id": "A1", "site": "S1"}}], invoke)
    assert errors == []
    assert evidence[-1]["tool"] == "vibration.list_vibration_sensors"


def replace_artifact(workspace, name, value):
    (workspace / f"output/{name}.json").write_text(json.dumps(value))


def one_contract(workspace):
    allocation = {"iot": 1}
    save_contract(workspace, {"asset_class": "Transformer", "scenario_plan": allocation},
                  [{"id": 1, "type": "iot"}], allocation)


def test_class_catalog_grounding_does_not_require_an_iot_asset():
    calls = []
    def invoke(tool, args):
        calls.append((tool, args))
        return {"failure_modes": ["Overheating"], "asset_class": "Transformer"}
    errors, _ = check_grounding([{"id": 1, "type": "fmsr",
                                "grounding": {"scope": "class", "asset_class": "Transformer"}}], invoke)
    assert errors == []
    assert calls == [("fmsr.get_failure_modes", {"asset_class": "Transformer"})]


def test_vibration_grounding_checks_requested_channel_and_window():
    def invoke(tool, args):
        if tool == "iot.asset_detail":
            return {"asset_id": "T1", "site_name": "S1"}
        if tool == "vibration.list_vibration_sensors":
            return {"sensors": ["tank"]}
        assert tool == "vibration.get_vibration_data"
        assert args["start"] == "2020-01-01T00:00:00"
        assert args["final"] == "2020-01-01T00:00:01"
        return {"error": "No vibration data found for asset T1"}
    errors, evidence = check_grounding([{"id": 1, "type": "vibration",
                                 "grounding": {"site": "S1", "asset_id": "T1", "sensors": ["tank"],
                                               "start": "2020-01-01T00:00:00", "end": "2020-01-01T00:00:01"}}], invoke)
    assert errors == []
    assert evidence[-1]["result"] == {"error": "No vibration data found for asset T1"}


def test_manual_receipts_are_ignored_and_never_treated_as_execution_proof(tmp_path):
    one_contract(tmp_path)
    rows = json.loads((tmp_path / "output/scenarios.json").read_text())
    rows[0]["grounding"]["output_workorder_ids"] = ["NEW1"]
    replace_artifact(tmp_path, "scenarios", rows)
    before = check_contract(tmp_path)
    assert before["errors"] == []
    assert before["scenario_execution"] == "not_verified"
    # Neither malformed receipts nor forged success should change validation.
    for content in ("not-json", '[{"scenario_id":1,"execution_steps":[{"exit_code":0}]}]'):
        (tmp_path / "output/tool_checks.json").write_text(content)
        assert check_contract(tmp_path) == before
    invoked = []
    errors, evidence = check_grounding(rows, lambda tool, args: invoked.append(tool) or
                                      {"asset_id": "T1", "site_name": "S1"})
    assert errors == []
    assert invoked == ["iot.asset_detail"]
    assert evidence == [{"scenario_id": 1, "tool": "iot.asset_detail",
                         "arguments": {"site_name": "S1", "asset_id": "T1"},
                         "result": {"asset_id": "T1", "site_name": "S1"}}]


def test_cli_emits_machine_readable_report_and_nonzero_for_errors(tmp_path, capsys):
    from scenarios.generation.review import main
    assert main(["--workspace", str(tmp_path), "--stage", "profile"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] and report["stage"] == "profile"
    assert "do not verify scenario execution" in report["scope"]
    assert report["scenario_execution"] == "not_verified"


def test_new_server_tools_are_discovered_before_completion_gate(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace
    from scenarios.generation.review import main
    one_contract(tmp_path)
    profile = profile_fixture()
    profile["available_capabilities"]["transformer_health"] = [{"tool": "transformer_health.thermal_review"}]
    replace_artifact(tmp_path, "profile", profile)
    rows = json.loads((tmp_path / "output/scenarios.json").read_text())
    rows[0]["characteristic_form"] = "Use transformer_health.thermal_review to assess the recorded evidence."
    replace_artifact(tmp_path, "scenarios", rows)
    loaded = []
    class Universe:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def load_tools(self):
            loaded.append(True)
            print("Diagnostic log must not corrupt JSON stdout")
        def list_tools(self):
            return ["iot.asset_detail", "transformer_health.thermal_review"]
        def run(self, tool, arguments):
            assert tool == "iot.asset_detail"
            return {"asset_id": "T1", "site_name": "S1"}
    monkeypatch.setitem(sys.modules, "mcphub", SimpleNamespace(ToolUniverse=Universe))
    assert main(["--workspace", str(tmp_path), "--stage", "all"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert not report["errors"] and loaded == [True]
    assert report["grounding_checks"][0]["tool"] == "iot.asset_detail"
    assert report["grounding_checks"][0]["result"] == {"asset_id": "T1", "site_name": "S1"}
    assert report["scenario_execution"] == "not_verified"
    assert "transformer_health.thermal_review" in report["tools"]
    assert "Diagnostic log" in captured.err


def test_missing_asset_is_observed_but_tool_failure_still_blocks():
    scenario = {"id": 1, "type": "iot", "grounding": {"site": "S1", "asset_id": "Absent"}}
    errors, evidence = check_grounding([scenario], lambda *_: {"error": "unknown asset_id Absent at site S1"})
    assert errors == []
    assert evidence[0]["result"] == {"error": "unknown asset_id Absent at site S1"}
    def unavailable(*_):
        raise ConnectionError("Offline")
    errors, _ = check_grounding([scenario], unavailable)
    assert errors == ["Scenario 1: iot.asset_detail raised ConnectionError"]
    errors, _ = check_grounding([scenario], lambda *_: {}, tools=[])
    assert errors == ["Scenario 1: unavailable grounding tool iot.asset_detail"]


@pytest.mark.parametrize("message", ["database unavailable", "unable to inspect telemetry stream extent", "invalid date"])
def test_error_payloads_are_not_silently_accepted_as_missing_evidence(message):
    scenario = {"id": 1, "type": "iot", "grounding": {"site": "S1", "asset_id": "T1"}}
    errors, evidence = check_grounding([scenario], lambda *_: {"error": message})
    assert errors == ["Scenario 1: iot.asset_detail failed"]
    assert evidence[0]["result"]["error"] == message
