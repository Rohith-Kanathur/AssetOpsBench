import hashlib
import json

import pytest

from scenarios.generation.review import check_contract, check_grounding, local_file


def save_contract(workspace, request, scenarios, allocation, tools=("iot.asset_detail",)):
    (workspace / "output").mkdir()
    for name in ("README.md", "requirements.txt", "environment.md"):
        (workspace / "output" / name).write_text("fixture")
    evidence = workspace / "output/response.json"
    evidence.write_text("{}\n")
    for scenario in scenarios:
        scenario.update(text="Inspect this asset", category="Inspection",
                        characteristic_form="Inspect the recorded evidence", source_ids=["fixture"])
        if scenario["positive"] is False:
            scenario["missing_evidence"] = "Requested measurements are unavailable"
    values = {
        "profile": {}, "allocation": allocation, "scenarios": scenarios,
        "sources": [{"id": "fixture", "kind": "observed", "url": "repository:HEAD:fixture.json",
                     "files": [{"path": "output/response.json",
                                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}],
        "tool_checks": [{"scenario_id": s["id"], "calls": [
            {"tool": tool, "arguments": {}, "response_file": "output/response.json"}
            for tool in tools]} for s in scenarios],
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


def test_known_asset_does_not_hide_an_invented_sensor_or_empty_window():
    def invoke(tool, args):
        if tool == "iot.asset_detail":
            return {"result": {"asset_id": "T1", "site_name": "S1"}}
        if tool == "iot.installed_sensors":
            return {"result": {"sensors": ["oil_temperature"]}}
        return {"result": {"total_records": 0}}
    scenario = {"id": 1, "type": "iot", "positive": True,
                "grounding": {"asset_id": "T1", "site": "S1",
                              "sensors": ["imaginary", "oil_temperature"]}}
    errors, evidence = check_grounding([scenario], invoke)
    assert "Scenario 1: unknown sensor imaginary" in errors
    assert any("no readings for oil_temperature" in error for error in errors)
    assert len(evidence) == 4


def test_cross_asset_workorder_is_rejected():
    def invoke(tool, args):
        if tool == "iot.asset_detail":
            return {"asset_id": "T1", "site_name": "S1"}
        return {"work_order": {"assetnum": "T2"}}
    errors, _ = check_grounding([{"id": 1, "type": "wo", "positive": True,
        "grounding": {"asset_id": "T1", "site": "S1", "workorder_ids": ["W1"]}}], invoke)
    assert errors == ["Scenario 1: work order W1 belongs to another asset"]


@pytest.mark.parametrize("budget_key", ["scenario_counts", "scenario_plan"])
def test_contract_accepts_budget_and_matching_allocation(tmp_path, budget_key):
    allocation = {"iot": {"positive": 1, "negative": 1}}
    budget = allocation["iot"] if budget_key == "scenario_counts" else allocation
    scenarios = [{"id": 1, "type": "iot", "positive": True},
                 {"id": 2, "type": "iot", "positive": False}]
    save_contract(tmp_path, {budget_key: budget}, scenarios, allocation)
    report = check_contract(tmp_path)
    assert report["errors"] == []
    assert report["positive"] == report["negative"] == 1


@pytest.mark.parametrize("positive", [True, False])
def test_contract_rejects_positive_negative_budget_mismatch(tmp_path, positive):
    allocation = {"iot": {"positive": int(not positive), "negative": int(positive)}}
    save_contract(tmp_path, {"scenario_plan": allocation},
                  [{"id": 1, "type": "iot", "positive": positive}], allocation)
    report = check_contract(tmp_path)
    assert report["errors"]
    assert report["positive"] == int(positive)
    assert report["negative"] == int(not positive)


@pytest.mark.parametrize("budget_request", [
    {"scenario_plan": {"iot": {"positive": 1, "negative": 0}}},
    {"scenario_counts": {"positive": 2, "negative": 0}},
])
def test_contract_rejects_allocation_that_changes_requested_budget(tmp_path, budget_request):
    allocation = {"wo": {"positive": 1, "negative": 0}}
    save_contract(tmp_path, budget_request, [{"id": 1, "type": "wo", "positive": True}], allocation)
    assert check_contract(tmp_path)["errors"]


def test_contract_preserves_source_and_checksum_checks(tmp_path):
    allocation = {"iot": {"positive": 0, "negative": 1}}
    save_contract(tmp_path, {"scenario_counts": allocation["iot"]},
                  [{"id": 1, "type": "iot", "positive": False}], allocation)
    scenarios = json.loads((tmp_path / "output/scenarios.json").read_text())
    scenarios[0]["source_ids"] = ["unknown"]
    (tmp_path / "output/scenarios.json").write_text(json.dumps(scenarios))
    (tmp_path / "output/response.json").write_text('{"changed": true}\n')
    report = check_contract(tmp_path)
    assert "Scenario 1: unresolved source IDs" in report["errors"]
    assert "Source checksum mismatch: output/response.json" in report["errors"]
    assert report["negative"] == 1


@pytest.mark.parametrize("tools, valid", [
    (("iot.asset_detail", "iot.installed_sensors"), False),
    (("iot.asset_detail", "utilities.current_time"), False),
    (("iot.asset_detail", "fmsr.list_failure_modes"), True),
])
def test_positive_multiagent_checks_require_two_domain_servers(tmp_path, tools, valid):
    allocation = {"multiagent": {"positive": 1, "negative": 0}}
    save_contract(tmp_path, {"scenario_plan": allocation},
                  [{"id": 1, "type": "multiagent", "positive": True}], allocation, tools)
    errors = check_contract(tmp_path)["errors"]
    if valid:
        assert errors == []
    else:
        assert errors == ["Scenario 1: positive multiagent tool checks must span at least two domain servers"]


def test_negative_multiagent_can_record_one_unavailable_domain(tmp_path):
    allocation = {"multiagent": {"positive": 0, "negative": 1}}
    save_contract(tmp_path, {"scenario_plan": allocation},
                  [{"id": 1, "type": "multiagent", "positive": False}], allocation)
    assert check_contract(tmp_path)["errors"] == []


@pytest.mark.parametrize("domain", ["vibration", "Vibration"])
def test_vibration_grounding_uses_vibration_sensor_tool(domain):
    def invoke(tool, args):
        if tool == "iot.asset_detail":
            return {"asset_id": "A1", "site_name": "S1"}
        assert tool == "vibration.list_vibration_sensors"
        return {"sensors": ["acceleration"]}
    errors, evidence = check_grounding([{"id": 1, "type": domain, "positive": True,
        "grounding": {"asset_id": "A1", "site": "S1"}}], invoke)
    assert errors == []
    assert evidence[-1]["tool"] == "vibration.list_vibration_sensors"
