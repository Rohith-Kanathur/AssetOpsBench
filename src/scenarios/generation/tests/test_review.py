import json

import pytest

from scenarios.generation.review import check_contract, check_grounding, local_file


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
    scenario = {"id": 1, "type": "IoT", "positive": True,
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
    errors, _ = check_grounding([{"id": 1, "type": "WO", "positive": True,
        "grounding": {"asset_id": "T1", "site": "S1", "workorder_ids": ["W1"]}}], invoke)
    assert errors == ["Scenario 1: work order W1 belongs to another asset"]


def test_contract_reports_missing_domain_and_unresolved_source(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "scripts").mkdir()
    for name in ("output/README.md", "output/requirements.txt", "output/environment.md"):
        (tmp_path / name).write_text("fixture")
    for name, value in {"profile": {}, "sources": [], "tool_checks": [],
                       "scenarios": [{"id": 1, "type": "IoT", "positive": False,
                                      "source_ids": ["unknown"]}]}.items():
        (tmp_path / f"output/{name}.json").write_text(json.dumps(value))
    (tmp_path / "request.json").write_text(json.dumps({"count": 5,
        "domains": ["IoT", "FMSR", "TSFM", "WO", "Vibration"]}))
    report = check_contract(tmp_path)
    assert "Scenario count or domains do not match request.json" in report["errors"]
    assert "Scenario 1: unresolved source IDs" in report["errors"]
    assert report["negative"] == 1
