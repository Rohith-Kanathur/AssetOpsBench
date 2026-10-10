"""Contract failures are repairable output errors, rather than checker crashes."""

import json

import pytest

from scenarios.generation.review import check_contract, check_grounding
from scenarios.generation.contracts import validate_scenarios
from scenarios.generation.tests.test_review import (
    one_contract, profile_fixture, replace_artifact, save_contract,
)


def test_profile_stage_does_not_require_scenarios_or_allocation(tmp_path):
    one_contract(tmp_path)
    for name in ("scenarios", "allocation"):
        (tmp_path / f"output/{name}.json").unlink()
    assert check_contract(tmp_path, stage="profile")["errors"] == []
    assert check_contract(tmp_path)["errors"]


def test_profile_enforces_research_and_resolves_evidence_without_frozen_tools(tmp_path):
    one_contract(tmp_path)
    profile = profile_fixture()
    profile["available_capabilities"]["fmsr"] = [{"tool": "fmsr.new_health_check", "description": "Added calculation"}]
    profile["research"]["diagnostics"] = {"status": "supported", "summary": "Documented calculation", "source_ids": ["fixture"]}
    replace_artifact(tmp_path, "profile", profile)
    assert check_contract(tmp_path, "profile", ["iot.asset_detail", "fmsr.new_health_check"])["errors"] == []
    errors = check_contract(tmp_path, "profile", ["iot.asset_detail"])["errors"]
    assert "Unavailable profile tool: fmsr.new_health_check" in errors
    profile["research"]["diagnostics"]["source_ids"] = ["unknown"]
    profile["operator_tasks"] = []
    replace_artifact(tmp_path, "profile", profile)
    assert any("operator_tasks" in e for e in check_contract(tmp_path, "profile")["errors"])
    assert any("unresolved source" in e for e in check_contract(tmp_path, "profile")["errors"])


@pytest.mark.parametrize("name,value", [("profile", []), ("sources", [None]), ("scenarios", [None])])
def test_malformed_artifacts_return_errors_not_tracebacks(tmp_path, name, value):
    one_contract(tmp_path)
    replace_artifact(tmp_path, name, value)
    assert check_contract(tmp_path)["errors"]


def test_malformed_nested_values_return_errors_not_tracebacks(tmp_path):
    one_contract(tmp_path)
    profile = profile_fixture()
    profile["research"]["diagnostics"]["status"] = []
    replace_artifact(tmp_path, "profile", profile)
    replace_artifact(tmp_path, "sources", [{"id": "fixture", "kind": [], "files": []}])
    scenario = json.loads((tmp_path / "output/scenarios.json").read_text())[0]
    scenario.update(id=[], source_ids=[{}], grounding={"scope": []})
    replace_artifact(tmp_path, "scenarios", [scenario])
    assert check_contract(tmp_path)["errors"]


def test_scenario_requires_string_rubric_and_detects_api_leakage(tmp_path):
    one_contract(tmp_path)
    scenario = json.loads((tmp_path / "output/scenarios.json").read_text())[0]
    scenario.update(text="Use iot.asset_detail to inspect T1", characteristic_form={"tools": ["iot.asset_detail"]})
    replace_artifact(tmp_path, "scenarios", [scenario])
    errors = check_contract(tmp_path)["errors"]
    assert "Scenario 1: characteristic_form must be a nonempty string" in errors
    assert "Scenario 1: text contains qualified API references" in errors


def test_unknown_rubric_tools_rejected_after_discovery(tmp_path):
    one_contract(tmp_path)
    scenario = json.loads((tmp_path / "output/scenarios.json").read_text())[0]
    scenario["characteristic_form"] = "Read iot.invented_diagnostic"
    replace_artifact(tmp_path, "scenarios", [scenario])
    assert "Scenario 1: unavailable rubric tool iot.invented_diagnostic" in check_contract(tmp_path, tools=["iot.asset_detail"])["errors"]


def test_near_duplicate_requests_are_rejected(tmp_path):
    allocation = {"iot": {"positive": 2, "negative": 0}}
    save_contract(tmp_path, {"scenario_plan": allocation}, [{"id": 1, "type": "iot", "positive": True},
                                                         {"id": 2, "type": "iot", "positive": True}], allocation)
    rows = json.loads((tmp_path / "output/scenarios.json").read_text())
    rows[0]["text"] = "Review T1 at S1 and summarize every available oil temperature observation over the last seven days."
    rows[1]["text"] = rows[0]["text"].replace("Review", "Inspect")
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("near-duplicate" in e for e in check_contract(tmp_path)["errors"])


def test_negative_requires_a_grounded_reason_without_manual_response_files(tmp_path):
    one_contract(tmp_path, positive=False)
    rows = json.loads((tmp_path / "output/scenarios.json").read_text())
    rows[0]["grounding"]["asset_id"] = "INTENTIONALLY_UNAVAILABLE"
    rows[0]["text"] = "Inspect INTENTIONALLY_UNAVAILABLE at S1"
    replace_artifact(tmp_path, "scenarios", rows)
    report = check_contract(tmp_path)
    assert report["errors"] == []
    assert report["scenario_execution"] == "not_verified"
    rows[0]["missing_evidence"][0]["reason"] = ""
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("dependency and reason" in e for e in check_contract(tmp_path)["errors"])
    rows[0]["missing_evidence"][0]["reason"] = "Unavailable"
    rows[0]["missing_evidence"][0]["gap_ids"] = ["invented-gap"]
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("unresolved missing-evidence gap" in e for e in check_contract(tmp_path)["errors"])
    calls = []
    errors, evidence = check_grounding(rows, lambda tool, args: calls.append(tool))
    assert errors == evidence == calls == []


def test_class_catalog_contract_requires_no_fabricated_registry_identity():
    scenario = {"id": 1, "type": "fmsr", "positive": True, "text": "List stored transformer failure modes",
                "category": "catalog", "characteristic_form": "Retrieve fmsr.get_failure_modes",
                "source_ids": ["fixture"], "grounding": {"scope": "class", "asset_class": "Transformer"}}
    errors, _ = validate_scenarios([scenario], {"fixture"}, profile_fixture(), ["fmsr.get_failure_modes"])
    assert errors == []
    scenario["grounding"]["asset_class"] = "Chiller"
    errors, _ = validate_scenarios([scenario], {"fixture"}, profile_fixture())
    assert "Scenario 1: grounded class does not match profile" in errors


def test_class_scope_cannot_omit_required_asset_references():
    scenario = {"id": 1, "type": "iot", "positive": True, "text": "Review oil readings",
                "category": "statistics", "characteristic_form": "Retrieve iot.sensor_stats",
                "source_ids": ["fixture"], "grounding": {"scope": "class", "asset_class": "Transformer",
                                                         "sensors": ["oil_temperature"]}}
    errors, _ = validate_scenarios([scenario], {"fixture"}, profile_fixture())
    assert any("requires justification" in e for e in errors)
    assert any("cannot hide asset-specific references" in e for e in errors)


def test_text_referencing_another_known_asset_is_rejected(tmp_path):
    one_contract(tmp_path)
    profile = profile_fixture()
    profile["assets"].append({"site": "S1", "asset_id": "T2", "sensors": [], "source_ids": ["fixture"]})
    replace_artifact(tmp_path, "profile", profile)
    scenarios = json.loads((tmp_path / "output/scenarios.json").read_text())
    scenarios[0]["text"] = "Inspect T2 at S1"
    replace_artifact(tmp_path, "scenarios", scenarios)
    assert any("instead of grounded asset" in e for e in check_contract(tmp_path)["errors"])
