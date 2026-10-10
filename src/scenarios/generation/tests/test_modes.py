"""Keep task feasibility separate from generator preparation capabilities."""

import json

import pytest

from scenarios.generation import cli
from scenarios.generation.modes import write_guidance
from scenarios.generation.review import check_contract
from scenarios.generation.tests.test_review import one_contract, replace_artifact


def fixture(workspace, mode="mcp-only", positive=True):
    one_contract(workspace, positive)
    request = json.loads((workspace / "request.json").read_text())
    request["generation_mode"] = mode
    (workspace / "request.json").write_text(json.dumps(request))
    rows = json.loads((workspace / "output/scenarios.json").read_text())
    rows[0]["execution"] = {"requires": ["mcp"], "input_files": [], "output_files": []}
    if not positive:
        rows[0]["missing_evidence"][0]["kind"] = "missing_sensor"
    replace_artifact(workspace, "scenarios", rows)
    return rows


def helper_output(workspace, mode):
    rows = fixture(workspace, mode)
    rows[0]["execution"] = {"requires": ["mcp", "general-execution"], "input_files": [],
                            "output_files": [{"path": "export.json", "created_by": "general-execution"}]}
    replace_artifact(workspace, "scenarios", rows)
    return rows


def test_mode_specific_prompt_does_not_restrict_the_generator_itself(tmp_path):
    (tmp_path / "request.json").write_text('{"generation_mode":"mcp-only"}')
    for name in ("profile.md", "generate.md"):
        (tmp_path / name).write_text("Base guidance\n")
    write_guidance(tmp_path)
    for name in ("profile.md", "generate.md"):
        text = (tmp_path / name).read_text()
        assert "Evaluation mode: MCP only" in text
        assert "You, the generator, still have" in text
        assert "Evaluation mode: general execution" not in text


def test_mcp_only_accepts_an_answer_and_rejects_a_helper_export(tmp_path):
    fixture(tmp_path)
    assert check_contract(tmp_path)["errors"] == []
    rows = json.loads((tmp_path / "output/scenarios.json").read_text())
    rows[0]["execution"]["requires"].append("general-execution")
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("unavailable in mcp-only" in error for error in check_contract(tmp_path)["errors"])


def test_general_execution_declares_outputs_without_manual_execution_evidence(tmp_path):
    helper_output(tmp_path, "general-execution")
    report = check_contract(tmp_path)
    assert report["errors"] == []
    assert report["scenario_execution"] == "not_verified"
    assert not (tmp_path / "export.json").exists()
    assert not (tmp_path / "output/tool_checks.json").exists()


def test_general_execution_output_requires_its_capability(tmp_path):
    rows = helper_output(tmp_path, "mcp-only")
    rows[0]["execution"]["requires"] = ["mcp"]
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("requires general-execution capability" in e for e in check_contract(tmp_path)["errors"])


def test_server_native_output_requires_a_discovered_producer(tmp_path):
    rows = fixture(tmp_path)
    rows[0]["execution"]["output_files"] = [{"path": "forecast.csv", "created_by": "tsfm.run_recipe"}]
    replace_artifact(tmp_path, "scenarios", rows)
    report = check_contract(tmp_path, tools=["iot.asset_detail", "tsfm.run_recipe"])
    assert report["errors"] == []
    assert report["scenario_execution"] == "not_verified"
    assert any("unavailable MCP output producer" in e
               for e in check_contract(tmp_path, tools=["iot.asset_detail"])["errors"])


@pytest.mark.parametrize("path", ["/tmp/export.csv", "../export.csv", "output/../../export.csv", "."])
def test_expected_output_paths_cannot_escape_workspace(tmp_path, path):
    rows = helper_output(tmp_path, "general-execution")
    rows[0]["execution"]["output_files"][0]["path"] = path
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("workspace-relative files" in e for e in check_contract(tmp_path)["errors"])


@pytest.mark.parametrize("kind", ["missing_file_writer", "missing_shell", "runtime_failure", [], None])
def test_negative_case_requires_a_domain_evidence_gap(tmp_path, kind):
    rows = fixture(tmp_path, positive=False)
    assert check_contract(tmp_path)["errors"] == []
    rows[0]["missing_evidence"][0]["kind"] = kind
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("domain gap kind" in error for error in check_contract(tmp_path)["errors"])


def test_declared_inputs_must_exist_and_cannot_contain_completed_outputs(tmp_path):
    rows = fixture(tmp_path)
    rows[0]["execution"]["input_files"] = ["missing.csv"]
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("invalid input file" in error for error in check_contract(tmp_path)["errors"])
    (tmp_path / "missing.csv").write_text("value\n4")
    rows[0]["execution"]["output_files"] = [{"path": "missing.csv", "created_by": "iot.history"}]
    replace_artifact(tmp_path, "scenarios", rows)
    assert any("cannot also be a supplied input" in error for error in check_contract(tmp_path)["errors"])


def test_legacy_runs_are_not_silently_classified_as_mcp_only(tmp_path):
    one_contract(tmp_path)
    report = check_contract(tmp_path)
    assert report["errors"] == []
    assert any("Legacy run" in warning for warning in report["warnings"])
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/request.json").write_bytes((tmp_path / "request.json").read_bytes())
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path), "--followup", "Continue"])


def test_followup_cannot_reclassify_an_existing_run(tmp_path):
    fixture(tmp_path, "general-execution")
    before = (tmp_path / "request.json").read_bytes()
    # CLI directories contain the actual workspace as a child.
    outer = tmp_path / "saved"
    (outer / "workspace").mkdir(parents=True)
    (outer / "workspace/request.json").write_bytes(before)
    with pytest.raises(SystemExit):
        cli.main(["run", str(outer), "--followup", "Continue", "--mode", "mcp-only"])
    assert (outer / "workspace/request.json").read_bytes() == before
