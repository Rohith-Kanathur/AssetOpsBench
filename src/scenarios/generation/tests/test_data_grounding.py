"""Data lineage cannot be replaced by a citation or an arbitrary synthetic fixture."""

import hashlib
import json

import pytest

from scenarios.generation.data_grounding import validate_data_sources
from scenarios.generation.review import check_contract
from scenarios.generation.tests.test_review import one_contract, replace_artifact


def add_extension(workspace, kind="synthetic"):
    one_contract(workspace)
    sources = json.loads((workspace / "output/sources.json").read_text())
    script = workspace / "extend.py"
    script.write_text("# Retain the measured column; shift the identifier for this test.\n")
    sources.append({
        "id": "extended", "role": "data", "kind": kind,
        "url": "repository:HEAD:extend.py", "input_source_ids": ["fixture"],
        "files": [{"path": "extend.py", "sha256": hashlib.sha256(script.read_bytes()).hexdigest()}],
        "transform": {"script": "extend.py", "description": "Remap the instance identifier",
                      "field_mapping": {"oil_temperature": "fixture.oil_temperature, unchanged"},
                      "assumptions": "Same measurements, fictional instance", "limitations": "No new physical observations"},
    })
    replace_artifact(workspace, "sources", sources)
    profile = json.loads((workspace / "output/profile.json").read_text())
    profile["assets"][0]["data_source_ids"] = ["extended"]
    replace_artifact(workspace, "profile", profile)
    rows = json.loads((workspace / "output/scenarios.json").read_text())
    rows[0]["grounding"]["data_source_ids"] = ["extended"]
    replace_artifact(workspace, "scenarios", rows)
    return sources


@pytest.mark.parametrize("kind", ["derived", "simulated", "synthetic"])
def test_extension_retains_observed_root_and_checked_transformation(tmp_path, kind):
    add_extension(tmp_path, kind)
    assert check_contract(tmp_path)["errors"] == []
    (tmp_path / "extend.py").write_text("# changed\n")
    assert "Source checksum mismatch: extend.py" in check_contract(tmp_path)["errors"]


@pytest.mark.parametrize("role", [None, "literature", "receipt", "code"])
def test_literature_or_receipt_cannot_substitute_for_data(tmp_path, role):
    sources = add_extension(tmp_path)
    sources[0]["role"] = role
    replace_artifact(tmp_path, "sources", sources)
    errors = check_contract(tmp_path, "profile")["errors"]
    assert any("input_source_ids must reference retained data" in e for e in errors)


@pytest.mark.parametrize("inputs", [None, [], ["missing"], ["extended"], [{}]])
def test_unrooted_or_circular_extensions_fail(tmp_path, inputs):
    sources = add_extension(tmp_path)
    sources[1]["input_source_ids"] = inputs
    replace_artifact(tmp_path, "sources", sources)
    assert "Data source extended: no complete lineage to observed data" in check_contract(tmp_path)["errors"]


def test_every_input_chain_must_be_grounded():
    base = {"role": "data", "kind": "observed"}
    sources = [{"id": "real", **base}, {"id": "invented", "role": "data", "kind": "synthetic"}]
    sources.append({"id": "mixed", "role": "data", "kind": "derived", "input_source_ids": ["real", "invented"],
                    "transform": {"script": "x.py", "description": "Mix", "field_mapping": {"x": "real.x"},
                                  "assumptions": "Test", "limitations": "Test"}, "files": {"x.py": "hash"}})
    errors, grounded = validate_data_sources(sources)
    assert grounded == {"real"}
    assert "Data source mixed: no complete lineage to observed data" in errors


@pytest.mark.parametrize("key", ["script", "description", "field_mapping", "assumptions", "limitations"])
def test_extension_requires_reproducibility_and_limits(tmp_path, key):
    sources = add_extension(tmp_path)
    del sources[1]["transform"][key]
    replace_artifact(tmp_path, "sources", sources)
    assert any("transform requires" in e for e in check_contract(tmp_path)["errors"])


def test_declared_observed_data_cannot_have_transformed_inputs(tmp_path):
    sources = add_extension(tmp_path, "observed")
    replace_artifact(tmp_path, "sources", sources)
    assert any("transformed records must be marked" in e for e in check_contract(tmp_path)["errors"])


def test_profile_coverage_and_scenario_must_identify_their_data(tmp_path):
    one_contract(tmp_path)
    profile = json.loads((tmp_path / "output/profile.json").read_text())
    profile["assets"][0]["iot"] = {"sensors": ["oil_temperature"]}
    replace_artifact(tmp_path, "profile", profile)
    assert any("Profile asset 0.iot: data_source_ids" in e for e in check_contract(tmp_path, "profile")["errors"])
    profile["assets"][0]["iot"]["data_source_ids"] = ["fixture"]
    replace_artifact(tmp_path, "profile", profile)
    rows = json.loads((tmp_path / "output/scenarios.json").read_text())
    rows[0]["grounding"].pop("data_source_ids")
    replace_artifact(tmp_path, "scenarios", rows)
    assert check_contract(tmp_path, "profile")["errors"] == []
    assert any("Scenario 1: data_source_ids" in e for e in check_contract(tmp_path)["errors"])


@pytest.mark.parametrize("value", [[], None, "unexplained", {"script": []}])
def test_malformed_transforms_are_errors(tmp_path, value):
    sources = add_extension(tmp_path)
    sources[1]["transform"] = value
    replace_artifact(tmp_path, "sources", sources)
    assert check_contract(tmp_path)["errors"]
