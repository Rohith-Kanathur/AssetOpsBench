"""Generated environments must be frozen, isolated and free of prepared answers."""

import json
from pathlib import Path

import pytest

from benchmark.generated import cli, sandbox


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def generated(tmp_path, monkeypatch):
    generation = tmp_path / "generation"
    workspace = generation / "workspace"
    request = {"asset_class": "Transformer", "generation_mode": "general-execution"}
    save(generation / "status.json", {"status": "complete"})
    save(generation / "baseline.json", {"baseline": "upstream-before-generated-changes"})
    save(generation / "request.json", request)
    save(workspace / "request.json", request)
    save(workspace / "src/servers/new_tool.json", {"generated": True})
    save(workspace / "data/input.json", {"observed": [1, 2]})
    save(workspace / "data/evidence/answer.json", {"answer": 3})
    save(workspace / "data/exports/old.json", {"answer": 3})
    save(workspace / "data/prepared-answer.json", {"answer": 3})
    save(workspace / "output/rubric.json", {"solution": "secret"})
    scenario = {"id": 1, "text": "Analyze the input", "positive": True, "type": "iot",
                "execution": {"requires": ["mcp", "general-execution"],
                              "input_files": ["data/input.json"],
                              "output_files": [{"path": "data/prepared-answer.json"}]}}
    save(workspace / "output/scenarios.json", [scenario])
    monkeypatch.setattr(sandbox.runtime, "start", lambda _: None)

    def export(*args, **kwargs):
        target = next(value for value in args if isinstance(value, str) and value.endswith(":/snapshot"))
        save(Path(target.removesuffix(":/snapshot")) / "generated-database.json", [{"generated": True}])

    monkeypatch.setattr(sandbox.runtime, "compose", export)
    return generation, scenario


def test_snapshot_and_case_use_final_source_data_and_only_declared_agent_inputs(tmp_path, monkeypatch):
    generation, scenario = generated(tmp_path, monkeypatch)
    root, case = tmp_path / "results", tmp_path / "case"
    sandbox.snapshot(generation, root)
    sandbox.prepare_case(root, case, scenario)
    assert json.loads((case / "tools-workspace/src/servers/new_tool.json").read_text()) == {"generated": True}
    assert json.loads((root / "database/generated-database.json").read_text()) == [{"generated": True}]
    assert sorted(str(p.relative_to(case / "workspace")) for p in (case / "workspace").rglob("*") if p.is_file()) == ["data/input.json"]
    assert not (case / "tools-workspace/data/evidence").exists()
    assert not (case / "tools-workspace/data/exports").exists()
    assert not (case / "tools-workspace/data/prepared-answer.json").exists()
    assert not (case / "tools-workspace/output").exists()
    config = json.loads((case / "compose.json").read_text())
    assert "volumes" not in config["services"]["database"]
    assert config["services"]["database"]["environment"]["ERL_FLAGS"] == "+S 2:2 +SDcpu 1 +SDio 2 +A 4"
    tools = config["services"]["tools"]
    assert tools["environment"]["TSFM_WORKDIR"] == "/workspace/artifacts"
    assert tools["environment"]["PYTHONPATH"] == "/environment/src"
    assert f"{case / 'workspace'}:/workspace" in tools["volumes"]
    assert f"{case / 'tools-workspace'}:/environment:ro" in tools["volumes"]
    assert tools["working_dir"] == "/workspace"
    with pytest.raises(ValueError, match="already exists"):
        sandbox.prepare_case(root, case, scenario)


@pytest.mark.parametrize("modified", ["environment/src/servers/new_tool.json", "database/generated-database.json", "inputs/data/input.json", "scenarios.json"])
def test_reuse_rejects_mutation_of_any_frozen_component(tmp_path, monkeypatch, modified):
    generation, _ = generated(tmp_path, monkeypatch)
    root = tmp_path / "results"
    sandbox.snapshot(generation, root)
    sandbox.snapshot(generation, root)
    save(root / modified, {"changed": True})
    with pytest.raises(ValueError, match="snapshot files changed"):
        sandbox.snapshot(generation, root)


def test_reuse_rejects_source_drift_and_incompatible_snapshot_format(tmp_path, monkeypatch):
    generation, _ = generated(tmp_path, monkeypatch)
    root = tmp_path / "results"
    sandbox.snapshot(generation, root)
    save(generation / "workspace/src/servers/new_tool.json", {"changed": True})
    with pytest.raises(ValueError, match="source or inputs changed"):
        sandbox.snapshot(generation, root)
    metadata = json.loads((root / "snapshot.json").read_text())
    metadata.pop("format_version")
    save(root / "snapshot.json", metadata)
    with pytest.raises(ValueError, match="Incompatible"):
        sandbox.snapshot(generation, root)


@pytest.mark.parametrize("relative", ["../secret", "/tmp/secret", "data/link.json", "alias/input.json"])
def test_inputs_reject_traversal_and_symlinked_ancestors(tmp_path, relative):
    save(tmp_path / "data/input.json", [1])
    (tmp_path / "data/link.json").symlink_to(tmp_path / "data/input.json")
    (tmp_path / "alias").symlink_to(tmp_path / "data", target_is_directory=True)
    with pytest.raises(ValueError):
        sandbox.checked_file(tmp_path, relative)


def test_source_symlinks_rejected_before_copy_and_failed_export_leaves_no_partial_snapshot(tmp_path, monkeypatch):
    generation, _ = generated(tmp_path, monkeypatch)
    link = generation / "workspace/src/servers/link"
    link.symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symlink"):
        sandbox.snapshot(generation, tmp_path / "results")
    link.unlink()
    monkeypatch.setattr(sandbox.runtime, "compose", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed export")))
    with pytest.raises(RuntimeError, match="failed export"):
        sandbox.snapshot(generation, tmp_path / "results")
    assert list((tmp_path / "results").iterdir()) == []


def test_every_case_drops_prior_database_before_restore_and_cleans_up_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSETOPS_DATABASE_MODE", "dedicated")
    monkeypatch.setenv("ASSETOPS_CACHE_RUNTIME_IMAGE", "0")
    generation, scenario = generated(tmp_path, monkeypatch)
    root, case = tmp_path / "results", tmp_path / "case"
    sandbox.snapshot(generation, root)
    calls = []

    def compose(case, *args, **kwargs):
        calls.append(args)
        if args[0] == "run":
            raise RuntimeError("restore failed")

    monkeypatch.setattr(sandbox, "compose", compose)
    with pytest.raises(RuntimeError, match="restore failed"):
        with sandbox.environment(root, case, scenario):
            pytest.fail("restore failure must not yield an environment")
    assert calls[0][:3] == ("down", "--volumes", "--remove-orphans")
    assert calls[-1][:3] == ("down", "--volumes", "--remove-orphans")
    assert any(args[0] == "logs" for args in calls)


def test_interrupted_case_is_archived_before_fresh_execution(tmp_path, monkeypatch):
    case = tmp_path / "cases/one"
    scenario = {"id": 1, "positive": True, "type": "iot", "text": "Question"}
    save(case / "result.json", {"status": "running"})
    save(case / "workspace/answer.json", {"old": True})
    save(case / "auth/.claude/.credentials.json", {"token": "test-only"})
    save(case / "compose.json", {})
    stopped = []
    monkeypatch.setattr(sandbox, "compose", lambda *args, **kwargs: stopped.append(args))
    def fresh(*args):
        assert not (case / "workspace").exists()
        raise RuntimeError("stop before actual execution")
    monkeypatch.setattr(sandbox, "environment", fresh)
    monkeypatch.setattr(cli, "judge_case", lambda _: None)
    result = cli.execute_case(tmp_path, case, scenario, "codex", "model", 1, {})
    assert result["status"] == "error"
    assert len(list((tmp_path / "attempts").glob("*/workspace/answer.json"))) == 1
    assert not list((tmp_path / "attempts").glob("*/auth"))
    assert stopped[0][1:4] == ("down", "--volumes", "--remove-orphans")


def test_subset_execution_retains_full_cohort_and_rejects_model_changes(tmp_path, monkeypatch):
    monkeypatch.setattr('benchmark.generated.pipeline.shared_admission.allow', lambda _: True)
    root = tmp_path / "results"
    save(root / "snapshot.json", {"request": {"generation_mode": "general-execution", "asset_class": "Transformer"}})
    save(root / "scenarios.json", [{"id": i, "positive": True, "type": "iot", "text": f"Question {i}"} for i in (1, 2)])
    monkeypatch.setattr(sandbox, "snapshot", lambda *_: None)
    executed, reports = [], []
    def execute(root, case, scenario, runner, model, *args, **kwargs):
        executed.append(scenario["id"])
        record = json.loads((case / "result.json").read_text())
        record["status"] = "completed"
        save(case / "result.json", record)
        return {**record, "grading": {"status": "pending"}}
    monkeypatch.setattr(cli, "execute_case", execute)
    monkeypatch.setattr(cli, "write_report", lambda root, cases, **kwargs: reports.append(cases))
    args = [str(tmp_path / "generation"), str(root), "--runners", '{"codex":"model-a"}', "--ids", "1", "--no-judge"]
    cli.main(args)
    assert executed == [1]
    assert len(reports[-1]) == 2
    assert {row["status"] for row in reports[-1]} == {"pending", "completed"}
    with pytest.raises(SystemExit):
        cli.main([*args[:2], "--runners", '{"codex":"model-b"}'])
    assert len(reports[-1]) == 2


def test_another_scenarios_prepared_output_is_not_an_input(tmp_path, monkeypatch):
    generation, scenario = generated(tmp_path, monkeypatch)
    second = {**scenario, "id": 2, "execution": {"input_files": ["data/prepared-answer.json"], "output_files": []}}
    save(generation / "workspace/output/scenarios.json", [scenario, second])
    with pytest.raises(ValueError, match="generation evidence"):
        sandbox.snapshot(generation, tmp_path / "results")


def test_shared_outputs_are_inventoried_once_without_private_resources(tmp_path):
    save(tmp_path / "workspace/data/input.json", [1])
    save(tmp_path / "workspace/deliverables/created.json", {"created_by": "agent"})
    save(tmp_path / "workspace/artifacts/analysis.json", {"created_by": "mcp"})
    save(tmp_path / "tools-workspace/src/private.json", {"secret": True})
    scenario = {"execution": {"input_files": ["data/input.json"]}}
    artifacts = sandbox.artifact_inventory(tmp_path, scenario)
    assert [item["path"] for item in artifacts] == ["artifacts/analysis.json", "deliverables/created.json"]
    assert all(item["location"] == "workspace" for item in artifacts)


def test_zcode_selection_uses_distinct_image_and_only_coding_plan_key(tmp_path, monkeypatch):
    assert cli.runner_models({"zcode": "GLM-5.3"}, "general-execution") == [("zcode", "GLM-5.3")]
    assert set(cli.DEFAULTS["general-execution"]) == {"stirrup"}
    with pytest.raises(ValueError, match="general execution"):
        cli.runner_models({"zcode": "GLM-5.3"}, "mcp-only")
    save(tmp_path / "compose.json", {"services": {}})
    monkeypatch.setattr(cli, "prepare_auth", lambda *args: pytest.fail("No personal login should be copied"))
    def compose(case, *args, **kwargs):
        config = json.loads((case / "compose.json").read_text())
        assert config["services"]["agent"]["image"] == "assetops-zcode-evaluation:local"
        assert "private-plan-test-key" not in json.dumps(config) + " ".join(args)
        assert kwargs["env"]["ZAI_API_KEY"] == "private-plan-test-key"
        assert args[args.index("--harness") + 1] == "zcode"
        assert args[args.index("--reasoning-effort") + 1] == "high"
        assert (case / "auth").is_dir() and not list((case / "auth").rglob("*"))
        save(case / "native/result.json", {"status": "completed", "answer": "Saved"})
    monkeypatch.setattr(sandbox, "compose", compose)
    result = cli.coding_case(tmp_path, "zcode", "GLM-5.3", 30, {"ZAI_API_KEY": "private-plan-test-key"})
    assert result["status"] == "completed" and not (tmp_path / "auth").exists()
