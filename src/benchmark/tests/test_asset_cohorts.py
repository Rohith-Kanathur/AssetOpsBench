"""Cohorts preserve the real tasks and use the existing execution/judging flow."""
import importlib.util
import json
from pathlib import Path

import pytest

from benchmark.asset_cohorts import existing_rows, prepare, stratified_sample, summarize_comparison, workorder_inventory
from benchmark.generated_suite_runner import completed_scenarios
from scenarios.config import GeneratorConfig


def corpus(tmp_path):
    rows = [{"id": i, "entity": "Chiller", "type": lane,
             "text": f"Query Chiller {i} at a site absent from local fixtures.",
             "category": "Data Query", "characteristic_form": f"Original rubric {i}"}
            for i, lane in enumerate(["IoT", "FMSR", "TSFM", "Workorder", "multiagent", "IoT"], 1)]
    rows.append({**rows[0], "id": 20, "entity": "Transformer", "text": "Compare a transformer with a chiller."})
    path = tmp_path / "original.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path, rows[:-1]


def test_existing_questions_and_rubrics_are_unchanged_and_not_filtered_for_coverage(tmp_path):
    path, source = corpus(tmp_path)
    actual = existing_rows("Chiller", path)
    assert len(actual) == len(source)
    assert [r["text"] for r in actual] == [r["text"] for r in source]
    assert [r["characteristic_form"] for r in actual] == [r["characteristic_form"] for r in source]
    assert actual[3]["type"] == "wo"
    assert actual[3]["provenance"]["source_type"] == "Workorder"


def test_pilot_is_reproducible_and_covers_each_lane(tmp_path):
    path, _ = corpus(tmp_path)
    rows = existing_rows("Chiller", path)
    assert stratified_sample(rows, 5, 7) == stratified_sample(rows, 5, 7)
    assert len({r["type"] for r in stratified_sample(rows, 5, 7)}) == 5
    assert stratified_sample(rows, None, 7) == rows
    with pytest.raises(ValueError, match="positive"):
        stratified_sample(rows, 0, 7)


def test_prepared_real_suite_loads_in_native_runner_and_matches_synthetic_plan(tmp_path):
    path, _ = corpus(tmp_path)
    out = tmp_path / "cohorts"
    prepare("Chiller", path, out, limit=5)
    rows, _ = completed_scenarios(out / "existing")
    config = GeneratorConfig.model_validate_json((out / "generation-config.json").read_text())
    assert config.live_data and config.num_scenarios == len(rows) == 5
    assert config.num_negative_scenarios == 0
    assert config.scenario_plan.positive_counts == {r.type: 1 for r in rows}
    assert json.loads((out / "existing/run.json").read_text())["suite_kind"] == "existing"
    targets = json.loads((out / "comparison-config.json").read_text())["targets"]
    assert len(targets) == 10
    assert targets[0]["model_id"] == targets[5]["model_id"]
    assert targets[0]["suite"] != targets[5]["suite"]
    with pytest.raises(FileExistsError):
        prepare("Chiller", path, out)


def test_summary_counts_final_execution_failures_without_inventing_grades(tmp_path):
    source, _ = corpus(tmp_path)
    cohorts = tmp_path / "cohorts"
    prepare("Chiller", source, cohorts, limit=1, execution_model="gpt-6.1-sol")
    config = json.loads((cohorts / "comparison-config.json").read_text())
    for spec in config["targets"]:
        suite = Path(spec["suite"])
        suite.mkdir(exist_ok=True)
        if spec["cohort"] == "synthetic":
            (suite / "scenarios.json").write_text((cohorts / "existing/scenarios.json").read_text())
        target = tmp_path / "results" / spec["name"]
        (target / "measurements").mkdir(parents=True)
        (target / "settings.json").write_text(json.dumps({"model": spec["model_id"], "agent": spec["agent"]}))
        case = json.loads((suite / "scenarios.json").read_text())[0]
        record = {"scenario_id": case["id"], "attempt": 3, "status": "failed", "metrics": {},
                  "settings": {"invocation_retry_policy": {"max_attempts": 3}}}
        for attempt in (1, 3):
            (target / "measurements" / f"case-{attempt}.json").write_text(json.dumps({**record, "attempt": attempt}))
    report = summarize_comparison(cohorts, tmp_path / "results", tmp_path / "summary.json")
    for models in report["cohorts"].values():
        cohort = models["openai"]
        assert cohort["attempted"] == 1
        assert cohort["execution_failed_cases"] == 1
        assert cohort["pass_rate"] == 0
        assert cohort["graded"] == 0
        assert cohort["mean_score"] is None


def test_summary_keeps_all_models_separate_and_rejects_foreign_scenario(tmp_path):
    source, _ = corpus(tmp_path)
    cohorts = tmp_path / "cohorts"
    prepare("Chiller", source, cohorts, limit=1)
    config = json.loads((cohorts / "comparison-config.json").read_text())
    for spec in config["targets"]:
        suite = Path(spec["suite"])
        suite.mkdir(exist_ok=True)
        if spec["cohort"] == "synthetic":
            (suite / "scenarios.json").write_text((cohorts / "existing/scenarios.json").read_text())
        target = tmp_path / "results" / spec["name"]
        target.mkdir(parents=True)
        (target / "settings.json").write_text(json.dumps({"model": spec["model_id"], "agent": spec["agent"]}))
    report = summarize_comparison(cohorts, tmp_path / "results", tmp_path / "summary.json")
    assert all(len(models) == 5 for models in report["cohorts"].values())
    assert {v["model_id"] for v in report["cohorts"]["existing"].values()} == {s["model_id"] for s in config["targets"]}
    assert all(v["assigned_cases"] == 1 and v["pass_rate"] is None
               for models in report["cohorts"].values() for v in models.values())
    bad = tmp_path / "results" / config["targets"][0]["name"] / "measurements"
    bad.mkdir()
    (bad / "case.json").write_text(json.dumps({"scenario_id": "foreign", "attempt": 1, "status": "completed"}))
    with pytest.raises(ValueError, match="does not belong"):
        summarize_comparison(cohorts, tmp_path / "results", tmp_path / "summary.json")


def test_workorder_audit_preserves_identifiers_and_explicit_classes(monkeypatch):
    from servers.wo import couch, workorders

    clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.closed = False
            clients.append(self)

        async def aclose(self):
            self.closed = True

    async def listing(client, *, page_size):
        assert page_size == 0
        return {"success": True, "data": {"workorders": [
            {"wonum": "1", "siteid": "MAIN", "assetnum": "CHILLER6"},
            {"wonum": "2", "siteid": "MAIN", "assetnum": "unrelated-name", "aob_asset_class": "Chiller"},
            {"wonum": "3", "siteid": "MAIN", "assetnum": "CHILLER6", "aob_asset_class": "Transformer"},
            {"wonum": "4", "siteid": "MAIN", "assetnum": "CHILLERIUS"},
        ]}}

    monkeypatch.setattr(couch, "CouchClient", FakeClient)
    monkeypatch.setattr(workorders, "list_workorders", listing)
    result = workorder_inventory("Chiller")
    assert result["available"] and result["matching_count"] == 2
    assert result["asset_references"] == [{"site_id": "MAIN", "asset_num": "CHILLER6"},
                                          {"site_id": "MAIN", "asset_num": "unrelated-name"}]
    assert clients[0].closed

    async def missing(client, *, page_size):
        return {"success": False, "error": "Database unavailable"}

    monkeypatch.setattr(workorders, "list_workorders", missing)
    result = workorder_inventory("Chiller")
    assert not result["available"] and result["matching_count"] is None
    assert clients[-1].closed


def test_launcher_resolves_each_cohort_suite_and_preserves_legacy_default(tmp_path):
    script = Path(__file__).resolve().parents[3] / "tools/run_generated_comparison.py"
    spec = importlib.util.spec_from_file_location("comparison_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"suite": "legacy/suite"}
    assert module.target_suite(config, {}) == module.ROOT / "legacy/suite"
    assert module.target_suite(config, {"suite": str(tmp_path)}) == tmp_path


@pytest.mark.parametrize("separate_suites", [False, True])
def test_report_renders_custom_target_names_and_honors_cohort_config(tmp_path, separate_suites):
    from benchmark.comparison_report import render

    root = tmp_path / "results"
    specs = []
    for index, (name, key, model) in enumerate([
        ("existing-custom-sol", "gpt-6-1-sol", "gpt-6.1-sol"),
        ("existing-custom-new", "new-model", "provider/custom-model"),
    ]):
        suite = tmp_path / (f"suite-{index}" if separate_suites else "suite")
        suite.mkdir(exist_ok=True)
        (suite / "scenarios.json").write_text(json.dumps([
            {"id": "1", "type": "iot", "text": "Query the real chiller."},
        ]))
        spec = {"name": name, "model_key": key, "model_id": model, "suite": str(suite)}
        specs.append(spec)
        target = root / name / "measurements"
        target.mkdir(parents=True)
        record = {"scenario_id": "1", "run_id": f"{name}_0001", "execution_index": 1,
                  "attempt": 1, "status": "completed", "execution_duration_ms": 25,
                  "settings": {"generation_run": str(suite), "model": model}, "metrics": {},
                  "grading": {"status": "completed", "duration_ms": 5,
                              "result": {"score": {"passed": True, "score": 1, "details": {}}}}}
        (target / "case.json").write_text(json.dumps(record))
    config = {"suite": "unused-legacy-default", "judge": "claude-code/claude-fable-5-1",
              "targets": specs}
    output = tmp_path / "report.html"
    render(root, output, config=config)
    document = output.read_text()
    stats = json.loads(output.with_suffix(".json").read_text())
    assert set(stats) == {spec["name"] for spec in specs}
    assert all(row["graded"] == 1 and row["pass_rate"] == 1 for row in stats.values())
    if separate_suites:
        assert 'id="snapshot-data"' not in document
        assert all(spec["name"] in document for spec in specs)
    else:
        payload = json.loads(document.split('id="snapshot-data">', 1)[1].split("</script>", 1)[0])
        assert payload["judge"] == config["judge"]
        assert [model["key"] for model in payload["models"]] == [spec["name"] for spec in specs]
        assert [model["model_id"] for model in payload["models"]] == [spec["model_id"] for spec in specs]
        assert [model["name"] for model in payload["models"]] == ["GPT-6.1 Sol", "existing-custom-new"]
        assert all(model["rows"][0]["question"] == "Query the real chiller."
                   for model in payload["models"])


def test_synthetic_report_uses_configured_suite_when_existing_results_sort_first(tmp_path):
    from benchmark.comparison_report import render

    root = tmp_path / "shared-repetition"
    suites = {}
    for cohort in ("existing", "synthetic"):
        suite = tmp_path / "suites" / cohort
        suite.mkdir(parents=True)
        suites[cohort] = suite
        sid = f"{cohort}-case"
        question = f"Inspect the {cohort} chiller task."
        (suite / "scenarios.json").write_text(json.dumps([
            {"id": sid, "type": "iot", "text": question},
        ]))
        name = f"{cohort}-sol"
        target = root / name / "measurements"
        target.mkdir(parents=True)
        record = {"scenario_id": sid, "run_id": f"{name}_0001", "execution_index": 1,
                  "attempt": 1, "status": "completed", "execution_duration_ms": 25,
                  "settings": {"generation_run": str(suite), "model": "gpt-6.1-sol"},
                  "metrics": {}, "grading": {"status": "completed", "duration_ms": 5,
                  "result": {"score": {"passed": True, "score": 1, "details": {}}}}}
        (target / "case.json").write_text(json.dumps(record))
    config = {"suite": str(suites["existing"]), "judge": "claude-code/claude-fable-5-1",
              "targets": [{"name": "synthetic-sol", "model_key": "gpt-6-1-sol",
                           "model_id": "gpt-6.1-sol", "suite": str(suites["synthetic"])}]}
    output = tmp_path / "synthetic-report.html"
    render(root, output, config=config)
    document = output.read_text()
    payload = json.loads(document.split('id="snapshot-data">', 1)[1].split("</script>", 1)[0])
    assert [model["key"] for model in payload["models"]] == ["synthetic-sol"]
    rows = payload["models"][0]["rows"]
    assert [(row["id"], row["question"], row["status"]) for row in rows] == [
        ("synthetic-case", "Inspect the synthetic chiller task.", "pass"),
    ]
