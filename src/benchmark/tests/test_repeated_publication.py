"""Moving repetition 1 must preserve evidence and remove external report dependencies."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from benchmark.measurement import suite_hash, write_json
from benchmark.repeated_comparison import load_experiment


def publisher():
    script = Path(__file__).resolve().parents[3] / "tools/publish_repeated_comparison.py"
    spec = importlib.util.spec_from_file_location("repeated_publication", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_published_repetition_copy_preserves_measurements_and_compressed_traces(tmp_path):
    module = publisher()
    source = tmp_path / "legacy/models"
    trace = source / "executor/events/case.jsonl.gz"
    trace.parent.mkdir(parents=True)
    trace.write_bytes(gzip.compress(b'{"kind":"final_answer","answer":"real evidence"}\n', mtime=0))
    record = {"scenario_id": "case", "attempt": 1, "score": .7,
              "trace_file": "models/executor/events/case.jsonl.gz", "grading": None}
    write_json(source / "executor/measurements/case.json", record)
    expected = hashes(source)
    dest = tmp_path / "report/repetition-1"
    module.export_repetition(source, dest, tmp_path / "suite", lambda text: text)
    assert hashes(dest / "models") == expected
    assert (dest / record["trace_file"]).is_file()
    assert hashes(source) == expected


def test_repeated_report_rebuilds_after_original_report_tree_is_removed(tmp_path, monkeypatch):
    module = publisher()
    config = json.loads((module.ROOT / "benchmarks/generated-comparison.json").read_text())
    legacy = tmp_path / "legacy"
    suite = legacy / "suite"
    write_json(suite / "run.json", {"status": "complete", "config": {
        "num_scenarios": 1, "num_negative_scenarios": 0}, "negative_count": 0})
    write_json(suite / "scenarios.json", [{"id": "case", "type": "iot", "text": "Inspect real data."}])
    write_json(suite / "negative_scenarios.json", [])
    (suite / "generation-evidence").mkdir()
    (suite / "generation-evidence/prompt.txt").write_text("Use registered assets and observed data.")
    digest = suite_hash([suite / "scenarios.json", suite / "negative_scenarios.json"])
    repetitions, evidence = [], {}
    for index in (1, 2, 3):
        folder = legacy / f"run-{index}"
        root = folder / "models"
        repetitions.append({"index": index, "root": str(root), "status": "completed"})
        for target in config["targets"]:
            name = target["name"]
            trace = root / name / "events/case.jsonl.gz"
            judge_trace = root / name / "events/case.judge.jsonl.gz"
            trace.parent.mkdir(parents=True)
            trace.write_bytes(gzip.compress(b'{"kind":"final_answer","answer":"observed data"}\n', mtime=0))
            judge_trace.write_bytes(gzip.compress((json.dumps({"kind": "judge_result", "payload": {
                "session_id": f"independent-{index}-{name}"}}) + "\n").encode(), mtime=0))
            settings = {"model": target["model_id"], "agent": target["agent"], "suite_sha256": digest,
                        "judge_model": config["judge"], "database_policy": {"snapshot_sha256": "snapshot"}}
            record = {"scenario_id": "case", "run_id": "case", "execution_index": 1,
                      "attempt": 1, "status": "completed", "execution_duration_ms": index * 10,
                      "trace_file": f"models/{name}/events/case.jsonl.gz", "settings": settings,
                      "metrics": {}, "grading": {"status": "completed", "duration_ms": 5,
                      "trace_file": f"models/{name}/events/case.judge.jsonl.gz", "attempts": [],
                      "result": {"score": {"passed": True, "score": 1, "details": {}}}}}
            write_json(root / name / "measurements/case.json", record)
            write_json(root / name / "settings.json", settings)
            write_json(root / name / "target.json", {"generation_run": "suite", "model": target["model_id"]})
            write_json(root / name / "environment.json", {"snapshot_sha256": "snapshot"})
            write_json(root / name / "trajectories/case.json", {"answer": "observed data"})
        (folder / "comparison.html").write_text("Obsolete dashboard")
        evidence[index] = hashes(root)
    experiment = {"status": "completed", "k": 3, "suite": str(suite),
                  "baseline": repetitions[0]["root"], "snapshot_sha256": "snapshot",
                  "repetitions": repetitions}
    path = legacy / "experiment.json"
    write_json(path, experiment)
    expected = deepcopy(module.rows_for(load_experiment(path), config))
    monkeypatch.setattr(module, "make_report", lambda dest, *_: (dest / "README.md").write_text("Report"))
    dest = tmp_path / "report"
    module.publish(path, dest, tmp_path / "missing.env")
    for index in (1, 2, 3):
        assert hashes(dest / f"repetition-{index}/models") == evidence[index]
    assert not list(dest.rglob("comparison.html"))
    assert (dest / "repetition-1/suite").resolve() == dest / "suite"
    assert (dest / "suite/generation-evidence/prompt.txt").read_text() == "Use registered assets and observed data."
    portable = json.loads((dest / "experiment.json").read_text())
    assert portable["baseline"] == "repetition-1/models"
    assert [row["root"] for row in portable["repetitions"]] == [f"repetition-{i}/models" for i in (1, 2, 3)]
    shutil.rmtree(legacy)
    assert module.rows_for(load_experiment(dest / "experiment.json"), config) == expected
    rebuilt = tmp_path / "rebuilt"
    module.publish(dest / "experiment.json", rebuilt, tmp_path / "missing.env")
    assert module.rows_for(load_experiment(rebuilt / "experiment.json"), config) == expected
    assert not list(rebuilt.rglob("comparison.html"))
    assert hashes(rebuilt / "repetition-1/models") == evidence[1]
