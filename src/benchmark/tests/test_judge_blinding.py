"""Judge-only blinding preserves task evidence and all authoritative originals."""

import hashlib
import json

import pytest

from benchmark.generated.blinding import prepare_view
from benchmark.generated.judge import CRITERIA, judge_case
from benchmark.generated.report import load_case
from llm.base import LLMBackend


class Backend(LLMBackend):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, temperature=0):
        self.calls += 1
        self.prompt = prompt
        return json.dumps({key: key != "hallucinations" for key in CRITERIA})


def make_case(root, *, source="human", model="openai/gpt-5.6-luna"):
    root.mkdir(parents=True)
    scenario = {"id": f"{source}-9", "text": "Forecast Chiller 6's load.",
                "characteristic_form": "Read the June measurements and save the forecast.",
                "source": source, "source_ids": [9], "grounding": {"authored_by": source}}
    execution = {"scenario_id": scenario["id"], "model": model, "runner": "stirrup",
                 "status": "completed", "source": source, "case_dir": str(root),
                 "answer": "The forecast is saved.", "settings": {"model": model},
                 "trajectory": {"model": model, "scenario_id": scenario["id"], "turns": [{
                     "index": 0, "source": "agent", "input_tokens": 101,
                     "text": "Check the measured load first.", "tool_calls": [{
                         "id": "call-1", "name": "forecast", "duration_ms": 42,
                         "input": {"model": "ttm-r2", "source": "observed"},
                         "output": {"model": "ttm-r2", "rows": [1, 2, 3]},
                     }]}]}}
    (root / "scenario.json").write_text(json.dumps(scenario))
    (root / "result.json").write_text(json.dumps(execution))
    (root / "native").mkdir()
    (root / "native/events.jsonl").write_text(json.dumps({"model": model, "source": source}))
    return scenario, execution


def test_blinds_prompt_files_and_paths_without_changing_originals(tmp_path):
    case = tmp_path / "human-gpt-5.6-luna"
    scenario, execution = make_case(case)
    workspace = case / "workspace"
    workspace.mkdir()
    relative = "human-forecast-gpt-5.6-luna.json"
    report = workspace / relative
    report.write_text(json.dumps({"producer": execution["model"], "load": 42,
                                  "file": str(report)}))
    original_content = report.read_bytes()
    execution["answer"] = f"I am ChatGPT (GPT-5.6). Forecast saved at {report}."
    execution["trajectory"]["source"] = "human"
    execution["trajectory"]["turns"][0]["tool_calls"][0]["output"]["file"] = str(report)
    execution["artifacts"] = [{"location": "workspace", "path": relative,
                               "bytes": len(original_content),
                               "sha256": hashlib.sha256(original_content).hexdigest(),
                               "preview": original_content.decode()}]
    (case / "result.json").write_text(json.dumps(execution))
    originals = {str(p.relative_to(case)): p.read_bytes() for p in case.rglob("*") if p.is_file()}

    backend = Backend()
    assert judge_case(case, backend=backend)["status"] == "completed"
    view = case / "judging/evidence"
    visible_scenario = json.loads((view / "scenario.json").read_text())
    visible_execution = json.loads((view / "result.json").read_text())
    content = backend.prompt + "\n".join(p.read_text() for p in view.rglob("*") if p.is_file())
    for identity in ("gpt-5.6-luna", "gpt-5.6", "chatgpt", "openai", "stirrup", str(case), "human-9", relative):
        assert identity.lower() not in content.lower()
    assert set(visible_scenario) == {"id", "text", "characteristic_form"}
    assert visible_scenario["id"] == "case"
    assert visible_scenario["characteristic_form"] == scenario["characteristic_form"]
    assert "source" not in visible_execution and "settings" not in visible_execution
    assert "source" not in visible_execution["trajectory"]
    turn = visible_execution["trajectory"]["turns"][0]
    assert turn["source"] == "agent" and turn["input_tokens"] == 101
    call = turn["tool_calls"][0]
    assert call["id"] == "call-1" and call["duration_ms"] == 42
    assert call["input"] == {"model": "ttm-r2", "source": "observed"}
    assert call["output"]["rows"] == [1, 2, 3]
    artifact = visible_execution["artifacts"][0]
    data = (view / "workspace" / artifact["path"]).read_bytes()
    assert artifact["sha256"] == hashlib.sha256(data).hexdigest()
    assert artifact["bytes"] == len(data)
    assert artifact["preview"] == data.decode()
    assert call["output"]["file"] == "/evidence/workspace/" + artifact["path"]
    assert not (view / "blinding.json").exists()
    audit = json.loads((case / "judging/blinding.json").read_text())
    assert audit["model"] == execution["model"]
    assert audit["files"][relative]["original_sha256"] == hashlib.sha256(original_content).hexdigest()
    for path, original in originals.items():
        assert (case / path).read_bytes() == original


def test_source_and_model_do_not_change_equivalent_visible_evidence(tmp_path):
    views = []
    for source, model in [("human", "openai/gpt-5.6-luna"), ("synthetic", "anthropic/claude-fable-5-1")]:
        case = tmp_path / source
        scenario, execution = make_case(case, source=source, model=model)
        views.append(prepare_view(case, case / "judging/evidence", scenario, execution))
    assert views[0] == views[1]


def test_artifact_changes_invalidate_grade_and_archive_previous_view(tmp_path):
    case = tmp_path / "case"
    make_case(case)
    (case / "workspace").mkdir()
    report = case / "workspace/report.txt"
    report.write_text("first result")
    backend = Backend()
    first = judge_case(case, backend=backend)
    assert judge_case(case, backend=backend) == first and backend.calls == 1
    report.write_text("corrected result")
    assert load_case(case)["grading"]["status"] == "pending"
    second = judge_case(case, backend=backend)
    assert second["fingerprint"] != first["fingerprint"] and backend.calls == 2
    archive = next((case / "judging-attempts").iterdir())
    assert (archive / "judging/evidence/workspace/report.txt").read_text() == "first result"
    assert (case / "judging/evidence/workspace/report.txt").read_text() == "corrected result"


def test_old_unblinded_grade_and_interrupted_view_are_not_reused(tmp_path):
    case = tmp_path / "case"
    make_case(case)
    (case / "judge.json").write_text(json.dumps({"status": "completed", "fingerprint": "old"}))
    orphan = case / "judging/evidence"
    orphan.mkdir(parents=True)
    (orphan / "result.json").write_text("old unblinded evidence")
    backend = Backend()
    assert judge_case(case, backend=backend)["status"] == "completed" and backend.calls == 1
    (case / "judge.json").unlink()  # Simulate interruption before the grade was saved.
    assert judge_case(case, backend=backend)["status"] == "completed" and backend.calls == 2
    assert len(list((case / "judging-attempts").iterdir())) == 2


@pytest.mark.parametrize("unsafe", ["binary", "symlink", "path_collision"])
def test_unsafe_blinding_does_not_call_judge_or_mutate_data(tmp_path, unsafe):
    case = tmp_path / "case"
    make_case(case)
    workspace = case / "workspace"
    workspace.mkdir()
    if unsafe == "binary":
        path = workspace / "report.bin"
        path.write_bytes(b"\xff\x00producer:gpt-5.6-luna")
    elif unsafe == "symlink":
        path = workspace / "report.json"
        path.symlink_to(case / "result.json")
    else:
        path = workspace / "human-report.txt"
        path.write_text("one")
        (workspace / "synthetic-report.txt").write_text("two")
    original = path.read_bytes()
    backend = Backend()
    grade = judge_case(case, backend=backend)
    assert grade["status"] == "failed" and backend.calls == 0
    assert path.read_bytes() == original


def test_nonidentifying_binary_is_copied_exactly(tmp_path):
    case = tmp_path / "case"
    scenario, execution = make_case(case)
    (case / "workspace").mkdir()
    original = b"\xff\x00\x01measurement for this case\xfe"
    (case / "workspace/measurement.bin").write_bytes(original)
    prepare_view(case, case / "judging/evidence", scenario, execution)
    assert (case / "judging/evidence/workspace/measurement.bin").read_bytes() == original
