"""Evidence failures stay distinct from judged scenario failures."""

import csv
import json

import pytest

from benchmark.generated.judge import CRITERIA, ClaudeJudge, judge_case, main
from benchmark.generated.report import load_case, write_report
from llm.base import LLMBackend


class Backend(LLMBackend):
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt, temperature=0):
        self.calls += 1
        self.prompt = prompt
        return self.response


@pytest.fixture
def case(tmp_path):
    (tmp_path / "scenario.json").write_text(json.dumps({
        "id": "negative-1", "text": "Forecast the missing sensor.",
        "characteristic_form": "Verify absence and explain the missing evidence."}))
    (tmp_path / "result.json").write_text(json.dumps({
        "scenario_id": "negative-1", "status": "completed", "positive": False,
        "answer": "The sensor is absent.", "trajectory": [{"tool": "iot.sensors", "output": []}]}))
    return tmp_path


def passing():
    return json.dumps({key: key != "hallucinations" for key in CRITERIA})


def test_negative_success_and_matching_evidence_resume(case):
    backend = Backend(passing())
    grade = judge_case(case, backend=backend)
    assert grade["status"] == "completed"
    assert grade["score"]["passed"]
    assert judge_case(case, backend=backend) == grade
    assert backend.calls == 1
    result = json.loads((case / "result.json").read_text())
    result["answer"] = "Changed evidence"
    (case / "result.json").write_text(json.dumps(result))
    judge_case(case, backend=backend)
    assert backend.calls == 2


@pytest.mark.parametrize("response", ["not JSON", "{}", '{"task_completion": "true"}'])
def test_invalid_judge_is_ungraded_not_a_scenario_failure(case, response):
    assert judge_case(case, backend=Backend(response))["status"] == "failed"


def test_execution_failure_does_not_invoke_judge(case):
    result = json.loads((case / "result.json").read_text())
    result["status"] = "timed_out"
    (case / "result.json").write_text(json.dumps(result))
    backend = Backend(passing())
    assert judge_case(case, backend=backend)["status"] == "skipped"
    assert backend.calls == 0


def test_judge_mounts_only_full_evidence_readonly_with_isolated_auth(case, monkeypatch):
    (case / "workspace").mkdir()
    (case / "workspace/report.json").write_text('{"full": true}')
    auth_paths = []

    def auth(path, harness):
        assert harness == "claude"
        auth_paths.append(path)

    def launch(command, **kwargs):
        assert command[:2] == ["docker", "run"]
        assert command[command.index("--tools") + 1] == "Read,Glob,Grep"
        for flag in ("--safe-mode", "--restricted", "--strict-mcp-config", "--read-only"):
            assert flag in command
        assert command[command.index("--setting-sources") + 1] == ""
        assert command[command.index("--permission-mode") + 1] == "dontAsk"
        assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": {}}
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "--mount"]
        evidence = [mount for mount in mounts if "dst=/evidence/" in mount]
        assert len(evidence) == 3
        assert all(mount.endswith(",readonly") for mount in evidence)
        assert len(mounts) == 4
        assert not any("compose" in mount or "judging" in mount for mount in mounts)
        assert "ANTHROPIC_API_KEY" not in " ".join(command)
        assert kwargs["start_new_session"] is True

        class Process:
            returncode = 0

            def communicate(self, prompt, timeout):
                assert timeout == 12
                kwargs["stdout"].write(json.dumps({"type": "assistant", "message": {"content": [
                    {"type": "tool_use", "id": "read", "name": "Read",
                     "input": {"file_path": "/evidence/result.json"}}]}}) + "\n")
                kwargs["stdout"].write(json.dumps({"type": "result", "result": passing()}) + "\n")

        return Process()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-forward")
    monkeypatch.setattr("benchmark.generated.judge.prepare_auth", auth)
    monkeypatch.setattr("benchmark.generated.judge.subprocess.Popen", launch)
    assert ClaudeJudge(case, timeout=12).generate("evaluate") == passing()
    assert auth_paths and not auth_paths[0].exists()
    trace = json.loads((case / "judging/result.json").read_text())
    assert trace["trajectory"]["turns"][0]["tool_calls"][0]["name"] == "Read"
    assert (case / "workspace/report.json").read_text() == '{"full": true}'


def test_report_retains_missing_grades_and_failed_executions(tmp_path):
    grade = {"status": "completed", "score": {
        "passed": True, "details": json.loads(passing()), "rationale": "Supported abstention"}}
    base = {"runner": "claude", "model": "test", "domain": "iot"}
    rows = [
        {**base, "scenario_id": "1", "positive": False, "status": "completed", "grading": grade},
        {**base, "scenario_id": "2", "positive": True, "status": "completed"},
        {**base, "scenario_id": "3", "positive": True, "status": "failed", "error": "Timeout"},
    ]
    path = write_report(tmp_path, rows, chart=False)
    text = path.read_text()
    assert "2/3 | 1/3 | 1/3 (33%) | 0/2 (0%) | 1/1 (100%)" in text
    with (tmp_path / "cases.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 3
    assert saved[1]["strict_pass"] == ""
    assert saved[2]["error"] == "Timeout"


def test_strict_chart_counts_all_planned_cases_and_rejects_hallucinations(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    heights = []
    original = Axes.bar

    def capture(self, x, height, *args, **kwargs):
        heights.append(list(height))
        return original(self, x, height, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture)
    grade = {"status": "completed", "score": {"details": json.loads(passing())}}
    base = {"runner": "codex", "model": "test", "domain": "iot"}
    hallucinations = json.loads(passing())
    hallucinations["hallucinations"] = True
    rows = [
        {**base, "scenario_id": "1", "positive": True, "status": "completed", "grading": grade},
        {**base, "scenario_id": "2", "positive": True, "status": "completed"},
        {**base, "scenario_id": "3", "positive": False, "status": "error", "error": "Timeout"},
        {**base, "model": "hallucinated", "scenario_id": "1", "positive": True,
         "status": "completed", "grading": {"status": "completed", "score": {"details": hallucinations}}},
    ]
    text = write_report(tmp_path, rows).read_text()
    assert heights[0] == pytest.approx([100 / 3, 50, 0])
    assert heights[1] == [0, 0]  # No negative scenarios for this model.
    assert heights[2] == [100, 100, 100, 100, 100, 0]  # Criteria use judged cases only.
    for filename in ("strict_pass.png", "criteria.png"):
        assert filename in text
        assert (tmp_path / filename).is_file()


def test_report_ignores_native_and_archived_attempts(tmp_path):
    row = {"scenario_id": "1", "positive": True, "runner": "codex", "model": "test", "status": "pending"}
    for relative in ["cases/one/result.json", "cases/one/native/result.json", "attempts/one/result.json"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))
    report = write_report(tmp_path, chart=False).read_text()
    assert "1 scenarios · 1 executions planned" in report


def test_report_separates_execution_timeouts_from_tool_errors_and_setup_time(tmp_path):
    base = {"runner": "zcode", "model": "test", "domain": "iot", "positive": True}
    rows = [
        {**base, "scenario_id": "1", "status": "completed", "elapsed_seconds": 30, "duration_seconds": 90},
        {**base, "scenario_id": "2", "status": "completed", "elapsed_seconds": 90, "duration_seconds": 150},
        {**base, "scenario_id": "3", "status": "error", "elapsed_seconds": 900,
         "error": "Execution timed out after 900 seconds"},
        {**base, "scenario_id": "4", "status": "error", "elapsed_seconds": 5,
         "error": "A tool call timed out"},
    ]
    text = write_report(tmp_path, rows, chart=False).read_text()
    assert "| Runner / model | Completed | Timed out |" not in text
    with (tmp_path / "cases.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    assert saved[2]["execution_outcome"] == "timed_out"
    assert saved[3]["execution_outcome"] == "execution_error"
    assert saved[0]["elapsed_seconds"] == "30"
    assert saved[0]["duration_seconds"] == "90"


def test_report_links_the_evaluated_scenarios_and_removes_boilerplate(tmp_path):
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "scenarios.md").write_text("Operator requests")
    (tmp_path / "snapshot.json").write_text(json.dumps({"generation": str(generation)}))
    (tmp_path / "scenarios.json").write_text("[]")
    text = write_report(tmp_path, [], chart=False).read_text()
    assert "[Evaluated scenarios and rubrics](scenarios.json)" in text
    assert "[Read the scenario requests](generation/scenarios.md)" in text
    assert "Lower hallucination rates are better" not in text
    assert "without uncertainty estimates" not in text


def test_stale_judge_is_pending_after_execution_changes(case):
    judge_case(case, backend=Backend(passing()))
    assert load_case(case)["grading"]["status"] == "completed"
    row = json.loads((case / "result.json").read_text())
    row["answer"] = "New answer requiring a fresh grade"
    (case / "result.json").write_text(json.dumps(row))
    assert load_case(case)["grading"]["status"] == "pending"


def test_duplicate_execution_rejected(tmp_path):
    row = {"scenario_id": "1", "runner": "codex", "model": "test", "status": "pending"}
    with pytest.raises(ValueError, match="Duplicate execution"):
        write_report(tmp_path, [row, row], chart=False)


def test_report_checks_criteria_not_untrusted_pass_flag(tmp_path):
    details = json.loads(passing())
    details["hallucinations"] = True
    grade = {"status": "completed", "score": {"passed": True, "details": details}}
    row = {"scenario_id": "1", "positive": True, "runner": "codex", "model": "test",
           "status": "completed", "grading": grade}
    text = write_report(tmp_path, [row], chart=False).read_text()
    assert "1/1 | 1/1 | 0/1 (0%)" in text


def test_full_two_mode_ledger_retains_all_forty_planned_executions(tmp_path):
    matrix = {
        "general-execution": {"codex": "gpt-6-astra", "claude-code": "claude-opus-5-5"},
        "mcp-only": {"openai-agent": "zai/glm-5.3", "claude-agent": "claude-opus-5-5"},
    }
    rows = [{"generation_mode": mode, "runner": runner, "model": model,
             "scenario_id": str(index), "positive": index < 8, "status": "pending"}
            for mode, runners in matrix.items() for runner, model in runners.items()
            for index in range(10)]
    report = write_report(tmp_path, rows, chart=False).read_text()
    assert "20 scenarios · 40 executions planned" in report
    assert report.count("0/10 | 0/10 | 0/10 (0%) | 0/8 (0%) | 0/2 (0%)") == 4
    with (tmp_path / "cases.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 40



def test_judge_points_to_full_evidence_without_compaction(case):
    execution = json.loads((case / "result.json").read_text())
    execution["trajectory"] = {"turns": [{"tool_calls": [
        {"name": "iot.history", "output": {"rows": ["saved observation"] * 10000}},
        {"name": "wo.list_workorders", "output": {"total": 0}},
    ]}]}
    original = json.dumps(execution, indent=2)
    (case / "result.json").write_text(original)
    backend = Backend(passing())
    grade = judge_case(case, backend=backend)
    assert grade["status"] == "completed"
    assert "/evidence/result.json" in backend.prompt
    assert "/evidence/workspace" in backend.prompt
    assert "saved observation" not in backend.prompt
    assert "trajectory_character_limit" not in grade
    assert (case / "result.json").read_text() == original


def test_standalone_judge_discovers_only_cases_and_refreshes_report(tmp_path, monkeypatch):
    execution = tmp_path / "cases/one/result.json"
    execution.parent.mkdir(parents=True)
    execution.write_text("{}")
    (tmp_path / "result.json").write_text("{}")
    judged, reports = [], []

    def judge(path, **kwargs):
        judged.append((path, kwargs))
        return {"status": "completed"}

    monkeypatch.setattr("benchmark.generated.judge.judge_case", judge)
    (tmp_path / "snapshot.json").write_text(json.dumps({"request": {
        "asset_class": "Transformer", "generation_mode": "mcp-only"}}))
    monkeypatch.setattr("benchmark.generated.report.write_report", lambda root, **kw: reports.append((root, kw)))
    assert main([str(tmp_path), "--jobs", "2", "--timeout", "42"]) == 0
    assert judged == [(execution.parent, {"timeout": 42})]
    assert reports == [(tmp_path, {"title": "Transformer · mcp-only"})]
