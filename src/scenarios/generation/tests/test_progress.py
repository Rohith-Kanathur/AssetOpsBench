import json

import pytest

from scenarios.generation import progress


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def item(kind, item_type, **fields):
    return {"type": kind, "item": {"id": "item_1", "type": item_type, **fields}}


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def test_empty_run_reports_unrecorded_views_without_claiming_completion(tmp_path):
    text = progress.inspect_run(tmp_path)
    assert "Generation: unknown" in text
    assert "Profile: not recorded" in text
    assert "Checks: not recorded" in text


def test_native_process_success_does_not_imply_validation_success(tmp_path):
    save(tmp_path / "logs/run-2.json", {"process_status": "succeeded"})
    (tmp_path / "logs/codex-2.jsonl").write_text('{"type":"turn.completed"}\n')
    text = progress.inspect_run(tmp_path)
    assert "Generation: incomplete · attempt 2" in text
    assert "validation not recorded" in text
    assert "validation separate" in text


@pytest.mark.parametrize("validation, expected", [("pending", "checking"), ("passed", "complete"), ("failed", "incomplete")])
def test_metadata_fallback_distinguishes_validation_states(tmp_path, validation, expected):
    save(tmp_path / "logs/run-1.json", {"process_status": "succeeded", "validation_status": validation})
    assert f"Generation: {expected}" in progress.inspect_run(tmp_path)


def test_authoritative_status_counts_and_numeric_latest_attempt(tmp_path):
    save(tmp_path / "logs/run-2.json", {"requested_model": "older"})
    save(tmp_path / "logs/run-10.json", {"requested_model": "newer", "process_status": "running"})
    save(tmp_path / "status.json", {"status": "checking", "attempt": 10, "counts": {"positive": 6, "negative": 6}})
    text = progress.inspect_run(tmp_path)
    assert "checking · attempt 10" in text
    assert "Native: codex / newer" in text and "Native: codex / older" not in text
    assert '"positive": 6' in text


def test_partial_metadata_and_native_lines_are_tolerated(tmp_path):
    save(tmp_path / "logs/run-1.json", {"process_status": "running"})
    (tmp_path / "logs/run-2.json").write_text('{"process_status":')
    (tmp_path / "status.json").write_text('{"status":')
    (tmp_path / "logs/codex-1.jsonl").write_text(
        json.dumps(item("item.completed", "command_execution", command="python scripts/check.py", exit_code=0))
        + '\nnot-json\n{"type":"item.started"')
    text = progress.inspect_run(tmp_path)
    assert "running · attempt 1" in text
    assert "command_execution=1" in text
    assert "python scripts/check.py" in text


def test_messages_commands_and_tool_names_are_visible_without_payload_secrets(tmp_path):
    save(tmp_path / "status.json", {"status": "running"})
    log = tmp_path / "logs/codex-1.jsonl"
    log.parent.mkdir()
    events = [item("item.completed", "agent_message", text='Checking the recorded sources. "api_key": "hidden-message"'),
              item("item.started", "command_execution", command="env API_KEY=hidden-command python scripts/check.py",
                   aggregated_output="hidden-output"),
              item("item.completed", "mcp_tool_call", server="iot", tool="asset_detail",
                   arguments={"password": "hidden-arguments"}, result="hidden-result")]
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    text = progress.inspect_run(tmp_path)
    assert "Checking the recorded sources." in text
    assert "python scripts/check.py" in text
    assert "iot.asset_detail" in text
    assert "hidden-" not in text


def test_index_preserves_native_bytes_and_links_real_stage_artifacts(tmp_path):
    save(tmp_path / "workspace/output/profile.json", {})
    save(tmp_path / "logs/run-1.json", {"process_status": "running"})
    native = tmp_path / "logs/codex-1.jsonl"
    native.write_bytes(b'{"type":"turn.started"}\n{"partially_written":')
    before = native.read_bytes()
    path = progress.write_index(tmp_path)
    text = path.read_text()
    assert native.read_bytes() == before
    assert "Profile: recorded" in text
    assert "[profile.json](<../workspace/output/profile.json>)" in text
    assert "not validated" in text
    assert "Checks: not recorded" in text
    assert not (tmp_path / "logs/.index.md.tmp").exists()


@pytest.mark.parametrize("command", [
    'curl -H "X-API-Key: hidden" https://example.test',
    'curl -H "Authorization: Basic hidden" https://example.test',
    'cli --password "hidden" --mode check',
    'curl https://user:hidden@example.test',
])
def test_common_command_credentials_are_redacted(command):
    text = progress._event(item("item.started", "command_execution", command=command), "codex-1")
    assert "hidden" not in text
    assert "redacted" in text


def test_research_receipts_display_query_and_link_saved_response(tmp_path):
    save(tmp_path / "workspace/data/research/search-1.json", {"result": "not printed"})
    receipts = tmp_path / "logs/research.jsonl"
    receipts.parent.mkdir()
    receipts.write_text(json.dumps({"query": "transformer thermal aging", "status": "succeeded",
                                   "time": "2026-10-02", "response_file": "workspace/data/research/search-1.json"}) + "\n")
    text = progress.inspect_run(tmp_path)
    assert "Research: recorded" in text
    assert "research succeeded: transformer thermal aging" in text
    assert "search-1.json" in text
    assert "not printed" not in text


def test_watch_follows_partial_line_new_attempt_and_status_until_complete(tmp_path, monkeypatch, capsys):
    save(tmp_path / "status.json", {"status": "running", "attempt": 1})
    log = tmp_path / "logs/codex-1.jsonl"
    log.parent.mkdir()
    event = json.dumps(item("item.completed", "agent_message", text="Finished checking."))
    log.write_text(event[:20])
    calls = []

    def advance(_):
        calls.append(1)
        if len(calls) == 1:
            with log.open("a") as stream:
                stream.write(event[20:] + "\n")
            other = tmp_path / "logs/codex-2.jsonl"
            other.write_text(json.dumps(item("item.started", "mcp_tool_call", server="wo", tool="get_workorder")) + "\n")
            save(tmp_path / "status.json", {"status": "checking", "attempt": 2})
        else:
            save(tmp_path / "status.json", {"status": "complete", "attempt": 2})

    monkeypatch.setattr(progress.time, "sleep", advance)
    progress.watch_run(tmp_path)
    text = capsys.readouterr().out
    assert text.count("Finished checking.") == 1
    assert "wo.get_workorder" in text
    assert "Generation status: checking" in text
    assert "Generation status: complete" in text


def test_watch_handles_log_truncation(tmp_path, monkeypatch, capsys):
    save(tmp_path / "status.json", {"status": "running"})
    log = tmp_path / "logs/codex-1.jsonl"
    log.parent.mkdir()
    append(log, item("item.completed", "agent_message", text="A very long previous native event " * 20))

    def advance(_):
        log.write_text('{"type":"turn.failed","message":"new attempt"}\n')
        save(tmp_path / "status.json", {"status": "failed"})

    monkeypatch.setattr(progress.time, "sleep", advance)
    progress.watch_run(tmp_path)
    assert "turn.failed: new attempt" in capsys.readouterr().out


def test_watch_keeps_following_during_repair(tmp_path, monkeypatch, capsys):
    save(tmp_path / "status.json", {"status": "repairing", "attempt": 1})
    sleeps = []

    def advance(_):
        sleeps.append(1)
        save(tmp_path / "status.json", {"status": "complete", "attempt": 2})

    monkeypatch.setattr(progress.time, "sleep", advance)
    progress.watch_run(tmp_path)
    text = capsys.readouterr().out
    assert sleeps == [1]
    assert "Generation status: repairing" in text
    assert "Generation status: complete" in text


def test_ctrl_c_stops_only_viewer(tmp_path, monkeypatch, capsys):
    save(tmp_path / "status.json", {"status": "running"})
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(progress.time, "sleep", interrupt)
    progress.watch_run(tmp_path)
    assert "generation continues independently" in capsys.readouterr().out
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "running"


def test_manual_receipts_are_not_reported_as_runtime_checks(tmp_path):
    save(tmp_path / "workspace/output/tool_checks.json", [{"scenario_id": 1, "calls": []}])
    text = progress.inspect_run(tmp_path)
    assert "Checks: not recorded" in text
    assert "tool_checks" not in text
    save(tmp_path / "logs/review-1.json", {"errors": []})
    assert "Checks: recorded" in progress.inspect_run(tmp_path)
