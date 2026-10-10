"""Harbor compatibility and evidence boundaries for the pipeline adapter."""

import json
from pathlib import Path

import pytest

pytest.importorskip('harbor')
from harbor.models.trajectories.trajectory import Trajectory

from benchmark.harbor.trajectory import from_turns, generation_trajectory, write, manifest
from benchmark.harbor.tasks import create_task


def test_atif_pairs_calls_and_results_without_fabricating_inference_boundaries():
    value = from_turns([{'text': 'Inspect.', 'tool_calls': [
        {'id': 'x', 'name': 'iot.sensors', 'input': {'asset': 'chiller'}, 'output': []}]}],
        name='stirrup', model='test', prompt='Solve.', identity='one')
    result = Trajectory.model_validate(value)
    step = result.steps[1]
    assert step.tool_calls[0].tool_call_id == step.observation.results[0].source_call_id
    assert step.llm_call_count is None
    assert result.final_metrics is None


def test_generation_retains_multiple_attempts_and_failed_attempt(tmp_path):
    logs = tmp_path / 'logs'
    logs.mkdir()
    for number, status in [(1, 'failed'), (2, 'succeeded')]:
        write(logs / f'run-{number}.json', {'process_status': status, 'requested_model': 'test', 'prompt': f'Attempt {number}'})
        (logs / f'codex-{number}.jsonl').write_text(json.dumps({'type': 'item.completed',
            'item': {'type': 'agent_message', 'text': status}}) + '\n')
    result = Trajectory.model_validate(generation_trajectory(tmp_path))
    assert len(result.subagent_trajectories) == 2
    assert result.subagent_trajectories[0].extra['status'] == 'failed'
    assert result.subagent_trajectories[1].steps[0].message == 'Attempt 2'



def test_manifest_omits_runtime_credentials_and_tracks_mutation(tmp_path):
    case = tmp_path / 'evaluation/cases/one'
    write(case / 'result.json', {'answer': 'done'})
    write(case / 'compose.json', {'password': 'secret'})
    write(case / 'auth/auth.json', {'token': 'secret'})
    values = manifest(tmp_path)
    assert list(values) == ['evaluation/cases/one/result.json']
    write(case / 'result.json', {'answer': 'changed'})
    assert manifest(tmp_path) != values


def test_task_verifier_distinguishes_completion_from_benchmark_pass(tmp_path):
    task = create_task(tmp_path, 'judging-one', {'stage': 'judging', 'timeout': 10})
    verifier = (task / 'tests/test.sh').read_text()
    assert 'stage_completed' in verifier and 'benchmark_pass' in verifier
    assert 'python:3.12-slim' in (task / 'environment/Dockerfile').read_text()
    with pytest.raises(ValueError):
        create_task(tmp_path, 'judging-one', {'stage': 'judging', 'timeout': 10})


def test_full_matrix_uses_astra_then_each_model_then_fable_and_continues_after_failure(tmp_path, monkeypatch):
    from benchmark.harbor import cli
    from benchmark.generated import sandbox, report
    from scenarios.generation import runtime

    calls = []

    async def fake_trial(root, name, spec, stages):
        calls.append(spec)
        failed = spec['stage'] == 'execution' and len(calls) == 2
        stages.append({'name': name, 'status': 'failed' if failed else 'completed', 'trajectory': 'unused.json'})
        return not failed

    def snapshot(generation, destination):
        destination.mkdir()
        write(destination / 'scenarios.json', [{'id': 'iot-1', 'type': 'iot', 'text': 'Inspect'}])

    monkeypatch.setattr(cli, 'trial', fake_trial)
    monkeypatch.setattr(cli, 'workflow_index', lambda *args: None)
    monkeypatch.setattr(sandbox, 'snapshot', snapshot)
    monkeypatch.setattr(runtime, 'stop', lambda *args: None)
    monkeypatch.setattr(report, 'write_report', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='all requested cases'):
        cli.main(['run', str(tmp_path / 'run'), '--runners',
                  '{"stirrup":["model-a","model-b"],"codex":"model-c"}'])
    assert calls[0]['stage'] == 'generation'
    assert calls[0]['model'] == 'gpt-6-astra'
    assert calls[0]['arguments'][calls[0]['arguments'].index('--count') + 1] == '25'
    executions = [s for s in calls if s['stage'] == 'execution']
    assert [s['model'] for s in executions] == ['model-a', 'model-b', 'model-c']
    judges = [s for s in calls if s['stage'] == 'judging']
    assert len(judges) == 3 and all(s['model'] == 'claude-fable-5-1' for s in judges)


def test_prepared_human_cohort_uses_same_execution_and_judge_without_generation(tmp_path, monkeypatch):
    from benchmark.harbor import cli
    from benchmark.generated import sandbox, report

    original = {'id': 404, 'type': 'wo', 'text': 'Summarize the events.',
                'characteristic_form': '14 work orders, 0 alerts, 0 anomalies.'}
    calls = []

    def import_snapshot(source, destination):
        assert source == tmp_path / 'human-snapshot'
        write(destination / 'scenarios.json', [original])

    async def fake_trial(root, name, spec, stages):
        calls.append(spec)
        assert json.loads((root / spec['case'] / 'scenario.json').read_text()) == original
        stages.append({'name': name, 'status': 'completed', 'trajectory': 'unused.json'})
        return True

    monkeypatch.setattr(sandbox, 'import_snapshot', import_snapshot)
    monkeypatch.setattr(cli, 'trial', fake_trial)
    monkeypatch.setattr(cli, 'workflow_index', lambda *args: None)
    monkeypatch.setattr(report, 'write_report', lambda *args, **kwargs: None)
    root = tmp_path / 'run'
    cli.main(['evaluate', str(root), '--snapshot', str(tmp_path / 'human-snapshot'),
              '--runners', '{"stirrup":"litellm_proxy/openai/gpt-5.6-luna"}'])
    assert [call['stage'] for call in calls] == ['execution', 'judging']
    assert calls[0]['model'] == 'litellm_proxy/openai/gpt-5.6-luna'
    assert calls[1]['model'] == 'claude-fable-5-1'
    assert (root / 'controller/src/evaluation/scorers/llm_judge.py').is_file()
    assert not (root / 'generation').exists()


def test_invalid_prepared_snapshot_never_launches_a_trial(tmp_path, monkeypatch):
    from benchmark.harbor import cli
    from benchmark.generated import sandbox

    calls = []

    def invalid_snapshot(*args):
        raise ValueError('Validated snapshot files do not match their manifest')

    monkeypatch.setattr(sandbox, 'import_snapshot', invalid_snapshot)
    monkeypatch.setattr(cli, 'trial', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='manifest'):
        cli.main(['evaluate', str(tmp_path / 'run'), '--snapshot', str(tmp_path / 'invalid'),
                  '--runners', '{"stirrup":"model"}'])
    assert calls == []


def test_codex_web_search_keeps_order_without_inventing_unlogged_results():
    from benchmark.harbor.trajectory import parse_codex
    events = [
        {'type': 'item.completed', 'item': {'id': 'web', 'type': 'web_search',
            'action': {'type': 'search', 'query': 'chiller diagnostics'}}},
        {'type': 'item.completed', 'item': {'id': 'cmd', 'type': 'command_execution',
            'command': 'python check.py', 'aggregated_output': 'valid'}},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Done'}},
    ]
    result = parse_codex('\n'.join(json.dumps(event) for event in events))
    turns = result['trajectory']['turns']
    assert turns[0]['tool_calls'][0]['name'] == 'web_search'
    assert 'output' not in turns[0]['tool_calls'][0]
    assert turns[1]['tool_calls'][0]['output'] == 'valid'
    assert turns[2]['text'] == 'Done'


def test_interrupted_judge_keeps_partial_native_tool_trace(tmp_path):
    from benchmark.harbor.trajectory import case_trajectory
    write(tmp_path / 'scenario.json', {'id': 'one'})
    write(tmp_path / 'judge.json', {'status': 'failed', 'model': 'claude-fable-5-1'})
    path = tmp_path / 'judging/events.jsonl'
    path.parent.mkdir()
    path.write_text(json.dumps({'type': 'assistant', 'message': {'content': [
        {'type': 'tool_use', 'id': 'read1', 'name': 'Read', 'input': {'file_path': '/evidence/result.json'}}]}}) + '\n')
    trace = case_trajectory(tmp_path, 'judging')
    assert trace['agent']['model_name'] == 'claude-fable-5-1'
    assert trace['steps'][-1]['tool_calls'][0]['function_name'] == 'Read'
    assert trace['extra']['grade']['status'] == 'failed'


def test_judge_retry_adds_trial_and_retains_original_failure(tmp_path, monkeypatch):
    from benchmark.harbor import cli
    from benchmark.generated import report

    original = {'name': 'judging-case-one', 'status': 'failed',
                'trajectory': 'trials/judging-case-one/agent/trajectory.json'}
    write(tmp_path / 'trajectory.json', {'extra': {'stages': [original]}})
    write(tmp_path / 'evaluation/cases/case-one/result.json', {'status': 'completed'})
    captured = []

    async def fake_trial(root, name, spec, stages):
        captured.append((name, spec))
        stages.append({'name': name, 'status': 'completed', 'trajectory': 'new.json'})
        write(root / 'saved-stages.json', stages)
        return True

    monkeypatch.setattr(cli, 'trial', fake_trial)
    monkeypatch.setattr(cli, 'workflow_index', lambda *args: None)
    monkeypatch.setattr(report, 'write_report', lambda *args, **kwargs: None)
    cli.main(['judge', str(tmp_path), '--case', 'case-one'])
    stages = json.loads((tmp_path / 'saved-stages.json').read_text())
    assert stages[0]['status'] == 'failed'
    assert stages[0]['superseded_by'] == 'judging-case-one-retry1'
    assert captured[0][1]['model'] == 'claude-fable-5-1'
    assert captured[0][1]['stage'] == 'judging'


def test_interrupted_judge_cleans_only_recorded_container(tmp_path, monkeypatch):
    from benchmark.harbor.stage import cleanup
    calls = []
    task = create_task(tmp_path, 'judging-one', {'stage': 'judging', 'timeout': 10,
                       'case': 'evaluation/cases/one'})
    marker = tmp_path / 'evaluation/cases/one/judging/container.json'
    write(marker, {'name': 'assetops-judge-123456abcdef'})
    monkeypatch.setattr('subprocess.run', lambda command, **kwargs: calls.append(command))
    cleanup(task / 'stage.json')
    assert calls == [['docker', 'rm', '--force', 'assetops-judge-123456abcdef']]
    write(marker, {'name': 'unrelated-container'})
    cleanup(task / 'stage.json')
    assert len(calls) == 1
