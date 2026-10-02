"""Parallel groups share immutable snapshots and never reset partial executions."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time

import pytest

from benchmark.generated_suite_runner import completed_scenarios
from benchmark.measurement import suite_hash, write_json


@pytest.fixture
def orchestrator():
    script = Path(__file__).resolve().parents[3] / 'tools/run_asset_cohort_comparison.py'
    spec = importlib.util.spec_from_file_location('cohort_orchestrator', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def experiment(tmp_path):
    targets = []
    for cohort in ('existing', 'synthetic'):
        suite = tmp_path / 'suites' / cohort
        write_json(suite / 'run.json', {
            'status': 'complete', 'negative_count': 0,
            'config': {'num_scenarios': 2, 'num_negative_scenarios': 0},
        })
        write_json(suite / 'scenarios.json', [
            {'id': str(i), 'text': f'Inspect real Chiller {i}', 'type': 'iot',
             'deterministic': True, 'characteristic_form': f'Asset {i} found'}
            for i in (1, 2)
        ])
        for model in ('a', 'b'):
            targets.append({'name': f'{cohort}-{model}', 'cohort': cohort,
                            'agent': 'claude', 'model_id': f'model-{model}', 'suite': str(suite)})
    config = {'targets': targets, 'judge': 'claude-code/claude-fable-5-1'}
    config_path = tmp_path / 'config.json'
    write_json(config_path, config)
    snapshot_path = tmp_path / 'snapshot.json.gz'
    snapshot = {'iot': [{'_id': 'chiller-6', 'asset': 'Chiller 6'}]}
    snapshot_path.write_bytes(gzip.compress(json.dumps(snapshot).encode(), mtime=0))
    return {
        'config': config, 'config_path': config_path, 'output': tmp_path / 'results',
        'snapshot': snapshot_path,
        'digest': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        'snapshot_value': snapshot,
    }


def arguments(experiment, *, k=3):
    return ['--config', str(experiment['config_path']), '--output-dir', str(experiment['output']),
            '--snapshot-file', str(experiment['snapshot']), '--k', str(k)]


def value(command, flag):
    return command[command.index(flag) + 1]


def save_completed(root, target, digest, judge, *, scenario_limit=None):
    environment = {'snapshot_sha256': digest, 'namespace': f'{root.name}_{target["name"]}_'}
    write_json(root / target['name'] / 'environment.json', environment)
    rows, files = completed_scenarios(Path(target['suite']))
    for index, scenario in enumerate(rows[:scenario_limit], 1):
        write_json(root / target['name'] / 'measurements' / f'case-{index}.json', {
            'scenario_id': scenario.id, 'attempt': 1, 'status': 'completed',
            'execution_index': index, 'execution_duration_ms': 10, 'metrics': {},
            'settings': {'model': target['model_id'], 'agent': target['agent'],
                         'reasoning_effort': target.get('reasoning_effort'),
                         'suite_sha256': suite_hash(files), 'judge_model': judge,
                         'concurrency_level': 2, 'database_policy': environment},
            'grading': {'status': 'completed', 'duration_ms': 1,
                        'result': {'score': {'passed': True, 'score': 1, 'details': {}}}},
        })


def fake_launcher(experiment, calls, *, barrier=None, counts=None):
    lock = threading.Lock()

    def run(command, **kwargs):
        root = Path(value(command, '--output-dir'))
        config = json.loads(Path(value(command, '--config')).read_text())
        with lock:
            calls.append(command)
            if counts is not None:
                counts['active'] += 1
                counts['maximum'] = max(counts['maximum'], counts['active'])
        if barrier:
            barrier.wait(timeout=5)
        time.sleep(.02)
        snapshot = Path(value(command, '--snapshot-file'))
        if not snapshot.exists():
            snapshot.write_bytes(gzip.compress(json.dumps(experiment['snapshot_value']).encode(), mtime=0))
        for target in config['targets']:
            save_completed(root, target, experiment['digest'], config['judge'])
        with lock:
            if counts is not None:
                counts['active'] -= 1
        return subprocess.CompletedProcess(command, 0)

    return run


def test_k3_runs_distinct_groups_in_bounded_parallel_on_one_snapshot(orchestrator, experiment, monkeypatch):
    calls, counts = [], {'active': 0, 'maximum': 0}
    monkeypatch.setattr(orchestrator.subprocess, 'run', fake_launcher(
        experiment, calls, barrier=threading.Barrier(2), counts=counts))
    orchestrator.main(arguments(experiment))
    assert len(calls) == 6 and counts['maximum'] == 2
    assert {(value(c, '--repetition-index'), Path(value(c, '--config')).stem) for c in calls} == {
        (str(i), f'{cohort}-config') for i in (1, 2, 3) for cohort in ('existing', 'synthetic')}
    assert all(value(c, '--expected-snapshot-sha256') == experiment['digest'] for c in calls)
    saved = json.loads((experiment['output'] / 'experiment.json').read_text())
    assert saved['status'] == 'completed' and saved['snapshot_sha256'] == experiment['digest']
    assert saved['max_concurrent_groups'] == 2 and saved['max_execution_targets'] == 4
    assert saved['execution_policy']['concurrency_level_scope'] == 'native cohort launcher group width'
    starts = [entry for entry in saved['scheduling_history'] if entry['event'] == 'started']
    assert len(starts) == 6 and max(e['active_group_target_capacity'] for e in starts) == 4
    namespaces = set()
    for rep in saved['repetitions']:
        assert rep['cohorts'] == {'existing': 'completed', 'synthetic': 'completed'}
        for target in experiment['config']['targets']:
            environment = json.loads((Path(rep['root']) / target['name'] / 'environment.json').read_text())
            namespaces.add(environment['namespace'])
    assert len(namespaces) == 12


def test_resume_preserves_completed_outputs_without_model_or_database_invocations(orchestrator, experiment, monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator.subprocess, 'run', fake_launcher(experiment, calls))
    args = arguments(experiment, k=1)
    orchestrator.main(args)
    before = {p: p.read_bytes() for p in experiment['output'].rglob('case-*.json')}
    calls.clear()
    orchestrator.main(args + ['--resume'])
    assert calls == []
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize('state', ['partial_measurements', 'namespace_only', 'complete_target'])
def test_partial_group_including_later_target_refuses_database_reset_before_dispatch(
        orchestrator, experiment, monkeypatch, state):
    root = experiment['output'] / 'repetition-1'
    target = experiment['config']['targets'][1]  # First target is empty; still inspect this one.
    if state == 'namespace_only':
        write_json(root / target['name'] / 'environment.json', {'snapshot_sha256': experiment['digest']})
    else:
        save_completed(root, target, experiment['digest'], experiment['config']['judge'],
                       scenario_limit=1 if state == 'partial_measurements' else None)
    before = {p: p.read_bytes() for p in root.rglob('*.json')}
    calls = []
    monkeypatch.setattr(orchestrator.subprocess, 'run', fake_launcher(experiment, calls))
    with pytest.raises(ValueError, match='refusing to reset'):
        orchestrator.main(arguments(experiment))
    assert calls == []
    assert all(p.read_bytes() == contents for p, contents in before.items())


def test_preflight_validates_measurement_identity_even_after_an_empty_target(orchestrator, experiment):
    root = experiment['output'] / 'repetition-1'
    targets = experiment['config']['targets'][:2]
    save_completed(root, targets[1], experiment['digest'], 'wrong-judge')
    with pytest.raises(ValueError, match='suite/model/agent/judge/snapshot'):
        orchestrator.completed_group(root, targets, experiment['digest'],
                                     judge_model=experiment['config']['judge'])


def test_environment_must_match_measurements_even_before_reference_snapshot_is_known(orchestrator, experiment):
    root = experiment['output'] / 'repetition-1'
    targets = experiment['config']['targets'][:2]
    for target in targets:
        save_completed(root, target, experiment['digest'], experiment['config']['judge'])
        write_json(root / target['name'] / 'environment.json', {'snapshot_sha256': 'other-snapshot'})
    with pytest.raises(ValueError, match='suite/model/agent/judge/snapshot'):
        orchestrator.completed_group(root, targets, judge_model=experiment['config']['judge'])


def test_snapshot_changes_block_resume_before_launch(orchestrator, experiment, monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator.subprocess, 'run', fake_launcher(experiment, calls))
    args = arguments(experiment, k=1)
    orchestrator.main(args)
    calls.clear()
    experiment['snapshot'].write_bytes(gzip.compress(json.dumps({'iot': []}).encode()))
    with pytest.raises(ValueError, match='snapshot differs'):
        orchestrator.main(args + ['--resume'])
    assert calls == []


def test_first_snapshot_capture_finishes_alone_before_parallel_readers(orchestrator, experiment, monkeypatch):
    experiment['snapshot'].unlink()
    calls, counts = [], {'active': 0, 'maximum': 0}
    monkeypatch.setattr(orchestrator.subprocess, 'run', fake_launcher(experiment, calls, counts=counts))
    orchestrator.main(arguments(experiment))
    assert len(calls) == 6 and counts['maximum'] == 2
    assert '--expected-snapshot-sha256' not in calls[0]
    assert all(value(c, '--expected-snapshot-sha256') == experiment['digest'] for c in calls[1:])
    saved = json.loads((experiment['output'] / 'experiment.json').read_text())
    starts = [e for e in saved['scheduling_history'] if e['event'] == 'started']
    assert starts[0]['active_cohort_groups'] == 1


def test_group_failure_stops_new_dispatch_and_preserves_successful_sibling(orchestrator, experiment, monkeypatch):
    calls, barrier = [], threading.Barrier(2)
    successful_run = fake_launcher(experiment, calls)

    def run(command, **kwargs):
        barrier.wait(timeout=5)
        if Path(value(command, '--config')).stem == 'existing-config':
            calls.append(command)
            raise subprocess.CalledProcessError(1, command)
        return successful_run(command, **kwargs)

    monkeypatch.setattr(orchestrator.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        orchestrator.main(arguments(experiment))
    assert len(calls) == 2
    saved = json.loads((experiment['output'] / 'experiment.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['repetitions'][0]['cohorts'] == {'existing': 'failed', 'synthetic': 'completed'}
    assert saved['repetitions'][1]['cohorts'] == {'existing': 'pending', 'synthetic': 'pending'}


@pytest.mark.parametrize('flag', ['--k', '--max-concurrent-groups'])
def test_nonpositive_execution_bounds_are_rejected(orchestrator, experiment, flag):
    with pytest.raises(SystemExit):
        orchestrator.main(arguments(experiment) + [flag, '0'])


@pytest.mark.parametrize('field,other', [
    ('model_id', 'different-model'), ('agent', 'codex'), ('reasoning_effort', 'high'),
])
def test_configured_cohort_executor_mismatch_rejected_before_dispatch(
        orchestrator, experiment, monkeypatch, capsys, field, other):
    config = experiment['config']
    for spec in config['targets']:
        spec['model_key'] = spec['name'].split('-', 1)[1]
    config['targets'][2][field] = other
    write_json(experiment['config_path'], config)
    calls = []
    monkeypatch.setattr(orchestrator.subprocess, 'run', fake_launcher(experiment, calls))
    with pytest.raises(SystemExit):
        orchestrator.main(arguments(experiment))
    assert 'cohorts differ in executor models, agents, reasoning_effort or model keys' in capsys.readouterr().err
    assert calls == [] and not experiment['output'].exists()


def test_swapped_executor_assignments_cannot_hide_behind_the_same_model_multiset(orchestrator, experiment):
    targets = experiment['config']['targets']
    for spec in targets:
        spec['model_key'] = spec['name'].split('-', 1)[1]
    targets[2]['model_id'], targets[3]['model_id'] = targets[3]['model_id'], targets[2]['model_id']
    with pytest.raises(ValueError, match='cohorts differ'):
        orchestrator.validate_cohort_targets({'existing': targets[:2], 'synthetic': targets[2:]})


@pytest.mark.parametrize('saved_effort', [None, 'high'])
def test_saved_reasoning_must_match_explicit_configuration(orchestrator, experiment, saved_effort):
    root = experiment['output'] / 'repetition-1'
    target = {**experiment['config']['targets'][0], 'reasoning_effort': 'low'}
    save_completed(root, target, experiment['digest'], experiment['config']['judge'])
    for path in (root / target['name'] / 'measurements').glob('*.json'):
        record = json.loads(path.read_text())
        record['settings']['reasoning_effort'] = saved_effort
        write_json(path, record)
    with pytest.raises(ValueError, match='configured reasoning_effort'):
        orchestrator.completed_group(root, [target], experiment['digest'],
                                     judge_model=experiment['config']['judge'])


def test_matching_explicit_effort_and_unspecified_native_effort_are_preserved(
        orchestrator, experiment, monkeypatch):
    for spec in experiment['config']['targets']:
        if spec['name'].endswith('-a'):
            spec['reasoning_effort'] = 'low'
        elif spec['cohort'] == 'synthetic':
            spec['reasoning_effort'] = None  # Equivalent to the omitted native setting.
    write_json(experiment['config_path'], experiment['config'])
    calls = []
    monkeypatch.setattr(orchestrator.subprocess, 'run', fake_launcher(experiment, calls))
    orchestrator.main(arguments(experiment, k=1))
    assert len(calls) == 2
    assert json.loads((experiment['output'] / 'experiment.json').read_text())['status'] == 'completed'
