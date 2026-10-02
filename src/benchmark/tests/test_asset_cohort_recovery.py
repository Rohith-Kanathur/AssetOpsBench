"""Manual quota recovery retains saved namespaces, identities, attempts and grades."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmark.generated_suite_runner import completed_scenarios
from benchmark.measurement import suite_hash, write_json


@pytest.fixture
def recovery(monkeypatch):
    script = Path(__file__).resolve().parents[3] / 'tools/resume_asset_cohort_comparison.py'
    spec = importlib.util.spec_from_file_location('cohort_recovery', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'versions', lambda _: {'sdk_version': 'fixture-sdk', 'cli_version': 'fixture-cli'})
    return module


@pytest.fixture
def saved(tmp_path, recovery):
    snapshot = {'iot': [{'_id': 'chiller6'}], 'failure_mode': []}
    snapshot_path = tmp_path / 'snapshot.json.gz'
    snapshot_path.write_bytes(gzip.compress(json.dumps(snapshot).encode()))
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    output = tmp_path / 'results'
    rep = {'index': 1, 'root': str(output / 'repetition-1'),
           'cohorts': {'existing': 'completed', 'synthetic': 'failed'}}
    specs = []
    hashes = {}
    for cohort in ['existing', 'synthetic']:
        suite = tmp_path / 'suites' / cohort
        write_json(suite / 'run.json', {'status': 'complete', 'negative_count': 0,
                                      'config': {'num_scenarios': 3, 'num_negative_scenarios': 0}})
        write_json(suite / 'scenarios.json', [{'id': f's{i}', 'text': f'Inspect Chiller 6 item {i}'} for i in [1, 2, 3]])
        spec = {'name': f'{cohort}-claude', 'model_key': 'claude', 'agent': 'claude',
                'model_id': 'claude-opus-5-5', 'cohort': cohort, 'suite': str(suite)}
        specs.append(spec)
        _, files = completed_scenarios(suite)
        hashes[spec['name']] = suite_hash(files)
        target = Path(rep['root']) / spec['name']
        policy = {'snapshot_sha256': digest, 'namespace': f'eval_{cohort}_claude_saved_',
                  'suite': str(suite), 'cohort': cohort, 'database_count': 2, 'document_count': 1}
        settings = {'generation_run': str(suite), 'agent': spec['agent'], 'model': spec['model_id'],
                    'suite_sha256': hashes[spec['name']], 'repetition_index': 1,
                    'judge_model': 'claude-code/claude-fable-5-1', 'database_policy': policy,
                    'rubric_sha256': hashlib.sha256((recovery.ROOT / 'src/evaluation/scorers/llm_judge.py').read_bytes()).hexdigest(),
                    'sdk_version': 'fixture-sdk', 'cli_version': 'fixture-cli', 'reasoning_effort': None,
                    'timeout_seconds': 900, 'concurrency_level': 5, 'allow_same_model_judge': True,
                    'invocation_retry_policy': {'max_attempts': 3, 'scope': 'whole scenario invocation'}}
        write_json(target / 'environment.json', policy)
        write_json(target / 'settings.json', settings)
        write_json(target / 'target.json', {'generation_run': str(suite), 'agent': spec['agent'], 'model': spec['model_id']})
        for i in [1, 2, 3] if cohort == 'existing' else [1]:
            complete_record(target, spec, settings, i, graded=True)
        if cohort == 'synthetic':
            for attempt in [1, 2, 3]:
                quota_record(target, spec, settings, 2, attempt)
    experiment = {'k': 1, 'status': 'failed', 'config': {'targets': specs, 'judge': 'claude-code/claude-fable-5-1'},
                  'suite_hashes': hashes, 'snapshot_sha256': digest, 'snapshot_file': str(snapshot_path),
                  'repetitions': [rep], 'error': {'type': 'Quota', 'message': 'Claude session limit'}}
    experiment_path = output / 'experiment.json'
    write_json(experiment_path, experiment)
    return {'experiment_path': experiment_path, 'experiment': experiment, 'snapshot': snapshot,
            'target': Path(rep['root']) / 'synthetic-claude', 'spec': specs[1], 'digest': digest}


def complete_record(target, spec, settings, i, *, graded, attempt=1, recovery=None):
    rid = f'{spec["name"]}_{i:04d}'
    record = {'run_id': rid, 'scenario_id': f's{i}', 'execution_index': i, 'attempt': attempt,
              'status': 'completed', 'execution_duration_ms': 10, 'metrics': {}, 'settings': deepcopy(settings),
              'grading': {'status': 'completed', 'model': settings['judge_model'], 'duration_ms': 1,
                          'attempts': [{'status': 'completed'}],
                          'result': {'score': {'score': 1, 'passed': True, 'details': {}}}} if graded else None}
    if recovery:
        record['recovery'] = recovery
    path = target / 'measurements' / (f'{rid}.json' if attempt == 1 else f'{rid}.attempt-{attempt}.json')
    write_json(path, record)
    write_json(target / 'trajectories' / f'{rid}.json', {'run_id': rid, 'scenario_id': f's{i}',
               'runner': 'claude-agent', 'model': spec['model_id'], 'question': f'Inspect Chiller 6 item {i}',
               'answer': 'Evidence', 'trajectory': []})


def quota_record(target, spec, settings, i, attempt):
    rid = f'{spec["name"]}_{i:04d}'
    name = rid if attempt == 1 else f'{rid}.attempt-{attempt}'
    write_json(target / 'measurements' / f'{name}.json', {
        'run_id': rid, 'scenario_id': f's{i}', 'execution_index': i, 'attempt': attempt,
        'status': 'failed', 'execution_duration_ms': 5, 'metrics': {}, 'settings': deepcopy(settings),
        'agent_error': {'type': 'RuntimeError', 'message': "You've hit your session limit · resets 5:30pm"},
        'grading': None})
    target.joinpath(f'{name}.log').write_text('Preserved quota output')
    trace = target / 'traces' / f'{name}.jsonl'
    trace.parent.mkdir(exist_ok=True)
    trace.write_text('{"kind":"run_error","message":"quota"}\n')


def mock_services(recovery, saved, monkeypatch, *, missing_namespaces=False):
    requests, servers = [], []
    monkeypatch.setattr(recovery, 'active_worker_pids', lambda *args: [])
    monkeypatch.setenv('COUCHDB_URL', 'http://db.invalid:5984')
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            if missing_namespaces:
                return []
            return [f'eval_{cohort}_claude_saved_{db}' for cohort in ['existing', 'synthetic'] for db in saved['snapshot']]
    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url):
            requests.append(('GET', url))
            return Response()
    class Server:
        server_port = 12345
        def serve_forever(self):
            pass
        def shutdown(self):
            self.stopped = True
        def server_close(self):
            self.closed = True
    def serve(base, auth, prefix, audit, port, **kwargs):
        server = Server()
        server.prefix = prefix
        servers.append(server)
        return server
    monkeypatch.setattr(recovery.httpx, 'Client', Client)
    monkeypatch.setattr(recovery, 'serve', serve)
    return requests, servers


def test_plan_is_readonly_and_keeps_original_order_and_completed_cases(recovery, saved):
    before = {p: p.read_bytes() for p in saved['experiment_path'].parent.rglob('*') if p.is_file()}
    experiment, snapshot, plans = recovery.build_plan(saved['experiment_path'])
    assert len(plans) == 1
    assert plans[0]['scenario_ids'] == ['s2', 's3']
    assert plans[0]['prior_attempts'] == {'s2': 3, 's3': 0}
    assert plans[0]['pending_grade_ids'] == []
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert not list(saved['experiment_path'].parent.rglob('recoveries'))


def test_explicit_recovery_reuses_namespace_keeps_failed_evidence_and_runtime_identity(
        recovery, saved, monkeypatch, tmp_path):
    experiment, snapshot, plans = recovery.build_plan(saved['experiment_path'])
    requests, servers = mock_services(recovery, saved, monkeypatch)
    prior = {p: p.read_bytes() for p in saved['target'].rglob('*') if p.is_file()}
    profile = tmp_path / 'isolated-claude'
    profile.mkdir()
    secret = profile / 'credentials.json'
    secret.write_text('private credentials must never be read or copied')
    parent_profile = os.environ.get('CLAUDE_CONFIG_DIR')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['CLAUDE_CONFIG_DIR'] == str(profile)
        assert kwargs['env']['BENCHMARK_DB_PROXY_URL'] == 'http://127.0.0.1:12345'
        manifest = json.loads(Path(command[command.index('--quota-recovery-file') + 1]).read_text())
        assert manifest['scenario_ids'] == ['s2', 's3'] and manifest['max_new_attempts'] == 3
        for i, attempt in [(2, 4), (3, 1)]:
            complete_record(saved['target'], saved['spec'], manifest['saved_settings'], i,
                            graded=True, attempt=attempt,
                            recovery={'episode_id': manifest['episode_id'], 'prior_attempts': attempt - 1})
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(recovery.subprocess, 'run', run)
    recovery.execute_recovery(experiment, snapshot, plans, experiment_file=saved['experiment_path'], claude_config_dir=profile)
    assert len(calls) == 1
    assert all(p.read_bytes() == contents for p, contents in prior.items() if p.name != 'active-run.txt')
    assert os.environ.get('CLAUDE_CONFIG_DIR') == parent_profile
    assert requests == [('GET', 'http://db.invalid:5984/_all_dbs')]
    assert servers[0].prefix == 'eval_synthetic_claude_saved_' and servers[0].stopped and servers[0].closed
    final = json.loads(saved['experiment_path'].read_text())
    assert final['status'] == 'completed' and final['recoveries'][0]['status'] == 'completed'
    manifest, = saved['target'].glob('recoveries/*.json')
    assert 'private credentials' not in manifest.read_text()


@pytest.mark.parametrize('mutation', ['model', 'agent', 'judge', 'reasoning', 'snapshot', 'namespace', 'rubric', 'version'])
def test_identity_mismatch_refuses_before_any_recovery_writes(recovery, saved, monkeypatch, mutation):
    target = saved['target']
    settings_path = target / 'settings.json'
    settings = json.loads(settings_path.read_text())
    if mutation in ['model', 'agent']:
        settings[mutation] = 'wrong'
    elif mutation == 'judge':
        settings['judge_model'] = 'wrong-judge'
    elif mutation == 'reasoning':
        experiment = json.loads(saved['experiment_path'].read_text())
        for spec in experiment['config']['targets']:
            spec['reasoning_effort'] = 'low'
        write_json(saved['experiment_path'], experiment)
    elif mutation == 'snapshot':
        settings['database_policy']['snapshot_sha256'] = 'other'
        write_json(target / 'environment.json', settings['database_policy'])
    elif mutation == 'namespace':
        settings['database_policy']['namespace'] = 'production_'
        write_json(target / 'environment.json', settings['database_policy'])
    elif mutation == 'rubric':
        settings['rubric_sha256'] = 'other'
    elif mutation == 'version':
        monkeypatch.setattr(recovery, 'versions', lambda _: {'sdk_version': 'changed', 'cli_version': 'fixture-cli'})
    write_json(settings_path, settings)
    with pytest.raises(ValueError):
        recovery.build_plan(saved['experiment_path'])
    assert not list(saved['experiment_path'].parent.rglob('recoveries'))
    assert not list(saved['experiment_path'].parent.rglob('.recovery.lock'))


def test_nonquota_exhausted_failure_cannot_be_restarted(recovery, saved):
    path = saved['target'] / 'measurements/synthetic-claude_0002.attempt-3.json'
    record = json.loads(path.read_text())
    record['agent_error']['message'] = 'Tool invocation failed'
    write_json(path, record)
    with pytest.raises(ValueError, match='exhausted non-quota'):
        recovery.build_plan(saved['experiment_path'])


def test_missing_persisted_namespace_refuses_to_clone_or_launch(recovery, saved, monkeypatch):
    experiment, snapshot, plans = recovery.build_plan(saved['experiment_path'])
    requests, servers = mock_services(recovery, saved, monkeypatch, missing_namespaces=True)
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *args, **kwargs: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='refusing to reclone'):
        recovery.execute_recovery(experiment, snapshot, plans, experiment_file=saved['experiment_path'])
    assert len(requests) == 1 and servers == []
    assert not list(saved['experiment_path'].parent.rglob('recoveries'))


def test_active_native_workers_and_controllers_are_detected_without_echoing_args(recovery, saved):
    _, _, plans = recovery.build_plan(saved['experiment_path'])
    target = saved['target']
    processes = f'''112 python -m benchmark.generated_suite_runner suite --output-dir {target.parent} --name synthetic-claude
113 python tools/grade_live_comparison.py --target {target} --suite suite
114 python -m agent.claude_agent.cli --run-id synthetic-claude_0002 --json secret-text
115 python tools/run_asset_cohort_comparison.py --output-dir {saved['experiment_path'].parent}
116 python unrelated.py --name synthetic-claude
'''
    assert recovery.active_worker_pids(plans, saved['experiment_path'], process_output=processes) == [112, 113, 114, 115]


def test_still_blocked_quota_stops_after_one_new_attempt_without_churn(recovery, saved, monkeypatch):
    experiment, snapshot, plans = recovery.build_plan(saved['experiment_path'])
    mock_services(recovery, saved, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        quota_record(saved['target'], saved['spec'], plans[0]['saved_settings'], 2, 4)
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(recovery.subprocess, 'run', run)
    with pytest.raises(ValueError, match='quota is still blocked'):
        recovery.execute_recovery(experiment, snapshot, plans, experiment_file=saved['experiment_path'])
    assert len(calls) == 1
    assert json.loads(saved['experiment_path'].read_text())['status'] == 'failed'


def test_grade_only_recovery_uses_explicit_provider_resume_and_no_database_proxy(recovery, saved, monkeypatch):
    settings = json.loads((saved['target'] / 'settings.json').read_text())
    for i in [2, 3]:
        for path in (saved['target'] / 'measurements').glob(f'synthetic-claude_{i:04d}*.json'):
            path.unlink()
        complete_record(saved['target'], saved['spec'], settings, i, graded=True)
    path = saved['target'] / 'measurements/synthetic-claude_0001.json'
    record = json.loads(path.read_text())
    original_attempt = {'status': 'failed', 'error': {'message': "You've hit your session limit"}}
    record['grading'].update(status='failed', result=None, attempts=[original_attempt], error=original_attempt['error'])
    write_json(path, record)
    experiment, snapshot, plans = recovery.build_plan(saved['experiment_path'])
    assert plans[0]['scenario_ids'] == [] and plans[0]['pending_grade_ids'] == ['s1']
    _, servers = mock_services(recovery, saved, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[1] == '-c' and 'resume_provider_quota=True' in command[2]
        current = json.loads(path.read_text())
        current['grading'].update(status='completed', result={'score': {'passed': True, 'score': 1, 'details': {}}})
        current['grading']['attempts'].append({'status': 'completed'})
        write_json(path, current)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(recovery.subprocess, 'run', run)
    recovery.execute_recovery(experiment, snapshot, plans, experiment_file=saved['experiment_path'])
    assert len(calls) == 1 and servers == []
    assert json.loads(path.read_text())['grading']['attempts'][0] == original_attempt
    assert (saved['target'] / 'measurements/synthetic-claude_0001.attempt-2.json').exists() is False


def test_default_cli_is_a_reviewable_plan_and_never_runs_models(recovery, saved, monkeypatch, capsys):
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *args, **kwargs: pytest.fail('must not launch'))
    recovery.main(['--experiment-file', str(saved['experiment_path'])])
    summary = json.loads(capsys.readouterr().out)
    assert summary['mode'] == 'plan'
    assert summary['targets'][0]['execution_scenario_ids'] == ['s2', 's3']
    assert not list(saved['experiment_path'].parent.rglob('recoveries'))


def test_manifest_matches_native_runner_selected_attempt_schema(recovery, saved):
    from benchmark.generated_suite_runner import validate_quota_recovery
    _, _, plans = recovery.build_plan(saved['experiment_path'])
    plan = plans[0]
    manifest = {**plan, 'episode_id': 'reviewable-episode', 'reason': 'manual Claude quota recovery',
                'max_new_attempts': 3}
    path = saved['experiment_path'].parent / 'test-manifest.json'
    write_json(path, manifest)
    rows, _ = completed_scenarios(Path(plan['spec']['suite']))
    assert validate_quota_recovery(path, plan['saved_settings'], rows, saved['target']) == manifest
    assert set(manifest['prior_attempts']) == set(manifest['scenario_ids'])


def test_pinned_executable_is_local_and_path_is_restored_even_on_failure(recovery, tmp_path):
    executable = tmp_path / '2.1.286'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    original_path = os.environ.get('PATH')
    with pytest.raises(RuntimeError, match='fixture stop'):
        with recovery.pinned_claude_executable(executable):
            alias = Path(shutil.which('claude'))
            assert alias.resolve() == executable
            assert alias.parent.name.startswith('assetops-claude-cli-')
            raise RuntimeError('fixture stop')
    assert os.environ.get('PATH') == original_path
    assert executable.exists() and not alias.exists()


def test_grade_only_recovery_rejects_changed_judge_cli_version(recovery, saved):
    path = saved['target'] / 'measurements/synthetic-claude_0001.json'
    record = json.loads(path.read_text())
    record['grading']['runtime_versions'] = {'sdk_version': 'fixture-sdk', 'cli_version': 'different-judge-cli'}
    write_json(path, record)
    with pytest.raises(ValueError, match='current judge SDK/CLI'):
        recovery.build_plan(saved['experiment_path'])


def test_new_recovery_grading_quota_stops_without_execution_retry_churn(recovery, saved, monkeypatch):
    experiment, snapshot, plans = recovery.build_plan(saved['experiment_path'])
    mock_services(recovery, saved, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        for i, attempt in [(2, 4), (3, 1)]:
            complete_record(saved['target'], saved['spec'], plans[0]['saved_settings'], i,
                            graded=True, attempt=attempt)
        path = saved['target'] / 'measurements/synthetic-claude_0002.attempt-4.json'
        record = json.loads(path.read_text())
        record['grading'].update(status='failed', result=None,
                                 error={'message': "You've hit your session limit"})
        write_json(path, record)
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(recovery.subprocess, 'run', run)
    with pytest.raises(ValueError, match='Fable judge quota is still blocked'):
        recovery.execute_recovery(experiment, snapshot, plans, experiment_file=saved['experiment_path'])
    assert len(calls) == 1
