"""Plan or explicitly resume Claude quota-blocked work in saved database namespaces.

Without --resume this prints a reviewable plan and does not launch workers or write
artifacts. A recovery is a new transport-retry episode, not a fresh repetition.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from datetime import datetime, UTC
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from benchmark.database_audit import serve
from benchmark.generated_suite_runner import AGENTS, completed_scenarios
from benchmark.measurement import suite_hash, versions, write_json
from benchmark.repeated_comparison import aggregate
from evaluation.models import PersistedTrajectory
from run_asset_cohort_comparison import validate_cohort_targets

QUOTA_MESSAGES = ("you've hit your session limit", "you’ve hit your session limit",
                  "you've hit your usage limit", "you’ve hit your usage limit",
                  "you've hit your weekly limit", "you’ve hit your weekly limit",
                  'extra usage is required')
GRADE_COMMAND = (
    "import json,sys; from pathlib import Path; "
    "from benchmark.measured_grading import grade_target; "
    "p=json.loads(Path(sys.argv[1]).read_text()); "
    "grade_target(Path(p['target_dir']),[Path(f) for f in p['suite_files']],"
    "p['saved_settings']['judge_model'],"
    "allow_self_judge=p['saved_settings'].get('allow_same_model_judge',False),"
    "resume_provider_quota=True)"
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def pinned_claude_executable(executable):
    """Pin lookup for versions, SDK and judge without changing the global CLI."""
    if executable is None:
        yield
        return
    executable = Path(executable).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError('claude-executable must be an existing executable file')
    original = os.environ.get('PATH')
    with TemporaryDirectory(prefix='assetops-claude-cli-') as directory:
        (Path(directory) / 'claude').symlink_to(executable)
        os.environ['PATH'] = directory + os.pathsep + (original or '')
        try:
            yield
        finally:
            if original is None:
                os.environ.pop('PATH', None)
            else:
                os.environ['PATH'] = original


def is_quota_failure(record):
    if record.get('status') not in {'failed', 'timed_out', 'cancelled'}:
        return False
    messages = [str((record.get(key) or {}).get('message', '')).casefold()
                for key in ('agent_error', 'error')]
    return any(marker in message for marker in QUOTA_MESSAGES for message in messages)


def grading_quota_blocked(record):
    grading = record.get('grading') or {}
    return is_quota_failure({'status': grading.get('status'), 'error': grading.get('error')})


def records_for(target):
    return [(path, json.loads(path.read_text()))
            for path in sorted((target / 'measurements').glob('*.json'))]


def latest_records(records):
    latest, seen = {}, set()
    for path, record in records:
        key = record['scenario_id'], record['attempt']
        if key in seen:
            raise ValueError('Duplicate scenario/attempt measurements')
        seen.add(key)
        if key[0] not in latest or record['attempt'] > latest[key[0]][1]['attempt']:
            latest[key[0]] = path, record
    return latest


def validate_target(spec, rep, experiment, digest):
    target = Path(rep['root']) / spec['name']
    suite = Path(spec['suite'])
    rows, files = completed_scenarios(suite)
    expected = {row.id: row for row in rows}
    suite_digest = suite_hash(files)
    if experiment['suite_hashes'][spec['name']] != suite_digest:
        raise ValueError(f'{spec["name"]}: suite differs from the frozen experiment')
    settings = json.loads((target / 'settings.json').read_text())
    policy = json.loads((target / 'environment.json').read_text())
    metadata = json.loads((target / 'target.json').read_text())
    if metadata != {'generation_run': str(suite.resolve()), 'agent': spec['agent'], 'model': spec['model_id']}:
        raise ValueError(f'{spec["name"]}: target metadata differs from suite/model/agent')
    if (settings.get('model') != spec['model_id'] or settings.get('agent') != spec['agent'] or
            settings.get('judge_model') != experiment['config']['judge'] or
            settings.get('suite_sha256') != suite_digest or
            settings.get('repetition_index') != rep['index'] or
            (spec.get('reasoning_effort') is not None and
             settings.get('reasoning_effort') != spec['reasoning_effort'])):
        raise ValueError(f'{spec["name"]}: saved model/agent/judge/suite/reasoning/repetition differs')
    if (settings.get('database_policy') != policy or policy.get('snapshot_sha256') != digest or
            Path(policy.get('suite', '')).resolve() != suite.resolve() or
            policy.get('cohort') != spec['cohort']):
        raise ValueError(f'{spec["name"]}: saved database policy differs from environment/snapshot/cohort')
    prefix = policy.get('namespace', '')
    if not re.fullmatch(r'eval_[a-z0-9_]+', prefix) or not prefix.startswith(
            'eval_' + spec['name'].lower().replace('-', '_') + '_'):
        raise ValueError(f'{spec["name"]}: unsafe or unrelated saved database namespace')
    rubric = hashlib.sha256((ROOT / 'src/evaluation/scorers/llm_judge.py').read_bytes()).hexdigest()
    if settings.get('rubric_sha256') != rubric:
        raise ValueError(f'{spec["name"]}: judge rubric changed since execution')
    # A retired executor can still have its saved answers graded. Its runtime
    # is not invoked or relabeled; execution recovery remains Claude-only.
    current_versions = versions(spec['agent']) if spec['agent'] in AGENTS else {}
    if any(settings.get(key) != value for key, value in current_versions.items()):
        raise ValueError(f'{spec["name"]}: current SDK/CLI versions differ from recorded runtime')
    records = records_for(target)
    latest = latest_records(records)
    judge_versions = versions('claude') if settings['judge_model'].startswith('claude-code/') else None
    for _, record in records:
        if record['scenario_id'] not in expected or record['settings'] != settings:
            raise ValueError(f'{spec["name"]}: saved measurement scenario/runtime identity differs')
        index = next(i for i, row in enumerate(rows, 1) if row.id == record['scenario_id'])
        if record.get('run_id') != f'{spec["name"]}_{index:04d}' or record.get('execution_index') != index:
            raise ValueError(f'{spec["name"]}: saved measurement execution order differs')
        grading = record.get('grading') or {}
        if grading and grading.get('model') != settings['judge_model']:
            raise ValueError(f'{spec["name"]}: saved grading judge differs')
        if (judge_versions is not None and grading.get('runtime_versions') is not None and
                grading['runtime_versions'] != judge_versions):
            raise ValueError(f'{spec["name"]}: current judge SDK/CLI versions differ from recorded runtime')
    execute_ids, grade_ids, prior = [], [], {}
    for row in rows:
        entry = latest.get(row.id)
        prior[row.id] = entry[1]['attempt'] if entry else 0
        if entry is None:
            if spec['agent'] != 'claude':
                raise ValueError(f'{spec["name"]}: unstarted non-Claude executions require their native resume workflow')
            execute_ids.append(row.id)
            continue
        record = entry[1]
        if record['status'] == 'completed':
            trajectory = PersistedTrajectory.from_raw(json.loads(
                (target / 'trajectories' / f'{record["run_id"]}.json').read_text()))
            if (trajectory.scenario_id != row.id or trajectory.model != spec['model_id'] or
                    trajectory.question != row.text):
                raise ValueError(f'{spec["name"]}: completed trajectory differs from its scenario/model')
            if (record.get('grading') or {}).get('status') != 'completed':
                grade_ids.append(row.id)
            continue
        if record['status'] == 'running':
            raise ValueError(f'{spec["name"]}: an execution measurement is still running')
        if spec['agent'] != 'claude' or not is_quota_failure(record):
            budget = (settings.get('invocation_retry_policy') or {}).get('max_attempts', 3)
            qualifier = 'exhausted non-quota failure' if record['attempt'] >= budget else 'non-quota failure'
            raise ValueError(f'{spec["name"]}: refusing manual recovery of {qualifier} for {row.id}')
        if any(not is_quota_failure(r) for _, r in records if r['scenario_id'] == row.id):
            raise ValueError(f'{spec["name"]}: case {row.id} was not blocked solely by Claude quota')
        execute_ids.append(row.id)
    return {'name': spec['name'], 'spec': spec, 'repetition_index': rep['index'],
            'target_dir': str(target), 'suite_files': [str(path) for path in files],
            'environment': policy, 'saved_settings': settings, 'scenario_ids': execute_ids,
            'pending_grade_ids': grade_ids, 'prior_attempts': {sid: prior[sid] for sid in execute_ids}}


def build_plan(experiment_file, *, target_names=None, repetition_index=None, snapshot_file=None):
    experiment_file = Path(experiment_file).resolve()
    experiment = json.loads(experiment_file.read_text())
    config = experiment['config']
    for spec in config['targets']:
        spec['suite'] = str((experiment_file.parent / spec['suite']).resolve())
    for rep in experiment['repetitions']:
        rep['root'] = str((experiment_file.parent / rep['root']).resolve())
    cohorts = {cohort: [s for s in config['targets'] if s['cohort'] == cohort]
               for cohort in ('existing', 'synthetic')}
    validate_cohort_targets(cohorts)
    if len({s['name'] for s in config['targets']}) != len(config['targets']):
        raise ValueError('Target names must be unique')
    if any(not re.fullmatch(r'[a-zA-Z0-9_-]+', s['name']) for s in config['targets']):
        raise ValueError('Target names must be safe directory names')
    chosen = set(target_names or [s['name'] for s in config['targets']])
    if chosen - {s['name'] for s in config['targets']}:
        raise ValueError('Unknown selected target')
    if repetition_index is not None and repetition_index not in {r['index'] for r in experiment['repetitions']}:
        raise ValueError('Unknown repetition index')
    snapshot_path = Path(snapshot_file or experiment['snapshot_file'])
    if not snapshot_path.is_absolute():
        snapshot_path = experiment_file.parent / snapshot_path
    snapshot = json.loads(gzip.decompress(snapshot_path.read_bytes()))
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    if digest != experiment['snapshot_sha256']:
        raise ValueError('Captured database snapshot differs from the frozen experiment')
    plans, namespaces = [], set()
    for rep in experiment['repetitions']:
        if repetition_index is not None and rep['index'] != repetition_index:
            continue
        for spec in config['targets']:
            if spec['name'] not in chosen:
                continue
            plan = validate_target(spec, rep, experiment, digest)
            if (plan['environment'].get('database_count') != len(snapshot) or
                    plan['environment'].get('document_count') != sum(map(len, snapshot.values()))):
                raise ValueError(f'{spec["name"]}: saved initial database/document counts differ from snapshot')
            namespace = plan['environment']['namespace']
            if namespace in namespaces:
                raise ValueError('Recovery targets must have independent saved namespaces')
            namespaces.add(namespace)
            if plan['scenario_ids'] or plan['pending_grade_ids']:
                plans.append(plan)
    return experiment, snapshot, plans


def active_worker_pids(plans, experiment_file, *, process_output=None):
    """Inspect process arguments locally; return only IDs, never credentials/argv."""
    if process_output is None:
        process_output = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    active = []
    labels = ('benchmark.generated_suite_runner', 'agent.claude_agent.cli',
              'agent.openai_agent.cli', 'grade_live_comparison.py',
              'run_generated_comparison.py', 'run_asset_cohort_comparison.py',
              'resume_asset_cohort_comparison.py', 'grade_target(')
    experiment_file = Path(experiment_file).resolve()
    for line in process_output.splitlines():
        columns = line.strip().split(None, 1)
        if len(columns) != 2 or not columns[0].isdigit() or int(columns[0]) == os.getpid():
            continue
        pid, command = int(columns[0]), columns[1]
        if not any(label in command for label in labels):
            continue
        try:
            args = shlex.split(command)
        except ValueError:
            args = command.split()
        def option(flag):
            try:
                return args[args.index(flag) + 1]
            except (ValueError, IndexError):
                return None
        global_match = (option('--output-dir') == str(experiment_file.parent) or
                        option('--experiment-file') == str(experiment_file))
        target_match = any(
            str(Path(plan['target_dir'])) in command or
            option('--output-dir') == str(Path(plan['target_dir']).parent) or
            option('--name') == plan['name'] or
            (option('--run-id') or '').startswith(plan['name'] + '_')
            for plan in plans)
        if global_match or target_match:
            active.append(pid)
    return sorted(set(active))


def validate_namespaces(plans, snapshot, client, base):
    response = client.get(base + '/_all_dbs')
    response.raise_for_status()
    databases = set(response.json())
    for plan in plans:
        prefix = plan['environment']['namespace']
        if {prefix + name for name in snapshot} - databases:
            raise ValueError(f'{plan["name"]}: persisted namespace databases are missing; refusing to reclone')


def locked(path, stack):
    handle = stack.enter_context(path.open('a'))
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError('Another recovery owns this experiment/target') from exc
    return handle


def runner_command(plan, manifest):
    settings = plan['saved_settings']
    spec = plan['spec']
    command = [sys.executable, '-m', 'benchmark.generated_suite_runner', spec['suite'],
               '--output-dir', str(Path(plan['target_dir']).parent), '--name', spec['name'],
               '--agent', spec['agent'], '--model-id', spec['model_id'],
               '--judge-model', settings['judge_model'],
               '--timeout', str(settings['timeout_seconds']),
               '--concurrency-level', str(settings['concurrency_level']),
               '--max-invocation-attempts', str(settings['invocation_retry_policy']['max_attempts']),
               '--quota-recovery-file', str(manifest)]
    if settings.get('allow_same_model_judge'):
        command.append('--allow-self-judge')
    if spec.get('reasoning_effort'):
        command += ['--reasoning-effort', spec['reasoning_effort']]
    return command


def finish_experiment(experiment, experiment_file):
    complete = True
    for rep in experiment['repetitions']:
        for cohort in ('existing', 'synthetic'):
            group_complete = True
            for spec in experiment['config']['targets']:
                if spec['cohort'] != cohort:
                    continue
                rows, _ = completed_scenarios(Path(spec['suite']))
                records = records_for(Path(rep['root']) / spec['name'])
                result = aggregate([(rep['index'], [r for _, r in records])], {row.id for row in rows})
                blocked = any(is_quota_failure(r) for _, r in latest_records(records).values())
                group_complete &= result['complete_repetitions'] == 1 and not blocked
            rep.setdefault('cohorts', {})[cohort] = 'completed' if group_complete else 'pending'
            complete &= group_complete
    experiment.update(status='completed' if complete else 'failed', end=datetime.now(UTC).isoformat())
    if complete:
        experiment.pop('error', None)
    else:
        experiment['error'] = {'type': 'IncompleteRecovery', 'message': 'Quota-blocked or ungraded targets remain'}
    write_json(experiment_file, experiment)


def execute_recovery(experiment, snapshot, plans, *, experiment_file, claude_config_dir=None,
                     claude_executable=None, max_concurrent_targets=2):
    experiment_file = Path(experiment_file).resolve()
    active = active_worker_pids(plans, experiment_file)
    if active:
        raise ValueError(f'Refusing recovery while matching worker/controller PIDs are active: {active}')
    base = os.environ['COUCHDB_URL'].rstrip('/')
    auth = (os.environ.get('COUCHDB_USERNAME', 'admin'), os.environ.get('COUCHDB_PASSWORD', 'password'))
    with httpx.Client(auth=auth, timeout=120) as client:
        validate_namespaces(plans, snapshot, client, base)
    # All identities and namespace existence are checked before the first write.
    with ExitStack() as stack:
        locked(experiment_file.parent / '.quota-recovery.lock', stack)
        for plan in plans:
            locked(Path(plan['target_dir']) / '.recovery.lock', stack)
        episode_id = uuid4().hex
        session = {'episode_id': episode_id, 'reason': 'manual Claude quota recovery',
                   'start': datetime.now(UTC).isoformat(), 'end': None, 'status': 'running',
                   'pid': os.getpid(), 'claude_config_dir': str(claude_config_dir) if claude_config_dir else None,
                   'claude_executable': str(claude_executable) if claude_executable else None,
                   'claude_executable_sha256': file_hash(claude_executable) if claude_executable else None,
                   'max_concurrent_targets': max_concurrent_targets, 'targets': []}
        if experiment.get('error'):
            experiment.setdefault('interruptions', []).append({
                'end': experiment.get('end'), 'error': experiment.pop('error')})
        experiment.setdefault('recoveries', []).append(session)
        experiment.update(status='running', end=None)
        write_json(experiment_file, experiment)

        def recover(plan):
            target = Path(plan['target_dir'])
            manifest_path = target / 'recoveries' / f'{episode_id}.json'
            manifest = {**plan, 'episode_id': episode_id, 'reason': session['reason'],
                        'max_new_attempts': 3, 'status': 'running', 'start': datetime.now(UTC).isoformat(),
                        'target': {'name': plan['name'], 'agent': plan['spec']['agent'],
                                   'model_id': plan['spec']['model_id'], 'suite': plan['spec']['suite'],
                                   'repetition_index': plan['repetition_index'],
                                   'namespace': plan['environment']['namespace'],
                                   'snapshot_sha256': experiment['snapshot_sha256'],
                                   'suite_sha256': plan['saved_settings']['suite_sha256']},
                        'claude_config_dir': session['claude_config_dir'],
                        'claude_executable': session['claude_executable'],
                        'claude_executable_sha256': session['claude_executable_sha256']}
            write_json(manifest_path, manifest)
            server = None
            try:
                env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONUNBUFFERED': '1',
                       'BENCHMARK_REPETITION_INDEX': str(plan['repetition_index'])}
                if claude_config_dir:
                    env['CLAUDE_CONFIG_DIR'] = str(claude_config_dir)
                if plan['scenario_ids']:
                    audit = target / 'database_audit'
                    active_run = target / 'active-run.txt'
                    active_run.write_text('preflight')
                    server = serve(base, auth, plan['environment']['namespace'], audit, 0, run_id_file=active_run)
                    threading.Thread(target=server.serve_forever, daemon=True).start()
                    env.update(BENCHMARK_DB_PROXY_URL=f'http://127.0.0.1:{server.server_port}',
                               BENCHMARK_DB_AUDIT_DIR=str(audit), BENCHMARK_DB_RUN_ID_FILE=str(active_run))
                    command = runner_command(plan, manifest_path)
                    # One runner invocation may stop at one failed scenario. The
                    # manifest enforces a fixed three-attempt budget per case in
                    # this episode across all invocations, retaining old attempts.
                    for invocation in range(1, len(plan['scenario_ids']) * 3 + 2):
                        before_attempts = {sid: r['attempt'] for sid, (_, r) in
                                           latest_records(records_for(target)).items()}
                        with (manifest_path.parent / f'{episode_id}.invocation-{invocation}.log').open('x') as log:
                            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                                     stderr=subprocess.STDOUT, check=False)
                        if process.returncode == 0:
                            break
                        latest = latest_records(records_for(target))
                        if any(is_quota_failure(r) for sid, (_, r) in latest.items()
                               if sid in plan['scenario_ids'] and r['attempt'] > plan['prior_attempts'][sid]):
                            raise ValueError('Claude quota is still blocked; recovery stopped without further retry churn')
                        if any(grading_quota_blocked(r) for _, r in latest.values()):
                            raise ValueError('Fable judge quota is still blocked; recovery stopped without further retry churn')
                        if before_attempts == {sid: r['attempt'] for sid, (_, r) in latest.items()}:
                            raise subprocess.CalledProcessError(process.returncode, command)
                        if invocation == len(plan['scenario_ids']) * 3 + 1:
                            raise subprocess.CalledProcessError(process.returncode, command)
                else:
                    with (manifest_path.parent / f'{episode_id}.grading.log').open('x') as log:
                        subprocess.run([sys.executable, '-c', GRADE_COMMAND, str(manifest_path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                # Validate outcome completeness after both execution and grading.
                rows, _ = completed_scenarios(Path(plan['spec']['suite']))
                records = records_for(target)
                aggregate([(plan['repetition_index'], [r for _, r in records])],
                          {row.id for row in rows}, require_complete=True)
                if any(is_quota_failure(r) for _, r in latest_records(records).values()):
                    raise ValueError('Claude quota-blocked executions remain')
                manifest['status'] = 'completed'
            except BaseException as exc:
                manifest.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
                raise
            finally:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                manifest['end'] = datetime.now(UTC).isoformat()
                write_json(manifest_path, manifest)
            return {'name': plan['name'], 'repetition_index': plan['repetition_index'],
                    'manifest': str(manifest_path), 'status': 'completed'}

        failures = []
        with ThreadPoolExecutor(max_workers=max_concurrent_targets) as executor:
            futures = {executor.submit(recover, plan): plan for plan in plans}
            for future in as_completed(futures):
                plan = futures[future]
                try:
                    session['targets'].append(future.result())
                except Exception as exc:
                    failures.append(exc)
                    session['targets'].append({'name': plan['name'], 'repetition_index': plan['repetition_index'],
                                               'status': 'failed', 'error': {'type': type(exc).__name__, 'message': str(exc)}})
                write_json(experiment_file, experiment)
        session.update(status='failed' if failures else 'completed', end=datetime.now(UTC).isoformat())
        finish_experiment(experiment, experiment_file)
        if failures:
            raise failures[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-file', type=Path, required=True)
    parser.add_argument('--snapshot-file', type=Path)
    parser.add_argument('--target', action='append', help='Restrict to named targets; may be repeated')
    parser.add_argument('--repetition-index', type=int)
    parser.add_argument('--claude-config-dir', type=Path, help='Use this existing isolated Claude profile for workers/judge')
    parser.add_argument('--claude-executable', type=Path, help='Pin an existing Claude binary without changing global CLI/auth')
    parser.add_argument('--max-concurrent-targets', type=int, default=2)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--resume', action='store_true', help='Execute the validated plan; otherwise only inspect it')
    args = parser.parse_args(argv)
    if args.max_concurrent_targets <= 0:
        parser.error('max-concurrent-targets must be positive')
    load_dotenv(ROOT / '.env')
    if args.env_file:
        load_dotenv(args.env_file)
    profile = args.claude_config_dir.resolve() if args.claude_config_dir else None
    executable = args.claude_executable.resolve() if args.claude_executable else None
    if profile and not profile.is_dir():
        parser.error('claude-config-dir must be an existing isolated profile directory')
    try:
        with pinned_claude_executable(executable):
            experiment, snapshot, plans = build_plan(
                args.experiment_file, target_names=args.target,
                repetition_index=args.repetition_index, snapshot_file=args.snapshot_file)
            summary = {'mode': 'resume' if args.resume else 'plan',
                       'snapshot_sha256': experiment['snapshot_sha256'],
                       'claude_config_dir': str(profile) if profile else None,
                       'claude_executable': str(executable) if executable else None,
                       'transport_recovery': 'new episode of up to three attempts; original runtime settings and failed evidence preserved',
                       'targets': [{'name': p['name'], 'repetition_index': p['repetition_index'],
                                    'namespace': p['environment']['namespace'],
                                    'execution_scenario_ids': p['scenario_ids'],
                                    'pending_grade_ids': p['pending_grade_ids']} for p in plans]}
            print(json.dumps(summary, indent=2), flush=True)
            if args.resume and plans:
                execute_recovery(experiment, snapshot, plans, experiment_file=args.experiment_file,
                                 claude_config_dir=profile, claude_executable=executable,
                                 max_concurrent_targets=args.max_concurrent_targets)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Recovery refused/failed: {exc}\n')


if __name__ == '__main__':
    main()
