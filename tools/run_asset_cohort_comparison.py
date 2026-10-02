"""Run k matched repetitions with bounded parallel existing/synthetic cohort groups."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, UTC
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark.generated_suite_runner import completed_scenarios
from benchmark.measurement import suite_hash, write_json
from benchmark.repeated_comparison import aggregate


COHORTS = ('existing', 'synthetic')


def target_suite(spec):
    path = Path(spec['suite'])
    return path if path.is_absolute() else ROOT / path


def validate_cohort_targets(cohorts):
    """Each model key identifies the same configured executor in both cohorts."""
    identities = {}
    for cohort, targets in cohorts.items():
        identities[cohort] = {}
        for spec in targets:
            key = spec.get('model_key', spec['model_id'])
            if key in identities[cohort]:
                raise ValueError(f'Duplicate model key {key!r} in {cohort} cohort')
            identities[cohort][key] = (spec['model_id'], spec['agent'], spec.get('reasoning_effort'))
    if identities['existing'] != identities['synthetic']:
        raise ValueError('Existing and synthetic cohorts differ in executor models, agents, '
                         'reasoning_effort or model keys')


def snapshot_hash(path):
    """Use the launcher's canonical digest before any parallel database clones."""
    snapshot = json.loads(gzip.decompress(path.read_bytes()))
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def group_snapshot(root, targets):
    snapshots = {
        json.loads((root / spec['name'] / 'environment.json').read_text())['snapshot_sha256']
        for spec in targets
    }
    if len(snapshots) != 1:
        raise ValueError('Cohort targets did not share one snapshot')
    return snapshots.pop()


def completed_group(root, targets, snapshot_sha=None, *, judge_model=None):
    """Validate every target before deciding whether a group can safely start."""
    complete, has_existing_state = [], False
    for spec in targets:
        rows, files = completed_scenarios(target_suite(spec))
        digest = suite_hash(files)
        target = root / spec['name']
        records = [json.loads(p.read_text()) for p in (target / 'measurements').glob('*.json')]
        environment_path = target / 'environment.json'
        environment = json.loads(environment_path.read_text()) if environment_path.exists() else None
        has_existing_state |= bool(records) or environment is not None
        if environment is not None and snapshot_sha and environment['snapshot_sha256'] != snapshot_sha:
            raise ValueError('Cohorts/repetitions did not share one snapshot')
        target_snapshot = snapshot_sha or (environment or {}).get('snapshot_sha256')
        for record in records:
            settings = record['settings']
            if (spec.get('reasoning_effort') is not None and
                    settings.get('reasoning_effort') != spec['reasoning_effort']):
                raise ValueError('Existing measurements differ in configured reasoning_effort')
            if (settings.get('suite_sha256') != digest or
                    settings.get('model') != spec['model_id'] or
                    (settings.get('agent') is not None and settings['agent'] != spec['agent']) or
                    (judge_model is not None and settings.get('judge_model') != judge_model) or
                    (target_snapshot and
                     (settings.get('database_policy') or {}).get('snapshot_sha256') != target_snapshot)):
                raise ValueError('Existing measurements differ in suite/model/agent/judge/snapshot')
        result = aggregate([(1, records)], {r.id for r in rows})
        complete.append(result['complete_repetitions'] == 1)
    if all(complete):
        digest = group_snapshot(root, targets)
        if snapshot_sha and digest != snapshot_sha:
            raise ValueError('Cohorts/repetitions did not share one snapshot')
        return True
    if has_existing_state:
        raise ValueError('Partial cohort execution needs the single-target resume command; '
                         'refusing to reset its database')
    return False


def execute_groups(jobs, *, experiment, experiment_path, snapshot_file,
                   group_configs, config, max_concurrent_groups):
    """Only the coordinating thread changes the experiment or schedules new work."""
    pending = iter(jobs)
    active = {}
    failure = None

    def event(kind, rep, cohort, **extra):
        experiment.setdefault('scheduling_history', []).append({
            'event': kind, 'timestamp': datetime.now(UTC).isoformat(),
            'repetition': rep['index'], 'cohort': cohort,
            'active_cohort_groups': len(active),
            'active_group_target_capacity': sum(len(job[2]) for job in active.values()),
            **extra,
        })
        write_json(experiment_path, experiment)

    def launch(rep, cohort, targets):
        command = [sys.executable, str(ROOT / 'tools/run_generated_comparison.py'),
                   '--config', str(group_configs[cohort]), '--output-dir', rep['root'],
                   '--snapshot-file', str(snapshot_file),
                   '--repetition-index', str(rep['index'])]
        if experiment['snapshot_sha256']:
            command += ['--expected-snapshot-sha256', experiment['snapshot_sha256']]
        # The native launcher records its own group width (typically five).
        # Cross-group overlap is recorded separately in experiment.json.
        print(f"Repetition {rep['index']} / {experiment['k']}: {cohort}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    with ThreadPoolExecutor(max_workers=max_concurrent_groups) as executor:
        def schedule():
            while failure is None and len(active) < max_concurrent_groups:
                try:
                    rep, cohort, targets = next(pending)
                except StopIteration:
                    break
                rep['cohorts'][cohort] = 'running'
                active[executor.submit(launch, rep, cohort, targets)] = (rep, cohort, targets)
                event('started', rep, cohort, group_execution_targets=len(targets))

        schedule()
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                rep, cohort, targets = active.pop(future)
                try:
                    future.result()
                    digest = group_snapshot(Path(rep['root']), targets)
                    if experiment['snapshot_sha256'] not in (None, digest):
                        raise ValueError('Cohorts/repetitions did not share one snapshot')
                    experiment['snapshot_sha256'] = digest
                    if not completed_group(Path(rep['root']), targets, digest,
                                           judge_model=config['judge']):
                        raise ValueError('Execution/judging has not completed')
                    rep['cohorts'][cohort] = 'completed'
                    event('completed', rep, cohort)
                except Exception as exc:
                    rep['cohorts'][cohort] = 'failed'
                    event('failed', rep, cohort,
                          error={'type': type(exc).__name__, 'message': str(exc)})
                    failure = failure or exc
            # Let already-running independent groups finish and preserve their
            # results after a failure, but do not launch any new groups.
            schedule()
    if failure is not None:
        raise failure


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot-file', type=Path, required=True)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--max-concurrent-groups', type=int, default=2,
                        help='Maximum simultaneous cohort/repetition groups; each runs all configured models')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error('k must be positive')
    if args.max_concurrent_groups <= 0:
        parser.error('max-concurrent-groups must be positive')
    if args.env_file:
        load_dotenv(args.env_file)
    config = json.loads(args.config.read_text())
    cohorts = {cohort: [s for s in config['targets'] if s['cohort'] == cohort] for cohort in COHORTS}
    if not all(cohorts.values()):
        parser.error('Both existing and synthetic cohorts are required')
    if any(s['cohort'] not in COHORTS for s in config['targets']):
        parser.error('Targets must belong to the existing or synthetic cohort')
    if len({s['name'] for s in config['targets']}) != len(config['targets']):
        parser.error('Target names must be unique across cohorts')
    try:
        validate_cohort_targets(cohorts)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir.resolve()
    snapshot_file = args.snapshot_file.resolve()
    experiment_path = output / 'experiment.json'
    identity = {'k': args.k, 'config': config,
                'suite_hashes': {s['name']: suite_hash(completed_scenarios(target_suite(s))[1])
                                 for s in config['targets']}}
    if experiment_path.exists():
        if not args.resume:
            parser.error('Experiment exists; use --resume')
        experiment = json.loads(experiment_path.read_text())
        if any(experiment[key] != value for key, value in identity.items()):
            parser.error('Resume configuration differs from the saved experiment')
        if experiment.get('error'):
            experiment.setdefault('interruptions', []).append({
                'end': experiment.get('end'), 'error': experiment.pop('error')})
    else:
        experiment = {**identity, 'status': 'running', 'snapshot_sha256': None,
                      'start': datetime.now(UTC).isoformat(), 'repetitions': [
                          {'index': i, 'root': str(output / f'repetition-{i}'), 'cohorts': {}}
                          for i in range(1, args.k + 1)]}
    capacity = min(args.max_concurrent_groups, args.k * len(COHORTS))
    widths = sorted((len(cohorts[c]) for _ in range(args.k) for c in COHORTS), reverse=True)
    experiment.update(status='running', end=None, snapshot_file=str(snapshot_file),
                      max_concurrent_groups=capacity, max_execution_targets=sum(widths[:capacity]),
                      execution_policy={
                          'group_schedule': 'bounded parallel cohort/repetition groups',
                          'max_concurrent_groups': capacity,
                          'concurrency_level_scope': 'native cohort launcher group width',
                          'max_execution_targets_scope': 'configured cross-group capacity; not observed instantaneous concurrency',
                          'database_policy': 'same initial snapshot; independent namespace per model/cohort/repetition',
                      })
    write_json(experiment_path, experiment)
    try:
        if snapshot_file.exists():
            digest = snapshot_hash(snapshot_file)
            if experiment['snapshot_sha256'] not in (None, digest):
                raise ValueError('Database snapshot differs from the saved experiment; no runs were launched')
            experiment['snapshot_sha256'] = digest
        jobs = []
        # Preflight all saved state before launching any model or cloning a DB.
        for rep in experiment['repetitions']:
            for cohort, targets in cohorts.items():
                if completed_group(Path(rep['root']), targets, experiment['snapshot_sha256'],
                                   judge_model=config['judge']):
                    digest = group_snapshot(Path(rep['root']), targets)
                    if experiment['snapshot_sha256'] not in (None, digest):
                        raise ValueError('Cohorts/repetitions did not share one snapshot')
                    experiment['snapshot_sha256'] = digest
                    rep['cohorts'][cohort] = 'completed'
                else:
                    rep['cohorts'][cohort] = 'pending'
                    jobs.append((rep, cohort, targets))
        group_configs = {cohort: output / f'{cohort}-config.json' for cohort in COHORTS}
        for cohort, path in group_configs.items():
            write_json(path, {**config, 'targets': cohorts[cohort]})
        experiment.setdefault('scheduling_history', []).append({
            'event': 'scheduler_started', 'timestamp': datetime.now(UTC).isoformat(),
            'max_concurrent_groups': capacity, 'max_execution_targets': experiment['max_execution_targets'],
            'pending_groups': len(jobs),
        })
        write_json(experiment_path, experiment)
        options = dict(experiment=experiment, experiment_path=experiment_path,
                       snapshot_file=snapshot_file, group_configs=group_configs, config=config)
        if jobs and not snapshot_file.exists():
            # A single launcher owns first-time capture. Parallel launchers only
            # read the resulting snapshot, so they cannot race to overwrite it.
            execute_groups(jobs[:1], max_concurrent_groups=1, **options)
            if snapshot_hash(snapshot_file) != experiment['snapshot_sha256']:
                raise ValueError('Captured snapshot differs from the completed cohort')
            jobs = jobs[1:]
        execute_groups(jobs, max_concurrent_groups=capacity, **options)
        experiment['status'] = 'completed'
    except BaseException as exc:
        experiment.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        experiment['end'] = datetime.now(UTC).isoformat()
        write_json(experiment_path, experiment)


if __name__ == '__main__':
    main()
