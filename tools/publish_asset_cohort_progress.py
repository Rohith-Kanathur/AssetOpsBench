# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.6", "pydantic>=2.12.5,<3"]
# ///
"""Publish an explicitly incomplete cohort comparison without pretending k=3 finished."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import re

import publish_asset_cohorts as final


QUOTA = re.compile(r"hit your (?:session|usage) limit|subscription (?:limit|quota)|quota (?:exceeded|exhausted)|(?:usage|session) limit.*reset", re.I)


def quota_phase(record):
    """Only explicit backend error evidence identifies subscription interruption."""
    errors = json.dumps([record.get('error'), record.get('agent_error')])
    if QUOTA.search(errors):
        return 'execution'
    grade = record.get('grading') or {}
    judge_errors = [grade.get('error')]
    if grade.get('status') != 'completed':
        judge_errors.extend(attempt.get('error') for attempt in grade.get('attempts', []))
    if QUOTA.search(json.dumps(judge_errors)):
        return 'judge'
    return None


def collect_progress(experiment):
    # Shape/identity validation is reused, but the final publication guard remains
    # strict. This local copy cannot change the source experiment's status.
    final.validate_experiment({**experiment, 'status': 'completed'})
    records, scenario_ids, runtime = {}, {}, {}
    for spec in experiment['config']['targets']:
        rows, files = final.completed_scenarios(Path(spec['suite']))
        digest = final.suite_hash(files)
        scenario_ids[spec['name']] = {row.id for row in rows}
        if experiment.get('suite_hashes') and experiment['suite_hashes'].get(spec['name']) != digest:
            raise ValueError('Suite bytes differ from the frozen experiment')
        for rep in experiment['repetitions']:
            attempts = final.measurement_records(Path(rep['root']) / spec['name'])
            latest = final.latest_records(attempts)
            if not set(latest).issubset(scenario_ids[spec['name']]):
                raise ValueError('Progress evidence contains an unexpected scenario')
            for _, record in attempts:
                settings = record['settings']
                if (settings.get('model') != spec['model_id'] or settings.get('suite_sha256') != digest or
                    settings.get('judge_model') != experiment['config']['judge'] or
                    (settings.get('database_policy') or {}).get('snapshot_sha256') != experiment['snapshot_sha256'] or
                    settings.get('agent', spec['agent']) != spec['agent'] or
                    (spec.get('reasoning_effort') is not None and settings.get('reasoning_effort') != spec['reasoning_effort']) or
                    (settings.get('repetition_index') is not None and settings['repetition_index'] != rep['index'])):
                    raise ValueError('Progress measurements differ from the configured conditions')
                identity = {key: settings.get(key) for key in final.RUNTIME_IDENTITY_FIELDS}
                identity['agent'] = settings.get('agent', spec['agent'])
                policy = settings.get('database_policy') or {}
                identity['database_policy'] = {key: policy.get(key) for key in ('snapshot_sha256', 'reset_policy', 'database_count', 'document_count')}
                if runtime.setdefault(spec['model_key'], identity) != identity:
                    raise ValueError('Matched cohorts differ in actual runtime settings')
                if (record.get('grading') or {}).get('status') == 'completed' and record['status'] != 'completed':
                    raise ValueError('A successful judgment must belong to a completed execution')
            records[rep['index'], spec['name']] = attempts
    matched = []
    for rep in sorted(experiment['repetitions'], key=lambda r: r['index']):
        complete = True
        for spec in experiment['config']['targets']:
            latest = final.latest_records(records[rep['index'], spec['name']])
            if set(latest) != scenario_ids[spec['name']] or any(quota_phase(record) or not (
                record['status'] == 'completed' and (record.get('grading') or {}).get('status') == 'completed'
                or final.exhausted_execution(record)) for _, record in latest.values()):
                complete = False
                break
        if complete:
            matched.append(rep['index'])
    if not matched:
        raise ValueError('No common fully finished repetition exists for a comparison')
    if len(matched) == experiment['k']:
        raise ValueError('All requested repetitions are finished; use publish_asset_cohorts.py for the final report')
    groups = {spec['name']: final.aggregate(
        [(index, [record for _, record in records[index, spec['name']]]) for index in matched],
        scenario_ids[spec['name']], require_complete=True) for spec in experiment['config']['targets']}
    return matched, groups


def progress_cases(experiment, dest):
    cases, progress = [], []
    for spec in experiment['config']['targets']:
        scenarios, _ = final.completed_scenarios(Path(spec['suite']))
        for rep in sorted(experiment['repetitions'], key=lambda r: r['index']):
            target = Path(rep['root']) / spec['name']
            attempts = final.measurement_records(target)
            latest = final.latest_records(attempts)
            counts = Counter()
            for scenario in scenarios:
                path, record = latest.get(scenario.id, (None, None))
                grade = (record or {}).get('grading') or {}
                judged = grade.get('status') == 'completed'
                score = (grade.get('result') or {}).get('score') or {}
                phase = quota_phase(record) if record else None
                if judged:
                    status = 'pass' if score['passed'] else 'fail'
                elif phase:
                    status = 'quota_blocked'
                elif record and record['status'] == 'completed':
                    status = 'pending_judge'
                elif record and record['status'] == 'running':
                    status = 'running_execution'
                elif record and final.exhausted_execution(record):
                    status = 'execution_failed'
                else:
                    status = 'pending_execution'
                counts[status] += 1
                counts['executed'] += bool(record and record['status'] == 'completed')
                counts['judged'] += judged
                counts['quota_execution'] += phase == 'execution'
                counts['quota_judge'] += phase == 'judge'
                counts['judge_attempts'] += len(grade.get('attempts', []))
                counts['failed_judge_attempts'] += sum(attempt.get('status') == 'failed' for attempt in grade.get('attempts', []))
                trajectory = target / 'trajectories' / f"{record['run_id']}.json" if record else None
                answer = json.loads(trajectory.read_text()).get('answer') if trajectory and trajectory.is_file() else None
                cases.append({'target': spec['name'], 'model': final.model_name(spec), 'model_id': spec['model_id'],
                    'cohort': spec['cohort'], 'repetition': rep['index'], 'scenario_id': scenario.id,
                    'task_type': scenario.type, 'question': scenario.text, 'answer': answer,
                    'status': status, 'quota_phase': phase,
                    'passed': score.get('passed') if judged else False if status == 'execution_failed' else None,
                    'score': score.get('score') if judged else None,
                    'rubric': score.get('details') if judged else None,
                    'rationale': score.get('rationale') if judged else None,
                    'judge_attempts': grade.get('attempts', []),
                    'error': (record or {}).get('agent_error') or (record or {}).get('error') or grade.get('error'),
                    'attempt': (record or {}).get('attempt'), 'execution_ms': (record or {}).get('execution_duration_ms'),
                    'grading_ms': grade.get('duration_ms'),
                    'measurement': path.relative_to(dest).as_posix() if path else None,
                    'execution_trace': (record or {}).get('trace_file'), 'judge_trace': grade.get('trace_file'),
                    'trajectory': trajectory.relative_to(dest).as_posix() if trajectory and trajectory.is_file() else None,
                    'attempts': [{'attempt': r['attempt'], 'status': r['status'], 'quota_phase': quota_phase(r),
                                 'recovery': r.get('recovery'),
                                 'error': r.get('agent_error') or r.get('error') or (r.get('grading') or {}).get('error'),
                                 'measurement': p.relative_to(dest).as_posix(), 'trace': r.get('trace_file')}
                                for p, r in attempts if r['scenario_id'] == scenario.id]})
            progress.append({'target': spec['name'], 'model': final.model_name(spec), 'cohort': spec['cohort'],
                             'repetition': rep['index'], 'assigned': len(scenarios), 'invocation_attempts': len(attempts), **counts})
    return cases, progress


def comparison_tables(config, groups, matched):
    paper = ['| Model | Cohort | Task completion | Data retrieval accuracy | Result verification | Judged / assigned |', '|---|---|---:|---:|---:|---:|']
    hard = ['| Model | Cohort | ' + ' | '.join(f'R{i}' for i in matched) + ' | Mean ± SD | Median |',
            '|---|---|' + '---:|' * (len(matched) + 2)]
    for spec in config['targets']:
        group = groups[spec['name']]
        count = f"{group['pooled_cases']['graded']} / {group['pooled_cases']['assigned_cases']}"
        paper.append('| ' + ' | '.join([final.model_name(spec), spec['cohort'], *[
            final.spread(group['means']['rubric_success_rates'].get(key, {'mean': None, 'sd': None})) for key in final.CRITERIA], count]) + ' |')
        hard.append('| ' + ' | '.join([final.model_name(spec), spec['cohort'], *[
            final.percent(rep['cases']['pass_rate']) for rep in group['per_repetition']],
            final.spread(group['means']['pass_rate']), final.percent(group['median_pass_rate'])]) + ' |')
    return '\n'.join(paper), '\n'.join(hard)


def publish(experiment_path, cohorts, dest, env_file):
    experiment = final.load_experiment(experiment_path)
    matched, _ = collect_progress(experiment)
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    clean = final.cleaner(env_file)
    # Export all three repetitions before calculating progress counts, so those
    # counts refer to the saved evidence even while live workers are finishing.
    final.export_evidence(experiment, dest, clean)
    suites = {spec['cohort']: Path(spec['suite']) for spec in experiment['config']['targets']}
    for cohort, suite in suites.items():
        _, files = final.completed_scenarios(suite)
        for source in (suite / 'run.json', *files):
            target = dest / 'suites' / cohort / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    for filename in ('cohorts.json', 'all_existing.json', 'generation-config.json', 'synthetic-quality-audit.json', 'quota-recovery.json'):
        path = cohorts / filename
        if path.is_file():
            (dest / filename).write_text(clean(path.read_text()))
    generation_source = cohorts / 'synthetic/logs'
    if not generation_source.exists():
        generation_source = cohorts / 'generation-evidence'
    for path in generation_source.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.txt', '.md'):
            target = dest / 'generation-evidence' / path.relative_to(generation_source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(clean(path.read_text()))
    portable = deepcopy(experiment)
    for spec in portable['config']['targets']:
        spec['suite'] = f"suites/{spec['cohort']}"
    if 'suite' in portable['config']:
        portable['config']['suite'] = 'suites/existing'
    for rep in portable['repetitions']:
        rep['root'] = f"repetition-{rep['index']}"
    final.write_json(dest / 'experiment.json', json.loads(clean(json.dumps(portable))))
    published = final.load_experiment(dest / 'experiment.json')
    matched, groups = collect_progress(published)
    cases, progress = progress_cases(published, dest)
    final.export_cases(dest, cases)
    assigned = len(cases)
    judged = sum(case['score'] is not None for case in cases)
    executed = sum(row.get('executed', 0) for row in progress)
    quota = sum(case['status'] == 'quota_blocked' for case in cases)
    source_manifest = Path(experiment_path).parent / 'manifest.json'
    previous = json.loads(source_manifest.read_text()) if source_manifest.is_file() else {}
    observed_at = previous.get('observed_at') if previous.get('status') == 'incomplete' else None
    observed_at = observed_at or datetime.now(UTC).isoformat()
    final.write_json(dest / 'summary.json', {'status': 'incomplete', 'requested_k': 3,
        'observed_k': len(matched), 'matched_repetitions': matched, 'observed_at': observed_at,
        'groups': groups, 'progress': progress})
    final.write_json(dest / 'manifest.json', {'status': 'incomplete', 'requested_k': 3,
        'observed_k': len(matched), 'matched_repetitions': matched, 'assigned_trials': assigned,
        'executed_trials': executed, 'source_experiment_status': experiment['status'],
        'graded_results': judged, 'quota_blocked_cases': quota, 'observed_at': observed_at,
        'snapshot_sha256': experiment['snapshot_sha256'], 'judge': experiment['config']['judge'],
        'comparison_policy': 'Only common fully finished matched repetitions; quota interruptions are not scenario/model failures.',
        'evidence_policy': 'All three requested repetitions and every available attempt retained; missing outcomes remain unavailable.'})
    rows = final.criterion_rows(groups, {spec['name']: final.model_name(spec) + ' / ' + spec['cohort'] for spec in experiment['config']['targets']}, repeated=True)
    with (dest / 'criterion-averages.csv').open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['model_cohort', 'criterion', 'mean', 'sample_sd', 'observed_judgments', 'observed_repetitions'])
        for row in rows:
            for key, metric in row['metrics'].items():
                writer.writerow([row['model'], key, metric['mean'], metric['sd'], metric['observed_judgments'], metric['observed_repetitions']])
    info = json.loads((dest / 'cohorts.json').read_text())
    final.write_cohort_readme(dest, experiment['config'], groups, info,
                              judged=judged, assigned=assigned, matched=matched)
    final.graphs(dest, experiment['config'], groups)
    for path in dest.rglob('*.html'):
        path.unlink()
    (dest / 'checksums.sha256').write_text('\n'.join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(dest).as_posix()}"
        for path in sorted(dest.rglob('*')) if path.is_file() and path.name != 'checksums.sha256') + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--cohorts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=final.ROOT / '.env')
    args = parser.parse_args()
    publish(args.experiment, args.cohorts.resolve(), args.output_dir, args.env_file)
    print(args.output_dir / 'README.md')


if __name__ == '__main__':
    main()
