# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.6", "pydantic>=2.12.5,<3"]
# ///
"""Publish a portable existing-versus-synthetic k=3 report under benchmarks/runs."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark.criterion_report import CRITERIA, criterion_rows
from benchmark.generated_suite_runner import completed_scenarios
from benchmark.measurement import exhausted_execution, suite_hash, write_json
from benchmark.repeated_comparison import aggregate
from benchmark.live_results import NAMES
from publish_repeated_comparison import cleaner

RUNTIME_IDENTITY_FIELDS = (
    'agent', 'provider', 'harness', 'reasoning_effort', 'temperature', 'token_limit',
    'timeout_seconds', 'max_turns', 'concurrency_level', 'sdk_version', 'cli_version',
    'judge_model', 'rubric_sha256', 'grading_settings', 'invocation_retry_policy',
    'retry_policy', 'allow_same_model_judge', 'execution_order', 'token_semantics',
)


def load_experiment(path):
    """Resolve both live absolute paths and the published relative evidence paths."""
    path = Path(path).resolve()
    experiment = json.loads(path.read_text())
    for spec in experiment['config']['targets']:
        spec['suite'] = str((path.parent / spec['suite']).resolve())
    if experiment['config'].get('suite'):
        experiment['config']['suite'] = str((path.parent / experiment['config']['suite']).resolve())
    for rep in experiment['repetitions']:
        rep['root'] = str((path.parent / rep['root']).resolve())
    return experiment


def validate_experiment(experiment):
    if experiment.get('status') != 'completed' or experiment.get('k') != 3:
        raise ValueError('The requested k=3 experiment must complete before publishing final averages')
    if sorted(rep['index'] for rep in experiment['repetitions']) != [1, 2, 3]:
        raise ValueError('A k=3 report requires exactly repetitions 1, 2 and 3')
    config = experiment['config']
    if not config.get('judge') or not experiment.get('snapshot_sha256'):
        raise ValueError('Experiment must identify its judge and frozen database snapshot')
    names, pairs = set(), {}
    suites = {}
    for spec in config['targets']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', spec['name']) or spec['name'] in names:
            raise ValueError('Target names must be unique, safe directory names')
        names.add(spec['name'])
        pair = spec['model_key'], spec['cohort']
        if spec['cohort'] not in {'existing', 'synthetic'} or pair in pairs:
            raise ValueError('Each model must have one target in each cohort')
        pairs[pair] = spec
        suite = str(Path(spec['suite']).resolve())
        if suites.setdefault(spec['cohort'], suite) != suite:
            raise ValueError('All models in a cohort must execute the same frozen suite')
    keys = {key for key, _ in pairs}
    if not keys or any((key, cohort) not in pairs for key in keys for cohort in ('existing', 'synthetic')):
        raise ValueError('Each model must have one target in each cohort')
    for key in keys:
        a, b = pairs[key, 'existing'], pairs[key, 'synthetic']
        if (a['model_id'], a['agent'], a.get('reasoning_effort')) != (b['model_id'], b['agent'], b.get('reasoning_effort')):
            raise ValueError('Matched cohorts must use the same execution model, harness and configured reasoning effort')


def measurement_records(target):
    return [(path, json.loads(path.read_text()))
            for path in sorted((target / 'measurements').glob('*.json'))]


def latest_records(records):
    latest, attempts = {}, set()
    for path, record in records:
        identity = record['scenario_id'], record['attempt']
        if identity in attempts:
            raise ValueError('Duplicate scenario/attempt measurement would make outcomes ambiguous')
        attempts.add(identity)
        if record['scenario_id'] not in latest or record['attempt'] > latest[record['scenario_id']][1]['attempt']:
            latest[record['scenario_id']] = path, record
    return latest


def collect(experiment):
    validate_experiment(experiment)
    groups, runtime_identities = {}, {}
    for spec in experiment['config']['targets']:
        rows, files = completed_scenarios(Path(spec['suite']))
        digest = suite_hash(files)
        if experiment.get('suite_hashes') and experiment['suite_hashes'].get(spec['name']) != digest:
            raise ValueError('Suite bytes differ from the frozen experiment suite hash')
        repeats = []
        for rep in sorted(experiment['repetitions'], key=lambda r: r['index']):
            records = measurement_records(Path(rep['root']) / spec['name'])
            latest_records(records)
            for _, record in records:
                settings = record['settings']
                if (record.get('grading') or {}).get('status') == 'completed' and record['status'] != 'completed':
                    raise ValueError('A successful judgment must belong to a completed execution')
                if (settings.get('suite_sha256') != digest or
                        settings.get('model') != spec['model_id'] or
                        settings.get('judge_model') != experiment['config']['judge'] or
                        (settings.get('database_policy') or {}).get('snapshot_sha256') != experiment['snapshot_sha256'] or
                        settings.get('agent', spec['agent']) != spec['agent'] or
                        (spec.get('reasoning_effort') is not None and settings.get('reasoning_effort') != spec['reasoning_effort']) or
                        (settings.get('repetition_index') is not None and settings['repetition_index'] != rep['index'])):
                    raise ValueError('Suite/model/harness/reasoning/judge/repetition/snapshot differs from the experiment')
                # A cohort contrast cannot also change the actual runtime. Ignore
                # suite/namespace/repetition paths while checking native harness
                # versions, limits, judge settings and retry/resource semantics.
                identity = {key: settings.get(key) for key in RUNTIME_IDENTITY_FIELDS}
                identity['agent'] = settings.get('agent', spec['agent'])
                policy = settings.get('database_policy') or {}
                identity['database_policy'] = {key: policy.get(key) for key in
                                               ('snapshot_sha256', 'reset_policy', 'database_count', 'document_count')}
                previous = runtime_identities.setdefault(spec['model_key'], identity)
                if identity != previous:
                    differing = sorted(key for key in identity if identity[key] != previous[key])
                    raise ValueError('Matched cohorts/repetitions differ in actual runtime settings: ' + ', '.join(differing))
            repeats.append((rep['index'], [record for _, record in records]))
        groups[spec['name']] = aggregate(repeats, {r.id for r in rows}, require_complete=True)
    return groups


def percent(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def spread(metric):
    return '—' if metric['mean'] is None else percent(metric['mean']) + (
        f" ± {100 * metric['sd']:.1f} pp" if metric['sd'] is not None else '')


def numeric_spread(metric, *, divisor=1):
    if metric['mean'] is None:
        return '—'
    text = f"{metric['mean'] / divisor:,.1f}"
    return text + (f" ± {metric['sd'] / divisor:,.1f}" if metric['sd'] is not None else '')


def model_name(spec):
    name = NAMES.get(spec['model_key'], spec['model_id'])
    return name + (' (low)' if spec['model_key'] == 'glm-5-3-low' else '')


def report_tables(config, groups):
    paper = ['| Model | Cohort | Task completion | Data retrieval accuracy | Result verification | Judged / assigned |',
             '|---|---|---:|---:|---:|---:|']
    hard = ['| Model | Cohort | R1 | R2 | R3 | Mean ± SD | Median | Judged / assigned |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for spec in sorted(config['targets'], key=lambda s: (s['model_key'], s['cohort'])):
        group = groups[spec['name']]
        cases = group['pooled_cases']
        count = f"{cases['graded']} / {cases['assigned_cases']}"
        paper.append('| ' + ' | '.join([model_name(spec), spec['cohort'], *[
            spread(group['means']['rubric_success_rates'].get(key, {'mean': None, 'sd': None}))
            for key in CRITERIA], count]) + ' |')
        hard.append('| ' + ' | '.join([model_name(spec), spec['cohort'], *[
            percent(rep['cases']['pass_rate']) for rep in group['per_repetition']],
            spread(group['means']['pass_rate']), percent(group['median_pass_rate']), count]) + ' |')
    return '\n'.join(paper), '\n'.join(hard)


def compact_results_table(config, groups):
    lines = ['| Model | Existing (%) | Synthetic (%) |', '|---|---:|---:|']
    specs = {(spec['model_key'], spec['cohort']): spec for spec in config['targets']}
    for key in dict.fromkeys(spec['model_key'] for spec in config['targets']):
        pair = [specs[key, cohort] for cohort in ('existing', 'synthetic')]
        lines.append('| ' + ' | '.join([model_name(pair[0]), *[
            numeric_spread(groups[spec['name']]['means']['pass_rate'], divisor=.01)
            for spec in pair]]) + ' |')
    return '\n'.join(lines)


def write_cohort_readme(dest, config, groups, info, *, judged, assigned, matched=None):
    model_count = len(config['targets']) // 2
    repetitions = len(matched) if matched is not None else 3
    if matched is not None:
        status = (f'**In progress:** {judged}/{assigned} trials graded. Comparison uses {repetitions} completed '
                  'repetitions. R3 is paused by Claude quota.')
        planned = '3 repetitions planned'
        script = 'publish_asset_cohort_progress.py'
    else:
        status = f'**Complete:** {judged}/{assigned} trials graded.'
        planned = '3 repetitions'
        script = 'publish_asset_cohorts.py'
    report_path = dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else '<report-directory>'
    quality = (' Sensor and rubric gaps are recorded in the [quality audit](synthetic-quality-audit.json).'
               if (dest / 'synthetic-quality-audit.json').is_file() else '')
    (dest / 'README.md').write_text(f"""# {config['asset_class']} · existing vs. synthetic

{info['selected_count']} existing + {info['selected_count']} synthetic scenarios · {model_count} models · {planned} · Fable 5.1 judge.

{status}

## Criterion scores

![Completion, retrieval accuracy and verification](graphs/criterion-averages.png)

## Strict pass rates

![Existing and synthetic pass rates](graphs/pass-rate.png)

{compact_results_table(config, groups)}

Mean ± sample SD across {repetitions} repetitions. A strict pass requires all five positive criteria and no hallucinations.

## Method and run data

Synthetic scenarios were generated once using local asset data. Existing scenarios remain in the few-shot context. Each model and repetition starts from the same database snapshot in an isolated namespace; state persists between tasks. Judging uses a separate Fable session per execution and an 8,000-character trajectory limit.

FMSR's Watsonx backend was unavailable.{quality}

[Results CSV](cases.csv) · [Summary](summary.json)

Rebuild with local execution evidence:

```bash
uv run tools/{script} \\
  --experiment {report_path}/experiment.json \\
  --cohorts {report_path} --output-dir {report_path}
```
""")


def resource_table(config, groups):
    lines = ['| Model | Cohort | Median execution ± SD (s) | Mean tools ± SD | Input tokens / repetition ± SD | Output tokens / repetition ± SD | Judged R1/R2/R3 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for spec in sorted(config['targets'], key=lambda s: (s['model_key'], s['cohort'])):
        group = groups[spec['name']]
        means = group['means']
        lines.append('| ' + ' | '.join([model_name(spec), spec['cohort'],
                     numeric_spread(means['median_execution_ms'], divisor=1000),
                     numeric_spread(means['tool_call_count']['mean']),
                     numeric_spread(means['input_tokens']['total']),
                     numeric_spread(means['output_tokens']['total']),
                     '/'.join(str(rep['cases']['graded']) for rep in group['per_repetition'])]) + ' |')
    return '\n'.join(lines)


def resolve_trace(reference, target, report_root):
    path = Path(reference)
    if path.is_absolute():
        return path
    # Published records reference the report root; original local records use
    # absolute paths, but target-relative trace paths are accepted as well.
    for base in (report_root, target):
        if (base / path).is_file():
            return base / path
    return report_root / path


def export_evidence(experiment, dest, clean):
    """Copy all attempts and validate successful, distinct judge sessions.

    Early failed invocations can exit before emitting a trace. Their measurements
    and logs are retained, with trace availability explicit instead of a dead link.
    """
    sessions, latest_judged = [], 0
    for rep in experiment['repetitions']:
        root = Path(rep['root'])
        report_root = root.parent
        for spec in experiment['config']['targets']:
            source = root / spec['name']
            relative_target = Path(f"repetition-{rep['index']}") / spec['name']
            target = dest / relative_target
            records = measurement_records(source)
            latest = latest_records(records)
            refs, sources = {}, {}

            def register(reference, *, required=False):
                if not reference:
                    if required:
                        raise ValueError('Missing successful execution/judge trace reference')
                    return None
                path = resolve_trace(reference, source, report_root)
                if not path.is_file():
                    if required:
                        raise ValueError(f'Missing successful execution/judge trace: {path.name}')
                    return None
                relative = (relative_target / 'events' / (path.name if path.suffix == '.gz' else path.name + '.gz')).as_posix()
                if relative in sources and sources[relative].resolve() != path.resolve():
                    raise ValueError('Trace filenames collide in the portable evidence directory')
                sources[relative] = path
                refs[str(reference)] = relative
                refs[str(path.resolve())] = relative
                return relative

            for path in sorted((source / 'traces').glob('*.jsonl')):
                register(str(path.resolve()))
            for _, record in records:
                register(record.get('trace_file'), required=record['status'] == 'completed')
                grade = record.get('grading') or {}
                register(grade.get('trace_file'), required=grade.get('status') == 'completed')
                if grade.get('status') == 'completed':
                    if grade.get('model', experiment['config']['judge']) != experiment['config']['judge'] or grade.get('separate_session') is False:
                        raise ValueError('Successful grading must use the configured independent judge')
                    trace = resolve_trace(grade['trace_file'], source, report_root)
                    raw = gzip.decompress(trace.read_bytes()).decode() if trace.suffix == '.gz' else trace.read_text()
                    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    session = [(event.get('payload') or {}).get('session_id') for event in events if event.get('kind') == 'judge_result']
                    if len(session) != 1 or not isinstance(session[0], str) or not session[0]:
                        raise ValueError('Missing successful judge session identity')
                    sessions.extend(session)
                for attempt in grade.get('attempts', []):
                    register(attempt.get('trace_file'), required=attempt.get('status') == 'completed')
            latest_judged += sum((record.get('grading') or {}).get('status') == 'completed' for _, record in latest.values())
            # Raw *.log files are ignored by this repository. Publish text
            # suffixes so a normal git add retains the full invocation evidence.
            logs = sorted(source.glob('*.log')) + sorted(source.glob('*.log.txt'))
            for path in logs:
                name = path.name if path.name.endswith('.log.txt') else path.name + '.txt'
                refs[str(path.resolve())] = (relative_target / name).as_posix()

            def portable(text):
                for old, new in sorted(refs.items(), key=lambda pair: -len(pair[0])):
                    text = text.replace(old, new)
                for old, new in sorted(((spec['suite'], f"suites/{spec['cohort']}"),
                                         (str(source.resolve()), relative_target.as_posix())), key=lambda pair: -len(pair[0])):
                    text = text.replace(old, new)
                return clean(text)

            for relative, path in sources.items():
                output = dest / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                raw = gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()
                output.write_bytes(gzip.compress(portable(raw).encode(), mtime=0))

            def trace_availability(record):
                original = record.get('trace_file')
                available = register(original) if original else None
                record['trace_file_available'] = bool(available)
                record['trace_file'] = available
                if original and not available:
                    record['missing_trace_file'] = Path(original).name

            for path, record in records:
                record = deepcopy(record)
                trace_availability(record)
                grade = record.get('grading')
                if grade:
                    trace_availability(grade)
                    for attempt in grade.get('attempts', []):
                        trace_availability(attempt)
                write_json(target / 'measurements' / path.name, json.loads(portable(json.dumps(record))))
            for folder in ('trajectories', 'failed_trajectories', 'database_audit'):
                for path in sorted((source / folder).glob('*')):
                    if path.is_file():
                        output = target / folder / path.name
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(portable(path.read_text()))
            for path in logs:
                name = path.name if path.name.endswith('.log.txt') else path.name + '.txt'
                output = target / name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(portable(path.read_text()))
            for filename in ('settings.json', 'environment.json', 'target.json'):
                output = target / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(portable((source / filename).read_text()))
    if len(sessions) != len(set(sessions)):
        raise ValueError('Judge sessions were reused across cases')
    return latest_judged, len(sessions)


def case_rows(experiment, dest):
    """Build inspectable final cases without filling absent judgments with false."""
    cases = []
    for spec in experiment['config']['targets']:
        scenarios, _ = completed_scenarios(Path(spec['suite']))
        lookup = {scenario.id: scenario for scenario in scenarios}
        for rep in sorted(experiment['repetitions'], key=lambda r: r['index']):
            relative_target = Path(f"repetition-{rep['index']}") / spec['name']
            target = dest / relative_target
            records = measurement_records(target)
            for sid, (path, record) in latest_records(records).items():
                scenario = lookup[sid]
                grade = record.get('grading') or {}
                judged = grade.get('status') == 'completed'
                score = (grade.get('result') or {}).get('score') or {}
                trajectory_path = target / 'trajectories' / f"{record['run_id']}.json"
                trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else {}
                attempts = [{'attempt': r['attempt'], 'status': r['status'], 'error': r.get('error'),
                             'measurement': p.relative_to(dest).as_posix(), 'trace': r.get('trace_file')}
                            for p, r in records if r['scenario_id'] == sid]
                cases.append({'target': spec['name'], 'model': model_name(spec), 'model_id': spec['model_id'],
                              'cohort': spec['cohort'], 'repetition': rep['index'], 'scenario_id': sid,
                              'task_type': scenario.type, 'question': scenario.text,
                              'status': ('pass' if score['passed'] else 'fail') if judged else 'execution_failed',
                              'passed': score.get('passed') if judged else False if exhausted_execution(record) else None,
                              'score': score.get('score') if judged else None,
                              'rubric': score.get('details') if judged else None,
                              'rationale': score.get('rationale') if judged else None,
                              'answer': trajectory.get('answer'), 'attempt': record['attempt'],
                              'execution_ms': record.get('execution_duration_ms'),
                              'grading_ms': grade.get('duration_ms'), 'metrics': record.get('metrics'),
                              'measurement': path.relative_to(dest).as_posix(),
                              'execution_trace': record.get('trace_file'), 'judge_trace': grade.get('trace_file'),
                              'trajectory': trajectory_path.relative_to(dest).as_posix() if trajectory_path.is_file() else None,
                              'attempts': sorted(attempts, key=lambda r: r['attempt'])})
    return cases


def export_cases(dest, cases):
    fields = ['model', 'model_id', 'target', 'cohort', 'repetition', 'scenario_id', 'task_type',
              'question', 'status', 'passed', 'score', *CRITERIA, 'attempt', 'execution_ms', 'grading_ms',
              'measurement', 'trajectory', 'execution_trace', 'judge_trace']
    with (dest / 'cases.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case.get(key) for key in fields if key not in CRITERIA} |
                            {key: (case['rubric'] or {}).get(key) for key in CRITERIA})
    write_json(dest / 'cases.json', cases)


def cohort_suite_dir(cohorts, cohort):
    """A published report is also a complete input for a model-free rebuild."""
    local = cohorts / cohort
    return local if (local / 'run.json').is_file() else cohorts / 'suites' / cohort


def graphs(dest, config, groups):
    from benchmark.paper_plots import PAPER_COLORS, criterion_figure, save_figure, strict_pass_figure
    specs = {(s['model_key'], s['cohort']): s for s in config['targets']}
    keys = list(dict.fromkeys(s['model_key'] for s in config['targets']))
    labels = [model_name(specs[key, 'existing']) for key in keys]
    graph_dir = dest / 'graphs'
    graph_dir.mkdir(exist_ok=True)
    panels, strict = [], []
    for cohort in ('existing', 'synthetic'):
        targets = [specs[key, cohort] for key in keys]
        rows = criterion_rows(groups, {spec['name']: model_name(spec) for spec in targets}, repeated=True)
        panels.append((cohort.title(), rows))
        strict.append((cohort.title(), [groups[spec['name']]['means']['pass_rate'] for spec in targets]))
    colors = PAPER_COLORS[:len(labels)]
    observed_k = next(iter(groups.values()))['k']
    caption = f"{config['asset_class']} · observed k={observed_k}" + (' · requested k=3 INCOMPLETE' if observed_k < 3 else '')
    save_figure(criterion_figure(panels, labels, colors, caption=caption), graph_dir, 'criterion-averages')
    save_figure(strict_pass_figure(strict, labels, colors, caption=caption), graph_dir, 'pass-rate')
    # Remove the superseded four-panel comparison so the report has two figures.
    for suffix in ('png', 'svg'):
        (graph_dir / f'cohort-comparison.{suffix}').unlink(missing_ok=True)


def publish(experiment_path, cohorts, dest, env_file):
    experiment = load_experiment(experiment_path)
    groups = collect(experiment)  # Refuse partial or inconsistent averages before writing.
    config = experiment['config']
    info = json.loads((cohorts / 'cohorts.json').read_text())
    suites = {spec['cohort']: Path(spec['suite']) for spec in config['targets']}
    for cohort, suite in suites.items():
        local_rows, files = completed_scenarios(cohort_suite_dir(cohorts, cohort))
        rows, executed_files = completed_scenarios(suite)
        if suite_hash(files) != suite_hash(executed_files):
            raise ValueError('Prepared cohort bytes differ from the executed suite')
        if len(rows) != info['selected_count'] or len(local_rows) != len(rows):
            raise ValueError('Cohort counts differ from the selected existing/synthetic plan')
    if Counter(row.type for row in completed_scenarios(suites['existing'])[0]) != Counter(row.type for row in completed_scenarios(suites['synthetic'])[0]):
        raise ValueError('Existing and synthetic cohorts must match task-type counts')
    dest = Path(dest).resolve()
    # Never overwrite source evidence while exporting it.
    if any(Path(rep['root']).resolve().is_relative_to(dest) and
           Path(rep['root']).resolve() != dest / f"repetition-{rep['index']}"
           for rep in experiment['repetitions']):
        raise ValueError('Report destination cannot contain its source repetition directories')
    dest.mkdir(parents=True, exist_ok=True)
    clean = cleaner(env_file)
    judged, all_judgments = export_evidence(experiment, dest, clean)
    assigned = sum(g['pooled_cases']['assigned_cases'] for g in groups.values())
    failures = sum(g['pooled_cases']['execution_failed_cases'] for g in groups.values())
    if judged != sum(g['pooled_cases']['graded'] for g in groups.values()):
        raise ValueError('Successful judge session count does not match the published judgments')
    for cohort, suite in suites.items():
        _, files = completed_scenarios(suite)
        for source in (suite / 'run.json', *files):
            target = dest / 'suites' / cohort / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    for filename in ('cohorts.json', 'all_existing.json', 'generation-config.json'):
        (dest / filename).write_text(clean((cohorts / filename).read_text()))
    quality_audit = cohorts / 'synthetic-quality-audit.json'
    if quality_audit.is_file():
        (dest / quality_audit.name).write_text(clean(quality_audit.read_text()))
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
    write_json(dest / 'experiment.json', json.loads(clean(json.dumps(portable))))
    published = load_experiment(dest / 'experiment.json')
    # Recompute from exported bytes to prove the report is portable and unchanged.
    if collect(published) != groups:
        raise ValueError('Exported evidence changed the published aggregates')
    write_json(dest / 'summary.json', groups)
    write_json(dest / 'manifest.json', {'k': 3, 'assigned_trials': assigned, 'graded_results': judged,
                                      'retained_successful_judge_sessions': all_judgments,
                                      'execution_failed_cases': failures,
                                      'snapshot_sha256': experiment['snapshot_sha256'],
                                      'suite_hashes': experiment.get('suite_hashes'),
                                      'judge': config['judge'], 'asset_class': config['asset_class'],
                                      'model_count': len(config['targets']) // 2,
                                      'redaction_policy': 'Credentials and local path prefixes removed; exact scenario suite bytes retained.',
                                      'evidence_policy': 'All measurements, logs, saved trajectories, database audits and observed compressed traces; missing failed-invocation traces explicitly marked.'})
    cases = case_rows(published, dest)
    export_cases(dest, cases)
    names = {s['name']: f"{model_name(s)} / {s['cohort']}" for s in config['targets']}
    criteria = criterion_rows(groups, names, repeated=True)
    with (dest / 'criterion-averages.csv').open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['model_cohort', 'target', 'criterion', 'mean', 'sample_sd', 'observed_judgments', 'observed_repetitions', 'judged_trials', 'assigned_trials'])
        for row in criteria:
            for key, metric in row['metrics'].items():
                writer.writerow([row['model'], row['target'], key, metric['mean'], metric['sd'], metric['observed_judgments'], metric['observed_repetitions'], row['judged_trials'], row['assigned_trials']])
    write_cohort_readme(dest, config, groups, info, judged=judged, assigned=assigned)
    graphs(dest, config, groups)
    for path in dest.rglob('*.html'):
        path.unlink()
    checksum_lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(dest).as_posix()}"
                      for path in sorted(dest.rglob('*')) if path.is_file() and path.name != 'checksums.sha256']
    (dest / 'checksums.sha256').write_text('\n'.join(checksum_lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--cohorts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    publish(args.experiment, args.cohorts.resolve(), args.output_dir, args.env_file)
    print(args.output_dir / 'README.md')


if __name__ == '__main__':
    main()
