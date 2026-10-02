# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6"]
# ///
"""Publish portable evidence, repetition means/variation, graphs and a README."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark.measurement import suite_hash, write_json
from benchmark.repeated_comparison import aggregate, load_experiment, mean_sd, snapshot
from benchmark.criterion_report import criterion_rows, publish_criterion_averages

COLORS = ['#42684f', '#60816a', '#90a98c', '#b2bc9c', '#d4b477']
RUBRICS = {'task_completion': 'Completion', 'data_retrieval_accuracy': 'Accuracy',
           'generalized_result_verification': 'Verification', 'agent_sequence_correct': 'Sequence',
           'clarity_and_justification': 'Clarity', 'hallucinations': 'No hallucinations'}


def cleaner(env_file):
    secrets = []
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)", line)
            if match and re.search(r'KEY|TOKEN|SECRET|PASSWORD', match[1], re.I) and not match[1].endswith(('URL', 'PATH', 'FILE')):
                value = match[2].strip().strip("\"'")
                if len(value) >= 8 and value not in {'password', 'PASSWORD', match[1]} and not value.startswith('${'):
                    secrets.append(value)

    def clean(text):
        for value in secrets: text = text.replace(value, '[REDACTED]')
        text = re.sub(r'(https?://)[^/\s:@]+:[^/\s@]+@', r'\1[REDACTED]@', text)
        return text.replace(str(ROOT), '<workspace>').replace(str(Path.home()), '<home>')
    return clean


def export_repetition(source, dest, suite, clean):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    output = dest / 'models'
    if source == output:
        return
    if output.is_relative_to(source):
        raise ValueError('A repetition cannot be exported inside its source model directory')
    # Already-published evidence uses paths relative to the repetition folder.
    # Copy its bytes exactly; moving R1 must not rewrite measurements or traces.
    records = [json.loads(path.read_text()) for path in source.glob('*/measurements/*.json')]
    references = [reference for record in records
                  for reference in [record.get('trace_file'),
                                    (record.get('grading') or {}).get('trace_file'),
                                    *[attempt.get('trace_file') for attempt in
                                      (record.get('grading') or {}).get('attempts', [])]]
                  if reference]
    if records and references and all(not Path(reference).is_absolute() and
                                     (source.parent / reference).is_file() for reference in references):
        shutil.copytree(source, output, dirs_exist_ok=True)
        for path in source.rglob('*'):
            if path.is_file() and path.read_bytes() != (output / path.relative_to(source)).read_bytes():
                raise ValueError('Published repetition evidence changed during copying')
        return
    config = json.loads((ROOT / 'benchmarks/generated-comparison.json').read_text())
    paths = {}
    for spec in config['targets']:
        target = source / spec['name']
        for path in (target / 'measurements').glob('*.json'):
            r = json.loads(path.read_text())
            grade = r.get('grading') or {}
            refs = [r['trace_file'], *[a['trace_file'] for a in grade.get('attempts', [])]]
            if grade.get('trace_file'): refs.append(grade['trace_file'])
            for ref in refs:
                if not Path(ref).is_file(): raise ValueError(f'Missing trace {Path(ref).name}')
                paths[ref] = f"models/{spec['name']}/events/{Path(ref).name}.gz"
    def portable(text):
        for old, new in sorted(paths.items(), key=lambda pair: -len(pair[0])):
            text = text.replace(old, new)
        text = text.replace(str(source), 'models').replace(str(suite), '../suite')
        return clean(text)
    for ref, relative in paths.items():
        path = dest / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(portable(Path(ref).read_text()).encode(), mtime=0))
    for spec in config['targets']:
        target = source / spec['name']; output = dest / 'models' / spec['name']
        for name in ('target.json', 'settings.json', 'environment.json'):
            write_json(output / name, json.loads(portable((target / name).read_text())))
        for path in (target / 'measurements').glob('*.json'):
            record=json.loads(portable(path.read_text()))
            if 'agent_error' not in record:
                raw=json.loads(path.read_text())
                events=[json.loads(line) for line in Path(raw['trace_file']).read_text().splitlines()]
                record['agent_error']=next((e.get('error') for e in reversed(events) if e['kind']=='run_error'),None)
                record=json.loads(clean(json.dumps(record)))
            write_json(output / 'measurements' / path.name, record)
        for path in (target / 'trajectories').glob('*.json'):
            write_json(output / 'trajectories' / path.name, json.loads(portable(path.read_text())))


def rows_for(experiment, config):
    groups = {}
    ids = set()
    suite=Path(experiment['suite'])
    digest=suite_hash([suite/name for name in ('scenarios.json','negative_scenarios.json')])
    for name in ('scenarios.json', 'negative_scenarios.json'):
        ids.update(str(s['id']) for s in json.loads((Path(experiment['suite']) / name).read_text()))
    for spec in config['targets']:
        repeats = []
        for rep in experiment['repetitions']:
            records = [json.loads(p.read_text()) for p in (Path(rep['root']) / spec['name'] / 'measurements').glob('*.json')]
            if any(r['settings']['suite_sha256'] != digest for r in records):
                raise ValueError('Published suite bytes differ from the executed suite hash')
            repeats.append((rep['index'], records))
        groups[spec['name']] = aggregate(repeats, ids, require_complete=True)
    return groups


def publish(experiment_path, dest, env_file):
    config = json.loads((ROOT / 'benchmarks/generated-comparison.json').read_text())
    experiment = load_experiment(experiment_path)
    groups = rows_for(experiment, config)  # Validate all results before writing a final average.
    clean = cleaner(env_file)
    dest.mkdir(parents=True, exist_ok=True)
    portable = {**experiment, 'suite': 'suite'}
    portable['baseline'] = 'repetition-1/models'
    portable['repetitions'] = []
    for rep in experiment['repetitions']:
        export_repetition(Path(rep['root']), dest / f"repetition-{rep['index']}", Path(experiment['suite']), clean)
        relative = f"repetition-{rep['index']}/models"
        portable['repetitions'].append({**rep, 'root': relative})
    # Retain exact suite bytes and its research/generation evidence.
    source_suite = Path(experiment['suite']).resolve()
    output_suite = dest / 'suite'
    if source_suite != output_suite.resolve():
        shutil.copytree(source_suite, output_suite, dirs_exist_ok=True)
    # Original R1 metadata says 'suite'; preserve it byte-for-byte and make
    # that relative reference resolve inside the relocated repetition folder.
    r1_suite = dest / 'repetition-1' / 'suite'
    if not r1_suite.exists():
        r1_suite.symlink_to('../suite', target_is_directory=True)
    write_json(dest / 'experiment.json', json.loads(clean(json.dumps(portable))))
    published = load_experiment(dest / 'experiment.json')
    groups = rows_for(published, config)
    payload = snapshot(published, config)
    sessions=[]
    for model in payload['models']:
        for row in model['rows']:
            if (row['record'].get('grading') or {}).get('status')!='completed':continue
            rep=next(rep for rep in published['repetitions'] if rep['index']==row['repetition'])
            path=Path(rep['root']).parent/row['record']['grading']['trace_file']
            raw=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
            events=[json.loads(line) for line in raw.splitlines()]
            sessions.extend(e['payload']['session_id'] for e in events if e['kind']=='judge_result')
    judged=sum(m['summary']['graded'] for m in payload['models'])
    if len(sessions)!=judged or len(set(sessions))!=judged or not all(sessions):
        raise ValueError('Successful judge sessions are missing or not distinct')
    for model in payload['models']:
        for row in model['rows']:
            r = row.get('record')
            if not r: continue
            rep = next(rep for rep in published['repetitions'] if rep['index'] == row['repetition'])
            def reference(ref):
                return os.path.relpath(Path(rep['root']).parent / ref, dest)
            r['trace_file'] = reference(r['trace_file'])
            grade = r.get('grading') or {}
            if grade.get('trace_file'): grade['trace_file'] = reference(grade['trace_file'])
            for attempt in grade.get('attempts', []): attempt['trace_file'] = reference(attempt['trace_file'])
            # Avoid duplicating large payloads already present in the row and linked full trace.
            r.pop('metrics', None)
            if grade.get('result'): grade['result'] = {'score': grade['result']['score']}
            row['grade'] = {'score': row['grade']['score']} if row.get('grade') else None
    for path in dest.rglob('comparison.html'):
        path.unlink()
    write_json(dest / 'summary.json', groups)
    existing = dest / 'manifest.json'
    write_json(existing, {'published_at': json.loads(existing.read_text())['published_at'] if existing.exists() else datetime.now(timezone.utc).isoformat(),
                             'implementation_commit_at_publication': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                             'k': 3, 'scenario_count': 52, 'model_count': 5,'assigned_trials':780,
                             'graded_results':judged,
                             'execution_failed_cases':sum(m['summary']['execution_failed_cases'] for m in payload['models']),
                             'evidence_policy': 'All three repetitions include exact portable measurement-ledgers and measurement-linked compressed full traces; repetition 1 is retained within this report.',
                         'redaction_policy': 'Credentials and local path prefixes removed; metrics and timestamps unchanged.'})
    make_report(dest, published, config, groups, payload)


def write_readme(dest, keys, labels, groups):
    first = groups[keys[0]]
    repetitions = len(first['per_repetition'])
    scenario_count = len(first['per_scenario'])
    judged = sum(group['pooled_cases']['graded'] for group in groups.values())
    assigned = sum(group['pooled_cases']['assigned_cases'] for group in groups.values())
    failed = sum(group['pooled_cases']['execution_failed_cases'] for group in groups.values())
    failure_note = (f" {failed} execution{'s' if failed != 1 else ''} failed." if failed else '')
    columns = ' | '.join(f"R{rep['index']}" for rep in first['per_repetition'])
    table = [f'| Model | {columns} | Mean ± SD (%) |',
             '|---|' + '---:|' * (repetitions + 1)]
    for key, label in zip(keys, labels):
        group = groups[key]
        rates = [f"{rep['cases']['pass_rate']:.1%}" for rep in group['per_repetition']]
        mean = group['means']['pass_rate']
        average = f"{mean['mean'] * 100:.1f}"
        if mean['sd'] is not None:
            average += f" ± {mean['sd'] * 100:.1f}"
        table.append('| ' + ' | '.join([label, *rates, average]) + ' |')
    report_path = dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else '<report-directory>'
    (dest / 'README.md').write_text(f"""# Transformer · {repetitions} repetitions

{scenario_count} synthetic scenarios · {len(keys)} models · {repetitions} repetitions · Fable 5.1 judge.

{judged}/{assigned} trials graded.{failure_note}

## Criterion scores

![Completion, retrieval accuracy and verification](graphs/criterion-averages.png)

## Strict pass rates

![Transformer pass rates](graphs/pass-rate.png)

{chr(10).join(table)}

Mean ± sample SD across {repetitions} repetitions. A strict pass requires all five positive criteria and no hallucinations.

## Method and run data

Opus 5.5 generated the scenarios once. Each model and repetition starts from the same database snapshot in an isolated namespace; state persists between tasks. Judging uses a separate Fable session per execution and an 8,000-character trajectory limit.

FMSR's Watsonx backend was unavailable. Failed executions count as nonpassing; unavailable judgments are excluded from criterion scores.

[Results CSV](cases.csv) · [Summary](summary.json)

Rebuild with local execution evidence:

```bash
uv run tools/publish_repeated_comparison.py \\
  --experiment {report_path}/experiment.json \\
  --output-dir {report_path}
```
""")


def make_report(dest, experiment, config, groups, payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from benchmark.paper_plots import PAPER_COLORS, save_figure, strict_pass_figure
    COLORS = PAPER_COLORS
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#d8dfd9', 'text.color': '#24382a',
                         'axes.labelcolor': '#526356', 'xtick.color': '#526356', 'ytick.color': '#526356',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'svg.hashsalt': 'transformer-k3'})
    keys = [s['name'] for s in config['targets']]
    labels = [m['name'] + (' (low)' if m['key'] == 'glm-5-3-low' else '') for m in payload['models']]
    graphs = dest / 'graphs'; graphs.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(graphs / f'{name}.png', dpi=180, bbox_inches='tight')
        svg = graphs / f'{name}.svg'
        fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(fig)
    publish_criterion_averages(
        dest, criterion_rows(groups, dict(zip(keys, labels)), repeated=True), COLORS, repeated=True)
    fig = strict_pass_figure([('Transformer', [groups[key]['means']['pass_rate'] for key in keys])], labels, COLORS)
    save_figure(fig, graphs, 'pass-rate')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    short_labels = [label.replace('GPT-6 Astra', 'GPT-6\nAstra').replace('GPT-6.1 Sol', 'GPT-6.1\nSol') for label in labels]
    for ax, metric, title in zip(axes, ['median_execution_ms', 'p95_execution_ms'], ['Median execution · mean ± SD', 'p95 execution · mean ± SD']):
        for i, (key, color) in enumerate(zip(keys, COLORS)):
            avg = groups[key]['means'][metric]
            if avg['mean'] is None: continue
            ax.errorbar(i, avg['mean'] / 1000, yerr=avg['sd'] / 1000 if avg['sd'] is not None else None, fmt='o', color=color, capsize=4)
            for j, rep in enumerate(groups[key]['per_repetition']):
                if rep['cases'][metric] is not None: ax.scatter(i + (j - 1) * .08, rep['cases'][metric] / 1000, color=color, s=14, alpha=.4)
        ax.set_xticks(range(len(keys)), short_labels, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Entire agent invocation (seconds)')
        ax.set_title(title, loc='left', pad=14, fontweight='bold')
    save(fig, 'execution-time')

    fig, ax = plt.subplots(figsize=(10, 3.3), layout='constrained')
    matrix = np.array([[groups[k]['means']['rubric_success_rates'][r]['mean'] * 100 for r in RUBRICS] for k in keys])
    ax.imshow(matrix, vmin=0, vmax=100, cmap=LinearSegmentedColormap.from_list('success', ['#faf9f2', '#cfdbc9', '#42684f']), aspect='auto')
    ax.set_xticks(range(6), list(RUBRICS.values()), fontsize=9); ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(6):
            sd = groups[keys[i]]['means']['rubric_success_rates'][list(RUBRICS)[j]]['sd'] * 100
            ax.text(j, i, f'{matrix[i,j]:.0f}% ± {sd:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if matrix[i,j] > 75 else '#24382a')
    ax.set_title('Rubric success · mean ± sample SD across three repetitions', loc='left', pad=14, fontweight='bold')
    save(fig, 'rubric-success')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout='constrained')
    for ax, metric, divisor, title in zip(axes, ['input_tokens', 'output_tokens', 'tool_call_count'], [1000, 1000, 1],
                                         ['Mean input · k tokens', 'Mean output · k tokens', 'Mean tool calls']):
        for i, (key, color) in enumerate(zip(keys, COLORS)):
            avg = groups[key]['means'][metric]['mean']
            if avg['mean'] is not None:
                ax.bar(i, avg['mean'] / divisor, width=.6, color=color, edgecolor='#303030', linewidth=.65)
                if avg['sd'] is not None: ax.errorbar(i, avg['mean'] / divisor, yerr=avg['sd'] / divisor, fmt='none', ecolor='#303030', capsize=3)
        ax.set_xticks(range(len(keys)), short_labels, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc='left', pad=14, fontweight='bold')
    save(fig, 'resources')

    fig, ax = plt.subplots(figsize=(10, 12), layout='constrained')
    scenarios = payload['models'][0]['repeated']['per_scenario']
    ids = list(scenarios)
    data = [[groups[k]['per_scenario'][sid]['pass_fraction'] for k in keys] for sid in ids]
    ax.imshow(data, vmin=0, vmax=1, cmap=LinearSegmentedColormap.from_list('frequency', ['#faf9f2', '#42684f']), aspect='auto')
    ax.set_xticks(range(5), labels, fontsize=9); ax.set_yticks(range(len(ids)), [sid.replace('transformer_', '') for sid in ids], fontsize=8)
    for i, sid in enumerate(ids):
        for j, key in enumerate(keys):
            n = sum(o['passed'] for o in groups[key]['per_scenario'][sid]['outcomes'])
            ax.text(j, i, f'{n}/3', ha='center', va='center', fontsize=8, color='white' if n >= 2 else '#24382a')
    ax.set_title('Scenario repeatability · successful repetitions / 3', loc='left', pad=14, fontweight='bold')
    save(fig, 'scenario-repeatability')

    columns = ['model', 'repetition', 'scenario_id', 'attempt','status','grading_status', 'passed', 'score', 'execution_duration_ms', 'grading_duration_ms',
               'input_tokens', 'output_tokens', 'reasoning_tokens', 'cache_read_tokens', 'cache_write_tokens',
               'tool_call_count', 'tool_errors', 'database_writes_attempted', 'database_writes_succeeded', *RUBRICS, 'judge_rationale']
    with (dest / 'cases.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator='\n'); writer.writeheader()
        for model in payload['models']:
            for row in model['rows']:
                r = row['record']; score = (row.get('grade') or {}).get('score')
                values = {k: r.get(k, (row['metrics'] or {}).get(k)) for k in columns}
                values.update(model=model['name'], repetition=row['repetition'], scenario_id=row['scenario_id'],
                              grading_status=(r.get('grading') or {}).get('status'),
                              passed=score['passed'] if score else False, score=score['score'] if score else None,
                              grading_duration_ms=row['grading_ms'],
                              judge_rationale=score['details'].get('suggestions', score['rationale']) if score else None)
                values.update({k: score['details'].get(k) if score else None for k in RUBRICS}); writer.writerow(values)
    mean_columns=['model','scenario_id','pass_fraction','observed_judgments','observed_outcomes',
                  *[f'{key}_{stat}' for key in ('score','execution_duration_ms','grading_duration_ms',
                                               'tool_call_count','input_tokens','output_tokens') for stat in ('mean','sd','observed_repetitions')]]
    with (dest/'scenario-averages.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=mean_columns,lineterminator='\n');writer.writeheader()
        for key,label in zip(keys,labels):
            for sid,result in groups[key]['per_scenario'].items():
                values={'model':label,'scenario_id':sid,'pass_fraction':result['pass_fraction'],'observed_judgments':result['observed_judgments'],
                        'observed_outcomes':result['observed_outcomes']}
                for metric in ('score','execution_duration_ms','grading_duration_ms','tool_call_count','input_tokens','output_tokens'):
                    values.update({f'{metric}_{stat}':value for stat,value in result[metric].items()})
                writer.writerow(values)

    write_readme(dest, keys, labels, groups)
    attempts = sum(group['pooled_attempts']['attempted'] for group in groups.values())
    judged = sum(group['pooled_cases']['graded'] for group in groups.values())
    failed = sum(group['pooled_cases']['execution_failed_cases'] for group in groups.values())
    files = sorted(p for p in dest.rglob('*') if p.is_file() and p.name != 'checksums.sha256')
    (dest / 'checksums.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n' for p in files))
    print(f'Published k=3: {judged} judgments, {failed} terminal execution failures, {attempts} retained attempts.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmarks/runs/2026-09-30-transformer-k3')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    publish(args.experiment.resolve(), args.output_dir.resolve(), args.env_file)


if __name__ == '__main__':
    main()
