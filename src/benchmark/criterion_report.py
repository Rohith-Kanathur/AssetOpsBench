"""Publish the three paper criteria from existing observed rubric summaries."""
from __future__ import annotations

import csv

CRITERIA = {
    'task_completion': 'Task completion',
    'data_retrieval_accuracy': 'Data retrieval accuracy',
    'generalized_result_verification': 'Result verification',
}


def criterion_rows(groups, names, *, repeated=False):
    rows = []
    for key, name in names.items():
        group = groups[key]
        cases = group['pooled_cases'] if repeated else group['cases']
        metrics = {}
        for criterion in CRITERIA:
            observed = cases['rubric_success_rates'].get(criterion, {})
            average = group['means']['rubric_success_rates'].get(criterion, {}) if repeated else {}
            metrics[criterion] = {
                'mean': average.get('mean') if repeated else observed.get('success_rate'),
                'sd': average.get('sd') if repeated else None,
                'observed_judgments': observed.get('observed', 0),
                'observed_repetitions': average.get('observed_repetitions', 0) if repeated else int(bool(observed)),
            }
        rows.append({'model': name, 'target': key, 'judged_trials': cases['graded'],
                     'assigned_trials': cases.get('assigned_cases', cases['attempted']), 'metrics': metrics})
    return rows


def publish_criterion_averages(dest, rows, colors, *, repeated=False):
    """Render averages without changing the underlying scores or pass decisions."""
    fields = ['model', 'target', 'criterion', 'mean', 'sd', 'observed_judgments',
              'observed_repetitions', 'judged_trials', 'assigned_trials']
    with (dest / 'criterion-averages.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            for criterion, metric in row['metrics'].items():
                writer.writerow({k: v for k, v in row.items() if k != 'metrics'} |
                                {'criterion': criterion} | metric)

    from .paper_plots import criterion_figure, save_figure
    fig = criterion_figure([('', rows)], [row['model'] for row in rows], colors, repeated=repeated)
    save_figure(fig, dest / 'graphs', 'criterion-averages')

    table = ['| Model | Task completion (%) | Data retrieval accuracy (%) | Result verification (%) | Judged / assigned |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        cells = []
        for metric in row['metrics'].values():
            cell = '—' if metric['mean'] is None else f"{metric['mean'] * 100:.1f}"
            if metric['sd'] is not None:
                cell += f" ± {metric['sd'] * 100:.1f}"
            cells.append(cell)
        table.append(f"| {row['model']} | {' | '.join(cells)} | {row['judged_trials']}/{row['assigned_trials']} |")
    return '\n'.join(table)
