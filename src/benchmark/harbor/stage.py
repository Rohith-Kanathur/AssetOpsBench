"""Trusted host-side stage dispatch; credentials never enter task specifications."""

import argparse
import json
import os
from pathlib import Path

from .trajectory import case_trajectory, generation_trajectory, read, write


def execute(spec_path):
    spec_path = Path(spec_path).resolve()
    root = spec_path.parent.parent.parent
    spec = read(spec_path)
    stage = spec['stage']
    if stage == 'generation':
        from scenarios.generation.cli import main
        main(spec['arguments'])
        return
    case = root / spec['case']
    if stage == 'execution':
        from benchmark.generated.cli import execute_case, evaluation_credentials
        scenario = read(case / 'scenario.json')
        execute_case(root / 'evaluation', case, scenario, spec['runner'], spec['model'],
                     spec['timeout'], evaluation_credentials({}), settings=spec['settings'], judge=False)
    elif stage == 'judging':
        from benchmark.generated.judge import judge_case
        judge_case(case, model=spec['model'], timeout=spec['timeout'],
                   repeats=spec.get('repeats', 5), jobs=spec.get('jobs', 5))
    else:
        raise ValueError(f'Unknown stage: {stage}')


def collect(spec_path, logs):
    """Always export partial evidence, including on a timeout or failed process."""
    spec_path = Path(spec_path).resolve()
    root = spec_path.parent.parent.parent
    spec = read(spec_path)
    if spec['stage'] == 'generation':
        record = read(root / 'generation/status.json', {})
        trajectory = generation_trajectory(root / 'generation')
        success = record.get('status') == 'complete'
        outcome = {'stage': 'generation', 'completed': success, 'status': record.get('status', 'missing')}
    else:
        case = root / spec['case']
        record = read(case / ('judge.json' if spec['stage'] == 'judging' else 'result.json'), {})
        trajectory = case_trajectory(case, spec['stage'])
        outcome = {'stage': spec['stage'], 'completed': record.get('status') == 'completed',
                   'status': record.get('status', 'missing')}
        if spec['stage'] == 'judging':
            score = record.get('score', {})
            outcome['benchmark_pass'] = score.get('strict_pass_rate', score.get('passed'))
            outcome['rubric'] = record.get('score', {}).get('details')
    write(Path(logs) / 'trajectory.json', trajectory)
    write(Path(logs) / 'outcome.json', outcome)
    return outcome


def cleanup(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = read(spec_path)
    root = spec_path.parent.parent.parent
    try:
        if spec['stage'] == 'generation' and (root / 'generation/compose.json').exists():
            from scenarios.generation.runtime import stop
            stop(root / 'generation')
            state = read(root / 'generation/status.json', {})
            if state.get('status') in {'running', 'checking', 'repairing'}:
                from scenarios.generation.runtime import save_status
                save_status(root / 'generation', 'failed', error_type='Interrupted')
        elif spec['stage'] == 'execution':
            from benchmark.generated.sandbox import compose
            case = root / spec['case']
            if (case / 'compose.json').exists():
                compose(case, 'down', '--volumes', '--remove-orphans')
            record = read(case / 'result.json', {})
            if record.get('status') in {'running', 'pending'}:
                record.update(status='error', error='Interrupted')
                write(case / 'result.json', record)
        elif spec['stage'] == 'judging':
            import re
            import subprocess
            for marker in (root / spec['case'] / 'judging').rglob('container.json'):
                name = read(marker, {}).get('name', '')
                if re.fullmatch(r'assetops-judge-[0-9a-f]{12}', name):
                    subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
    finally:
        if spec.get('case'):
            import shutil
            shutil.rmtree(root / spec['case'] / 'auth', ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    execute(args.spec)
