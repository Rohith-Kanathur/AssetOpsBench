"""Run generation → isolated executions → independent judging as Harbor trials."""

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
import tomllib

from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.models.task.config import TaskConfig as TaskFormat
from harbor.models.trajectories.trajectory import Trajectory
from harbor.trial.trial import Trial
from benchmark.generated.judge import JUDGE_MODEL
from scenarios.generation.harnesses import DEFAULT_MODEL, DEFAULT_REASONING

from .tasks import create_task
from .trajectory import generation_trajectory, manifest, read, run_id, write


def snapshot_controller(repo, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copyfile(repo / name, destination / name)
    for name in ('benchmark/harbor', 'benchmark/generated', 'scenarios/generation'):
        shutil.copytree(repo / 'src' / name, destination / 'src' / name,
                        ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    rubric = destination / 'src/evaluation/scorers/llm_judge.py'
    rubric.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / 'src/evaluation/scorers/llm_judge.py', rubric)


def workflow_index(root, stages):
    imported = read(root / 'invocation.json', {}).get('settings', {}).get('action') == 'evaluate'
    description = ('Execute the prepared cohort independently, then judge each saved execution.' if imported else
                   'Generate scenarios, execute each independently, then judge each saved execution.')
    steps = [{'step_id': 1, 'source': 'user', 'message': description}]
    for stage in stages:
        steps.append({'step_id': len(steps) + 1, 'source': 'agent', 'llm_call_count': 0,
            'message': stage['name'], 'observation': {'results': [{'content': stage['status'],
            'subagent_trajectory_ref': [{'trajectory_path': stage['trajectory']}]}]}})
    trajectory = {'schema_version': 'ATIF-v1.8', 'session_id': run_id(root),
        'agent': {'name': 'assetops-pipeline', 'version': '0.1.0'}, 'steps': steps,
        'extra': {'stages': stages}}
    write(root / 'trajectory.json', Trajectory.model_validate(trajectory).to_json_dict())
    lines = ['# AssetOpsBench Harbor run', '',
             ('Input: a prepared cohort. ' if imported else 'Creation: the configured authoring model. ') +
             'Execution: the configured model matrix. Judge: the configured Claude Code model.', '',
             '| Stage | Model | Status | Trajectory |', '|---|---|---|---|']
    for stage in stages:
        status = stage['status'] + (' (superseded)' if stage.get('superseded_by') else '')
        lines.append(f"| {stage['name']} | {stage.get('model', '')} | {status} | [ATIF]({stage['trajectory']}) |")
    lines += ['', 'Operational completion and benchmark pass are separate rewards.', '',
              '[Workflow trajectory](trajectory.json) · [Evidence manifest](evidence-manifest.json)', '',
              'This is a local run record, not an anonymized publication package.', '']
    (root / 'README.md').write_text('\n'.join(lines))
    manifest(root)


async def trial(root, name, spec, stages):
    task = create_task(root, name, spec)
    config = TrialConfig(task=TaskConfig(path=task), trial_name=name, trials_dir=root / 'trials',
        agent=AgentConfig(import_path='benchmark.harbor.agent:PipelineAgent',
                          model_name=spec['model'], kwargs={'spec_path': str(task / 'stage.json')}))
    instance = await Trial.create(config)
    result = await instance.run()
    trajectory = root / 'trials' / name / 'agent/trajectory.json'
    if not trajectory.exists():
        from .stage import collect
        collect(task / 'stage.json', trajectory.parent)
    rewards = (result.verifier_result.rewards or {}) if result.verifier_result else {}
    status = 'completed' if not result.exception_info and rewards.get('stage_completed') == 1 else 'failed'
    stages.append({'name': name, 'model': spec['model'], 'status': status, 'task': str(task.relative_to(root)),
                   'trajectory': str(trajectory.relative_to(root)), 'rewards': rewards})
    workflow_index(root, stages)
    print(f'{name}: {status}; {rewards}', flush=True)
    return status == 'completed'


def prepare_run(args):
    from benchmark.generated.cli import runner_models
    root = args.directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'trials').exists():
        raise ValueError('Use a fresh directory; existing trials are never overwritten')
    runner_models(json.loads(args.runners), 'general-execution')
    repo = args.repo.resolve()
    snapshot_controller(repo, root / 'controller')
    settings = {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    write(root / 'invocation.json', {'argv': sys.argv, 'settings': settings})
    return root


async def run(args):
    from benchmark.generated import sandbox
    root = prepare_run(args)
    stages = []
    generation = root / 'generation'
    if args.reuse_generation:
        if not (generation / 'status.json').exists():
            raise ValueError('--reuse-generation requires DIRECTORY/generation')
        write(root / 'imports/generation/trajectory.json', generation_trajectory(generation))
        stages.append({'name': 'generation', 'status': read(generation / 'status.json')['status'],
                       'trajectory': 'imports/generation/trajectory.json', 'origin': 'external native run; not a Harbor trial'})
        workflow_index(root, stages)
    else:
        arguments = ['run', str(generation), '--repo', str(args.repo.resolve()), '--asset', args.asset,
                     '--environment', args.environment,
                     '--model', args.generation_model, '--reasoning', args.reasoning, '--tier', 'default']
        arguments += ['--plan', args.plan] if args.plan is not None else ['--count', str(args.count)]
        if args.seed:
            arguments += ['--seed', str(args.seed.expanduser().resolve())]
        complete = await trial(root, 'generation', {'stage': 'generation', 'arguments': arguments,
                    'timeout': args.generation_timeout, 'model': args.generation_model}, stages)
        if not complete:
            raise RuntimeError('Generation did not pass validation; see trials/generation')
    if args.stop_after == 'generation':
        return
    sandbox.snapshot(generation, root / 'evaluation')
    # Snapshotting may restart the generator's database. It is no longer needed.
    from scenarios.generation.runtime import stop
    stop(generation)
    await execute_cohort(root, args, stages)


async def evaluate(args):
    from benchmark.generated import sandbox
    root = prepare_run(args)
    sandbox.import_snapshot(args.snapshot, root / 'evaluation')
    await execute_cohort(root, args, [])


async def execute_cohort(root, args, stages):
    from benchmark.generated.auth import private_json
    from benchmark.generated.cli import slug, runner_models
    from benchmark.generated.report import write_report
    selections = runner_models(json.loads(args.runners), 'general-execution')
    scenarios = read(root / 'evaluation/scenarios.json')
    if args.max_cases:
        scenarios = scenarios[:args.max_cases]
    settings = {'max_turns': args.max_turns, 'max_output_tokens': 8192,
                'reasoning_effort': args.execution_reasoning, 'temperature': None}
    write(root / 'evaluation/cohort.json', {'format_version': 1, 'runners': json.loads(args.runners),
        'scenario_ids': [row['id'] for row in scenarios], 'stirrup_settings': settings})
    for scenario in scenarios:
        for runner, model in selections:
            key = f"{slug(runner)}-{slug(model)}-{slug(scenario['id'])}"
            case = root / 'evaluation/cases' / key
            case.mkdir(parents=True)
            private_json(case / 'scenario.json', scenario)
            private_json(case / 'result.json', {'scenario_id': scenario['id'], 'status': 'pending',
                         'runner': runner, 'model': model})
            spec = {'stage': 'execution', 'case': str(case.relative_to(root)),
                    'runner': runner, 'model': model, 'timeout': args.timeout, 'settings': settings}
            await trial(root, 'execution-' + key, spec, stages)
            if args.stop_after != 'execution':
                spec = {'stage': 'judging', 'case': str(case.relative_to(root)),
                        'model': args.judge_model, 'timeout': args.timeout}
                await trial(root, 'judging-' + key, spec, stages)
    write_report(root / 'evaluation', title='Harbor pipeline evaluation')
    workflow_index(root, stages)
    if any(stage['status'] == 'failed' for stage in stages):
        raise RuntimeError('One or more stages failed; all requested cases were attempted and saved')


async def rejudge(args):
    from benchmark.generated.report import write_report
    root = args.directory.expanduser().resolve()
    stages = read(root / 'trajectory.json')['extra']['stages']
    cases = sorted((root / 'evaluation/cases').glob(args.case or '*'))
    if not cases or any(not (case / 'result.json').is_file() for case in cases):
        raise ValueError('No matching saved execution cases')
    completed = True
    for case in cases:
        prefix = 'judging-' + case.name
        number = 1
        while (root / 'tasks' / f'{prefix}-retry{number}').exists():
            number += 1
        name = f'{prefix}-retry{number}'
        controller = root / 'controller/retries' / name
        snapshot_controller(Path(__file__).resolve().parents[3], controller)
        write(controller / 'invocation.json', {'argv': sys.argv,
            'directory': str(root), 'case': case.name, 'model': args.model, 'timeout': args.timeout})
        for previous in stages:
            if previous['name'].startswith(prefix) and 'superseded_by' not in previous:
                previous['superseded_by'] = name
        completed &= await trial(root, name, {'stage': 'judging',
            'case': str(case.relative_to(root)), 'model': args.model, 'timeout': args.timeout}, stages)
    write_report(root / 'evaluation', title='Harbor pipeline evaluation')
    workflow_index(root, stages)
    if not completed:
        raise RuntimeError('Judge retry failed; saved executions were retained')


def validate(root):
    root = Path(root).resolve()
    if not (root / 'trajectory.json').is_file() or not (root / 'evidence-manifest.json').is_file():
        raise ValueError('Expected a workflow trajectory and evidence manifest')
    for task in (root / 'tasks').glob('*/task.toml'):
        TaskFormat.model_validate(tomllib.loads(task.read_text()))
    count = 0
    for path in root.glob('**/trajectory.json'):
        Trajectory.model_validate_json(path.read_text())
        count += 1
    for path in root.glob('trials/*/result.json'):
        TrialResult.model_validate_json(path.read_text())
    saved = read(root / 'evidence-manifest.json', {})
    import hashlib
    for relative, expected in saved.get('files', {}).items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError(f'Manifest path escapes root: {relative}')
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Evidence changed: {relative}')
    index = read(root / 'trajectory.json', {})
    for stage in index.get('extra', {}).get('stages', []):
        target = (root / stage['trajectory']).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f'Missing stage trajectory: {stage["name"]}')
    print(f'Validated {count} ATIF trajectories, native trial results, links and {len(saved.get("files", {}))} evidence hashes.')


def execution_options(command):
    command.add_argument('directory', type=Path)
    command.add_argument('--repo', type=Path, default=Path.cwd())
    command.add_argument('--runners', required=True, help='JSON runner → model or list of models; same matrix as scenario-evaluate')
    command.add_argument('--judge-model', default=JUDGE_MODEL)
    command.add_argument('--execution-reasoning', default='high')
    command.add_argument('--max-cases', type=int)
    command.add_argument('--max-turns', type=int, default=20)
    command.add_argument('--timeout', type=float, default=600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('run')
    execution_options(command)
    command.add_argument('--asset', default='Chiller')
    budget = command.add_mutually_exclusive_group()
    budget.add_argument('--count', type=int, default=25)
    budget.add_argument('--plan', help='JSON domain → integer count')
    command.add_argument('--seed', type=Path)
    command.add_argument('--environment', choices=['existing', 'extend'], default='existing')
    command.add_argument('--generation-model', default=DEFAULT_MODEL)
    command.add_argument('--reasoning', default=DEFAULT_REASONING)
    command.add_argument('--generation-timeout', type=float, default=1800)
    command.add_argument('--reuse-generation', action='store_true')
    command.add_argument('--stop-after', choices=['generation', 'execution', 'judging'], default='judging')
    command = sub.add_parser('evaluate', help='Execute and judge a prepared human or synthetic cohort without generation')
    execution_options(command)
    command.add_argument('--snapshot', type=Path, required=True)
    command.add_argument('--stop-after', choices=['execution', 'judging'], default='judging')
    command = sub.add_parser('judge', help='Judge saved executions without repeating generation or execution')
    command.add_argument('directory', type=Path)
    command.add_argument('--case', help='Saved execution case name; default: all cases')
    command.add_argument('--model', default=JUDGE_MODEL)
    command.add_argument('--timeout', type=float, default=600)
    command = sub.add_parser('validate')
    command.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'validate':
        validate(args.directory)
    elif args.action == 'judge':
        if args.timeout <= 0:
            parser.error('Timeout must be positive')
        asyncio.run(rejudge(args))
    else:
        if (args.timeout <= 0 or getattr(args, 'generation_timeout', 1) <= 0 or
                getattr(args, 'count', 1) < 1 or args.max_turns < 1 or
                (args.max_cases is not None and args.max_cases < 1)):
            parser.error('Timeouts and limits must be positive')
        # Gateway credentials are scoped to the subprocess; no values are persisted.
        import os
        if os.environ.get('AI_GATEWAY_API_KEY'):
            os.environ['LITELLM_API_KEY'] = os.environ['AI_GATEWAY_API_KEY']
            os.environ['LITELLM_BASE_URL'] = 'https://ai-gateway.vercel.sh/v1'
        asyncio.run(evaluate(args) if args.action == 'evaluate' else run(args))


if __name__ == '__main__':
    main()
