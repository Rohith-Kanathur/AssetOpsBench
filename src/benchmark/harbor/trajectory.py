"""Loss-aware conversion of saved pipeline evidence to Harbor's ATIF schema."""

import hashlib
import json
from pathlib import Path

from harbor.models.trajectories.trajectory import Trajectory


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def metrics(record, harness=None):
    result = {}
    for source, target in [('api_prompt_tokens', 'total_prompt_tokens'),
                           ('api_output_tokens', 'total_completion_tokens'),
                           ('cache_read_tokens', 'total_cached_tokens'), ('cost_usd', 'total_cost_usd')]:
        if record.get(source) is not None:
            result[target] = record[source]
    usage = record.get('usage') or {}
    if 'input_tokens' in usage:
        result['total_prompt_tokens'] = usage['input_tokens']
        if harness == 'claude':
            result['total_prompt_tokens'] += usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
    if 'output_tokens' in usage:
        result['total_completion_tokens'] = usage['output_tokens']
    for key in ('cached_input_tokens', 'cache_read_input_tokens'):
        if key in usage:
            result['total_cached_tokens'] = usage[key]
    return result


def from_turns(turns, *, name, model, prompt, identity, extra=None, system=None, final_metrics=None):
    steps = []
    if system:
        steps.append({'source': 'system', 'message': text(system)})
    steps.append({'source': 'user', 'message': prompt})
    for turn in turns:
        step = {'source': 'agent', 'message': text(turn.get('text') or '')}
        calls, results = [], []
        for i, call in enumerate(turn.get('tool_calls', [])):
            cid = str(call.get('id') or f'call-{len(steps)}-{i}')
            arguments = call.get('input', {})
            if not isinstance(arguments, dict):
                arguments = {'value': arguments}
            calls.append({'tool_call_id': cid, 'function_name': call.get('name') or 'unknown',
                          'arguments': arguments})
            if 'output' in call:
                results.append({'source_call_id': cid, 'content': text(call['output'])})
        if calls:
            step['tool_calls'] = calls
        if results:
            step['observation'] = {'results': results}
        steps.append(step)
    for i, step in enumerate(steps, 1):
        step['step_id'] = i
    payload = {'schema_version': 'ATIF-v1.8', 'session_id': identity,
               'trajectory_id': identity,
               'agent': {'name': name, 'version': 'assetopsbench-0.1.0', 'model_name': model},
               'steps': steps, 'extra': extra or {},
               'notes': 'Converted from recorded native events. Native files remain authoritative. '
                        'Unrecorded inference boundaries, timestamps, usage and private reasoning are not inferred.'}
    if final_metrics:
        payload['final_metrics'] = {**final_metrics, 'total_steps': len(steps)}
    return Trajectory.model_validate(payload).to_json_dict()


def parse_codex(native):
    from agent.coding_agent.trajectory import parse
    result = parse(native, 'codex')
    turns = []
    for line in native.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get('item') or {}
        if event.get('type') == 'item.completed' and item.get('type') == 'web_search':
            arguments = item.get('action') or {'query': item.get('query')}
            turns.append({'text': '', 'tool_calls': [{'id': item.get('id'),
                'name': 'web_search', 'input': arguments}]})
        else:
            turns.extend(parse(line, 'codex')['trajectory']['turns'])
    result['trajectory']['turns'] = turns
    return result


def run_id(root):
    # Stable across exports, distinct across local run directories; no path disclosure.
    return hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:20]


def generation_trajectory(directory):
    directory = Path(directory)
    identity = run_id(directory.parent)
    attempts = []
    for path in sorted((directory / 'logs').glob('codex-*.jsonl'), key=lambda p: int(p.stem.split('-')[-1])):
        number = path.stem.split('-')[-1]
        meta = read(directory / 'logs' / f'run-{number}.json', {})
        parsed = parse_codex(path.read_text())
        attempts.append(from_turns(parsed.get('trajectory', {}).get('turns', []), name='codex',
            model=meta.get('requested_model', 'unknown'),
            prompt=meta.get('prompt', 'Prompt not recorded by this older run; see generation guidance and request.json.'),
            identity=f'{identity}-generation-attempt-{number}', final_metrics=metrics(parsed, 'codex'), extra={'stage': 'generation', 'attempt': number,
                'native_log': str(path.relative_to(directory)), 'run_metadata': meta,
                'status': meta.get('process_status', 'unknown')}))
    if not attempts:
        attempts.append(from_turns([], name='codex', model='unknown', prompt='Generation failed before agent execution.',
                                   identity=f'{identity}-generation-attempt-0'))
    root = from_turns([], name='assetops-pipeline', model='none', prompt='Generate and validate the requested scenarios.',
                      identity=f'{identity}-generation', extra={'stage': 'generation', 'status': read(directory / 'status.json', {})})
    root['subagent_trajectories'] = attempts
    # Deterministic attempt dispatch, rather than claiming that orchestration is an inference.
    for i, attempt in enumerate(attempts, 2):
        root['steps'].append({'step_id': i, 'source': 'agent', 'message': 'Run generation attempt',
            'llm_call_count': 0, 'observation': {'results': [{'content': attempt['extra'].get('status', 'unknown'),
            'subagent_trajectory_ref': [{'trajectory_id': attempt['trajectory_id']}]}]}})
    return Trajectory.model_validate(root).to_json_dict()


def case_trajectory(case, stage):
    case = Path(case)
    if stage == 'execution':
        record = read(case / 'result.json', {})
        prompt = read(case / 'scenario.json', {}).get('text', '')
        system = read(case / 'native/system-prompt.json')
        if isinstance(system, dict):
            system = system.get('prompt')
    else:
        grade = read(case / 'judge.json', {})
        record = read(case / 'judging/result.json', {})
        if not record and (case / 'judging/events.jsonl').exists():
            from agent.coding_agent.trajectory import parse
            record = parse((case / 'judging/events.jsonl').read_text(), 'claude')
        record = {**record, 'model': grade.get('model', 'unknown'), 'runner': record.get('runner', 'judge')}
        prompt_path = case / 'judging/prompt.txt'
        prompt = prompt_path.read_text() if prompt_path.exists() else 'Judge did not start; see judge.json.'
        from benchmark.generated.judge import SYSTEM
        system = SYSTEM
    trajectory = record.get('trajectory') or {}
    turns = trajectory.get('turns', []) if isinstance(trajectory, dict) else []
    if record.get('answer') and (not turns or turns[-1].get('text') != record['answer']):
        turns = [*turns, {'text': record['answer']}]
    extra = {'stage': stage, 'status': record.get('status', record.get('completed', 'unknown')),
             'scenario_id': read(case / 'scenario.json', {}).get('id')}
    extra['settings'] = record.get('settings', {})
    if stage == 'judging':
        extra['grade'] = grade
    return from_turns(turns, name=record.get('runner', 'unknown'), model=record.get('model', 'unknown'),
                      prompt=prompt, system=system, identity=f'{run_id(case.parents[2])}-{stage}-{case.name}', extra=extra,
                      final_metrics=metrics(record, 'claude' if stage == 'judging' else record.get('runner')))


def manifest(root):
    """Hash selected evidence only, never runtime credentials or compose configs."""
    root = Path(root)
    evidence = []
    for pattern in ('controller/**/*', 'invocation.json', 'README.md', 'trajectory.json', 'imports/**/trajectory.json',
                    'tasks/*/task.toml', 'tasks/*/instruction.md', 'tasks/*/stage.json',
                    'tasks/*/tests/test.sh', 'tasks/*/environment/Dockerfile',
                    'trials/*/result.json', 'trials/*/agent/trajectory.json',
                    'trials/*/agent/outcome.json', 'trials/*/verifier/reward.json',
                    'generation/logs/run-*.json', 'generation/logs/codex-*.jsonl',
                    'generation/logs/research.jsonl', 'generation/logs/review-*.json',
                    'generation/request.json', 'generation/baseline.json', 'generation/status.json',
                    'generation/review.json', 'generation/environment-baseline.json',
                    'generation/workspace/*.md', 'generation/workspace/references/**/*',
                    'generation/workspace/data/**/*', 'generation/workspace/scripts/**/*',
                    'generation/workspace/src/**/*',
                    'generation/workspace/output/**/*.json', 'generation/workspace/output/**/*.md',
                    'evaluation/snapshot.json', 'evaluation/scenarios.json', 'evaluation/cohort.json',
                    'evaluation/README.md', 'evaluation/cases.csv', 'evaluation/*.png',
                    'evaluation/cases/*/scenario.json',
                    'evaluation/cases/*/result.json', 'evaluation/cases/*/judge.json',
                    'evaluation/cases/*/native/*.json*', 'evaluation/cases/*/judging/*.json*',
                    'evaluation/cases/*/judging/prompt.txt', 'evaluation/cases/*/judging/evidence/**/*',
                    'evaluation/cases/*/judging-attempts/**/*', 'evaluation/cases/*/workspace/**/*',
                    'evaluation/environment/**/*', 'evaluation/database/*', 'evaluation/inputs/**/*'):
        evidence.extend(p for p in root.glob(pattern) if p.is_file() and not p.is_symlink())
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(set(evidence)) if not any(x in p.parts for x in ('.env', 'auth', '__pycache__'))}
    write(root / 'evidence-manifest.json', {'format_version': 1, 'files': files,
          'publication_ready': False, 'note': 'Local evidence manifest. Review source/data licenses, anonymity and trace contents before publication. Runtime configuration and credentials are excluded.'})
    return files
