"""Independent subscription judges, bounded parallelism, and clean ATIF export."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4

from agent.codex_accounts import Pool, MODEL, REASONING, TIER, save
from .codex_judge import CodexJudge
from .judge import CRITERIA, evidence_fingerprint, judge_once, SYSTEM

PROTOCOL = 'independent-codex-judges-v1'


def aggregate(grades, *, repeats):
    if len(grades) != repeats or any(g.get('status') != 'completed' for g in grades):
        return {'status': 'failed', 'error': 'All independent judgments are required', 'repeats': repeats}
    if len({g['account_ref'] for g in grades}) != repeats:
        raise ValueError('Each repetition must use a distinct subscription')
    return {'status': 'completed', 'repeats': repeats,
            'score': {'scorer': 'llm_judge',
                      'score': sum(g['score']['score'] for g in grades) / repeats,
                      # No majority vote or all-five threshold is substituted for an average.
                      'strict_pass_rate': sum(g['score']['passed'] for g in grades) / repeats,
                      'details': {key: sum(g['score']['details'][key] for g in grades) / repeats for key in CRITERIA},
                      'rationale': 'Arithmetic means of independent judgments; see individual rationales.'}}


def export_success(directory, grade, model):
    """One ATIF per successful fresh session. No failed or mixed-account attempts."""
    from benchmark.harbor.trajectory import from_turns, metrics
    native = json.loads((directory / 'judging/result.json').read_text())
    turns = native['trajectory']['turns']
    trajectory = from_turns(turns, name='codex', model=model,
        prompt=(directory / 'judging/prompt.txt').read_text(), system=SYSTEM,
        identity='judge-' + uuid4().hex, final_metrics=metrics(native, 'codex'),
        extra={'stage': 'judging', 'status': 'completed', 'score': grade['score'],
               'settings': native.get('settings', {}), 'protocol': PROTOCOL})
    save(directory / 'trajectory.json', trajectory)


def judge_cases(cases, *, model=MODEL, timeout=600, repeats=5, jobs=5, pool=None,
                backend_factory=CodexJudge, export=export_success,
                case_source=None, source_done=None, on_complete=None, admission=None, preflight=True):
    cases = [Path(c).resolve() for c in cases]
    if repeats < 1 or jobs < 1:
        raise ValueError('Repetitions and jobs must be positive')
    if len(set(cases)) != len(cases):
        raise ValueError('Duplicate cases')
    pool = pool or Pool(model=model, sessions_per_account=2)
    if admission is None and backend_factory is CodexJudge:
        from .admission import shared_admission
        admission = shared_admission
    # Serial calls and separate processes share per-account file locks.
    if preflight:
        pool.preflight()
    eligible = [a for a in pool.accounts if a.get('health') in {'ready', 'exhausted'}
                and (a.get('remaining', 0) > 0 or (a.get('resets') or 0) > 0)]
    if len(eligible) < repeats:
        raise RuntimeError(f'Need {repeats} distinct usable subscriptions; found {len(eligible)}')
    start = time.monotonic()
    states = {}
    def add_case(case):
        scenario = json.loads((case / 'scenario.json').read_text())
        execution = json.loads((case / 'result.json').read_text())
        fingerprint = evidence_fingerprint(scenario, execution, model, case)
        if execution.get('status') != 'completed':
            states[case] = {'result': {'status': 'skipped', 'error': 'Execution did not complete'}}
            return
        target = case / 'judge.json'
        previous = json.loads(target.read_text()) if target.exists() else {}
        compatible = (previous.get('protocol') == PROTOCOL and previous.get('fingerprint') == fingerprint
                      and previous.get('repeats') == repeats and previous.get('settings') ==
                      {'reasoning_effort': pool.reasoning, 'service_tier': pool.tier})
        if compatible and previous.get('status') == 'completed':
            states[case] = {'result': previous}
            return
        if not compatible and (target.exists() or (case / 'judging').exists()):
            archive = case / 'judging-history' / uuid4().hex
            archive.mkdir(parents=True)
            for source in [target, case / 'judging']:
                if source.exists():
                    shutil.move(str(source), archive / source.name)
        states[case] = {'lock': threading.Lock(), 'reserved': set(), 'grades': {}, 'fingerprint': fingerprint}
        # Reuse only complete independent records from the same immutable evidence.
        for i in range(repeats):
            path = case / 'judging/repeats' / f'{i + 1:02d}'
            grade_path = path / 'judge.json'
            if compatible and grade_path.exists():
                g = json.loads(grade_path.read_text())
                if (g.get('status') == 'completed' and g.get('fingerprint') == fingerprint
                        and g.get('account_ref') not in states[case]['reserved']
                        and (path / 'trajectory.json').exists()):
                    states[case]['grades'][i] = g
                    states[case]['reserved'].add(g['account_ref'])
        save(target, {'status': 'running', 'protocol': PROTOCOL, 'model': model, 'fingerprint': fingerprint,
                      'repeats': repeats, 'settings': {'reasoning_effort': pool.reasoning, 'service_tier': pool.tier}})

    for case in cases:
        add_case(case)

    def work(case, index, lease, account):
        """An admitted slot always does work; account waiting stays in the dispatcher."""
        state = states[case]
        temporary = case / 'judging/inflight' / uuid4().hex
        temporary.mkdir(parents=True)
        success = False
        check_recovery = False
        try:
            backend = backend_factory(temporary / 'judging', account, model=model, timeout=timeout,
                                      reasoning=pool.reasoning, tier=pool.tier)
            grade = judge_once(case, model=model, timeout=timeout, backend=backend, output_dir=temporary)
            grade.update(account_ref=account['id'], repetition=index + 1)
            if grade['status'] == 'completed':
                export(temporary, grade, model)
                save(temporary / 'judge.json', grade)
                destination = case / 'judging/repeats' / f'{index + 1:02d}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    shutil.rmtree(destination)
                temporary.rename(destination)
                success = True
            else:
                check_recovery = True
        except Exception as exc:
            grade = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            pool.quarantine(account)
        finally:
            lease.__exit__(None, None, None)
            if not success:
                with state['lock']:
                    state['reserved'].discard(account['id'])
                failed = case / 'judging-failures' / temporary.name
                failed.parent.mkdir(parents=True, exist_ok=True)
                save(temporary / 'failure.json', grade)
                temporary.rename(failed)
        if check_recovery:
            try:
                recovered = pool.reset_if_exhausted(account)
            except Exception:
                recovered = False
            if recovered is False:
                pool.quarantine(account)
        return grade

    def finish_case(case):
        state = states[case]
        if 'result' in state:
            return
        grades = [state['grades'][i] for i in range(repeats)]
        result = aggregate(grades, repeats=repeats)
        result.update(protocol=PROTOCOL, model=model, fingerprint=state['fingerprint'],
                      settings={'reasoning_effort': pool.reasoning, 'service_tier': pool.tier},
                      judgments=[f'judging/repeats/{i + 1:02d}/judge.json' for i in range(repeats)],
                      wall_seconds=state.get('finished', start) - state.get('started', start),
                      duration_seconds=sum(g.get('duration_seconds', 0) for g in grades))
        save(case / 'judge.json', result)
        state['result'] = result
        if on_complete is not None:
            on_complete(case, result)

    all_futures = []
    pending = []
    peak_active = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}

        def submit_case(case):
            state = states[case]
            if 'result' in state:
                return
            for i in range(repeats):
                if i not in state['grades']:
                    pending.append({'case': case, 'index': i, 'attempted': set(),
                                    'duration': 0., 'queued': time.monotonic()})
            if len(state['grades']) == repeats:
                finish_case(case)

        for case in cases:
            submit_case(case)
        while True:
            done = case_source is None or (source_done is not None and source_done())
            if case_source is not None:
                for candidate in case_source():
                    candidate = Path(candidate).resolve()
                    if candidate in states:
                        continue
                    cases.append(candidate)
                    add_case(candidate)
                    submit_case(candidate)
            # Scan beyond a blocked case: other cases may use the idle accounts.
            for item in list(pending):
                if len(futures) >= jobs:
                    break
                case, index = item['case'], item['index']
                state = states[case]
                with state['lock']:
                    excluded = state['reserved'] | item['attempted']
                try:
                    lease = pool.lease(exclude=excluded, wait_seconds=0)
                    account = lease.__enter__()
                except RuntimeError:
                    possible = [a for a in pool.accounts if a['id'] not in excluded | pool.disabled
                                and a.get('health') in {'ready', 'exhausted'}]
                    if not possible or time.monotonic() - item['queued'] > timeout + 120:
                        pending.remove(item)
                        state['grades'][index] = {'status': 'failed', 'error': 'No usable distinct subscription',
                                                  'duration_seconds': item['duration']}
                        state['finished'] = time.monotonic()
                        if len(state['grades']) == repeats:
                            finish_case(case)
                    continue
                if admission is not None and not admission.allow('judge'):
                    lease.__exit__(None, None, None)
                    break
                with state['lock']:
                    state['reserved'].add(account['id'])
                    state.setdefault('started', time.monotonic())
                pending.remove(item)
                item['attempted'].add(account['id'])
                future = executor.submit(work, case, index, lease, account)
                futures[future] = item
                all_futures.append(future)
                peak_active = max(peak_active, len(futures))
            if not futures:
                if done and not pending:
                    break
                time.sleep(.05)
                continue
            completed, _ = wait(futures, timeout=.05, return_when=FIRST_COMPLETED)
            for future in completed:
                item = futures.pop(future)
                case, index = item['case'], item['index']
                grade = future.result()
                item['duration'] += grade.get('duration_seconds', 0)
                if grade['status'] != 'completed' and len(item['attempted']) < min(3, len(pool.accounts)):
                    pending.append(item)
                    continue
                states[case]['grades'][index] = grade
                states[case]['finished'] = time.monotonic()
                print(f'{case.name} judgment {index + 1}/{repeats}: {grade["status"]}', flush=True)
                if len(states[case]['grades']) == repeats:
                    finish_case(case)
    results = [states[case]['result'] for case in cases]
    elapsed = time.monotonic() - start
    timing = {'wall_seconds': elapsed, 'jobs': jobs, 'peak_active_sessions': peak_active, 'cases': len(cases), 'repeats': repeats,
              'session_seconds': sum(f.result().get('duration_seconds', 0) for f in all_futures),
              'case_wall_seconds': {case.name: result.get('wall_seconds') for case, result in zip(cases, results)},
              'note': 'Session sum is a serial-equivalent estimate, not a measured sequential rerun.'}
    if cases:
        save(cases[0].parent.parent / 'judging-timing.json', timing)
    return results


def copy_executions(source, destination):
    """Create a fresh regrading input without touching original grades or logs."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Regrading output must be a fresh directory')
    paths = sorted((source / 'cases').glob('*/result.json'))
    if not paths:
        raise ValueError('No saved executions found')
    destination.mkdir(parents=True)
    hashes = {}
    for path in paths:
        target = destination / 'cases' / path.parent.name
        target.mkdir(parents=True)
        for name in ('scenario.json', 'result.json', 'workspace', 'native'):
            item = path.parent / name
            if item.is_dir():
                shutil.copytree(item, target / name)
            elif item.is_file():
                shutil.copyfile(item, target / name)
        hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('snapshot.json', 'scenarios.json', 'cohort.json'):
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)
    save(destination / 'source-evidence.json', {'source': str(source), 'result_sha256': hashes})


def export_clean_judgments(cases, destination):
    """Export completed single-account ATIFs/native logs; omit retries and auth."""
    from harbor.models.trajectories.trajectory import Trajectory
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Clean export directory must be new')
    # Validate every result before creating a partial package.
    selected = []
    for case in map(Path, cases):
        grade = json.loads((case / 'judge.json').read_text())
        if grade.get('status') != 'completed' or grade.get('protocol') != PROTOCOL:
            raise ValueError('Clean export requires completed repeated judgments')
        current = evidence_fingerprint(json.loads((case / 'scenario.json').read_text()),
                                       json.loads((case / 'result.json').read_text()), grade['model'], case)
        if current != grade.get('fingerprint'):
            raise ValueError('Evidence changed after judging')
        repeats = list((case / 'judging/repeats').glob('*/trajectory.json'))
        if len(repeats) != grade['repeats']:
            raise ValueError('Missing successful ATIF repetition')
        accounts = set()
        for trace in repeats:
            Trajectory.model_validate_json(trace.read_text())
            single = json.loads((trace.parent / 'judge.json').read_text())
            if single.get('status') != 'completed' or single.get('account_ref') in accounts:
                raise ValueError('Export requires distinct completed sessions')
            accounts.add(single['account_ref'])
        selected.append((case, grade, sorted(repeats)))
    destination.mkdir(parents=True)
    hashes = {}
    for i, (case, grade, traces) in enumerate(selected, 1):
        for trace in traces:
            folder = destination / f'case-{i:03d}' / trace.parent.name
            folder.mkdir(parents=True)
            for original, filename in [(trace, 'trajectory.json'),
                                       (trace.parent / 'judging/events.jsonl', 'events.jsonl')]:
                shutil.copyfile(original, folder / filename)
                hashes[str((folder / filename).relative_to(destination))] = hashlib.sha256(original.read_bytes()).hexdigest()
            single = json.loads((trace.parent / 'judge.json').read_text())
            save(folder / 'score.json', {'model': grade['model'], 'settings': grade['settings'],
                                        'score': single['score']})
    save(destination / 'manifest.json', {'protocol': PROTOCOL, 'files': hashes,
         'note': 'Successful fresh sessions only. Failed attempts and account credentials are excluded.'})
