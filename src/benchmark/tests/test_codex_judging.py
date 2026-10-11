"""No live inference: test independent sessions, failure recovery, and averages."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

import pytest

from agent.codex_accounts import Pool, allowance, is_excluded, save, merge_credentials
from benchmark.generated.judge import CRITERIA
from benchmark.generated.repeated_judge import aggregate, judge_cases, copy_executions
from benchmark.generated.report import write_report
from llm.base import LLMBackend
import base64


def review(passes=True):
    return {key: (passes if key == 'task_completion' else key != 'hallucinations') for key in CRITERIA}


class FakePool:
    reasoning, tier = 'xhigh', 'fast'

    def __init__(self, count=6):
        self.accounts = [{'id': str(i), 'health': 'ready', 'remaining': 90, 'resets': 2} for i in range(count)]
        self.busy, self.disabled = set(), set()
        self.guard = threading.Lock()
        self.max_active = 0

    def preflight(self):
        return self.accounts

    @contextmanager
    def lease(self, exclude=(), wait_seconds=3):
        deadline = time.monotonic() + wait_seconds
        while True:
            with self.guard:
                eligible = [a for a in self.accounts if a['id'] not in set(exclude) | self.disabled]
                available = [a for a in eligible if a['id'] not in self.busy]
                if available:
                    account = available[0]
                    self.busy.add(account['id'])
                    self.max_active = max(self.max_active, len(self.busy))
                    break
                if not eligible or time.monotonic() > deadline:
                    raise RuntimeError('No account')
            time.sleep(.005)
        try:
            yield account
        finally:
            with self.guard:
                self.busy.remove(account['id'])

    def reset_if_exhausted(self, account):
        return False

    def quarantine(self, account):
        self.disabled.add(account['id'])


class FakeJudge(LLMBackend):
    calls = []
    fail = set()

    def __init__(self, audit, account, **settings):
        self.audit, self.account = audit, account

    def generate(self, prompt, temperature=0):
        type(self).calls.append(self.account['id'])
        self.audit.mkdir(exist_ok=True)
        (self.audit / 'events.jsonl').write_text('fresh session ' + self.account['id'])
        time.sleep(.04)
        if self.account['id'] in self.fail:
            raise RuntimeError('subscription usage limit')
        return json.dumps(review(self.account['id'] != '2'))


@pytest.fixture
def case(tmp_path):
    case = tmp_path / 'cases/one'
    case.mkdir(parents=True)
    (case / 'scenario.json').write_text(json.dumps({'id': '1', 'text': 'Inspect', 'characteristic_form': 'Read sensor'}))
    (case / 'result.json').write_text(json.dumps({'status': 'completed', 'answer': 'done', 'model': 'luna'}))
    FakeJudge.calls, FakeJudge.fail = [], set()
    return case


def fake_export(path, grade, model):
    save(path / 'trajectory.json', {'successful': True})


def test_parallel_five_distinct_judges_and_exact_averages(case):
    pool = FakePool()
    result = judge_cases([case], pool=pool, backend_factory=FakeJudge, export=fake_export)[0]
    assert pool.max_active == 5
    assert len(set(FakeJudge.calls)) == 5
    assert result['score']['strict_pass_rate'] == .8
    assert result['score']['details']['task_completion'] == .8
    assert result['score']['details']['hallucinations'] == 0
    assert result['score']['score'] == pytest.approx(.96)
    assert 'passed' not in result['score']
    text = write_report(case.parents[1], chart=False).read_text()
    assert '0.8/1 (80%)' in text
    assert '5 judgment(s)' in text
    judge_cases([case], pool=pool, backend_factory=FakeJudge, export=fake_export)
    assert len(FakeJudge.calls) == 5  # Cache uses the same immutable evidence.


def test_exhausted_attempt_is_replaced_whole_and_excluded_from_clean_traces(case):
    pool = FakePool()
    FakeJudge.fail = {'0'}
    result = judge_cases([case], pool=pool, backend_factory=FakeJudge, export=fake_export)[0]
    assert result['status'] == 'completed'
    completed = [json.loads(p.read_text()) for p in case.glob('judging/repeats/*/judge.json')]
    assert len(completed) == 5 and len({g['account_ref'] for g in completed}) == 5
    assert '0' not in {g['account_ref'] for g in completed}
    assert len(list(case.glob('judging-failures/*/judging/events.jsonl'))) == 1
    assert len(list(case.glob('judging/repeats/*/trajectory.json'))) == 5
    assert not list(case.glob('judging-failures/**/trajectory.json'))
    assert not pool.busy


def test_insufficient_accounts_blocks_before_inference(case):
    with pytest.raises(RuntimeError, match='found 4'):
        judge_cases([case], pool=FakePool(4), backend_factory=FakeJudge, export=fake_export)
    assert not FakeJudge.calls


def test_duplicate_account_cannot_satisfy_five_repetitions():
    grades = [{'status': 'completed', 'account_ref': 'one'}] * 5
    with pytest.raises(ValueError, match='distinct'):
        aggregate(grades, repeats=5)


def test_copy_preserves_original_judges_and_excludes_auth(case, tmp_path):
    (case / 'judge.json').write_text('original grade')
    (case / 'auth').mkdir()
    (case / 'auth/auth.json').write_text('private')
    destination = tmp_path / 'new'
    copy_executions(case.parents[1], destination)
    assert (case / 'judge.json').read_text() == 'original grade'
    assert not (destination / 'cases/one/judge.json').exists()
    assert not (destination / 'cases/one/auth').exists()
    assert (destination / 'cases/one/result.json').read_bytes() == (case / 'result.json').read_bytes()


def test_allowance_and_unknown_values():
    assert allowance({})['remaining'] is None
    usage = allowance({'rateLimits': {'primary': {'usedPercent': 100}},
                       'rateLimitResetCredits': {'availableCount': 2}})
    assert usage['remaining'] == 0 and usage['resets'] == 2


def test_account_leases_are_exclusive_and_release_after_exception(tmp_path, monkeypatch):
    pool = Pool(tmp_path)
    home = tmp_path / 'home'
    home.mkdir()
    account = {'id': 'a', 'home': str(home), 'health': 'ready', 'remaining': 90}
    pool.accounts = [account]
    monkeypatch.setattr(pool, 'inspect', lambda *a, **k: {'remaining': 90})
    with pytest.raises(ValueError):
        with pool.lease() as selected:
            assert selected == account
            assert pool._lock(account) is None
            raise ValueError()
    lock = pool._lock(account)
    assert lock is not None
    lock.close()


def test_reset_unknown_response_reuses_durable_key(tmp_path, monkeypatch):
    pool = Pool(tmp_path)
    home = tmp_path / 'home'
    home.mkdir()
    account = {'home': str(home)}
    keys = []
    responses = ['unknown', 'reset']

    @contextmanager
    def rpc(_):
        def call(method, params=None):
            if method.endswith('/read'):
                return {'rateLimits': {'primary': {'usedPercent': 100}},
                        'rateLimitResetCredits': {'availableCount': 1}}
            keys.append(params['idempotencyKey'])
            return {'outcome': responses.pop(0)}
        yield call

    monkeypatch.setattr('agent.codex_accounts.rpc', rpc)
    with pytest.raises(RuntimeError, match='Unknown reset'):
        pool.reset_if_exhausted(account)
    pool.reset_if_exhausted(account)
    assert len(keys) == 2 and keys[0] == keys[1]
    assert not pool.reset_if_exhausted(account)  # Awaiting clear: no second redemption.
    assert len(keys) == 2


def test_codex_judge_mounts_no_originals_or_other_account_files(case, tmp_path, monkeypatch):
    from benchmark.generated.codex_judge import CodexJudge
    from benchmark.generated.judge import judge_once
    from benchmark.generated.repeated_judge import export_success
    account_home = tmp_path / 'private-account'
    account_home.mkdir()
    token = base64.urlsafe_b64encode(json.dumps({'email': 'test@example.org'}).encode()).decode().rstrip('=')
    save(account_home / 'auth.json', {'tokens': {'id_token': f'x.{token}.x'}})
    (account_home / 'usage.json').write_text('private account metadata')
    account = {'home': str(account_home), 'email': 'test@example.org'}
    output = tmp_path / 'judge-output'

    def launch(command, **kwargs):
        assert '--read-only' in command and '--ignore-user-config' in command and '--ephemeral' in command
        assert command[command.index('--model') + 1] == 'gpt-6-astra'
        assert 'model_reasoning_effort="xhigh"' in command
        assert 'service_tier="fast"' in command
        assert 'web_search="disabled"' in command
        mounts = [command[i + 1] for i, word in enumerate(command) if word == '--mount']
        assert len(mounts) == 2 and any(m.endswith('dst=/evidence,readonly') for m in mounts)
        auth = next(m for m in mounts if 'dst=/auth' in m)
        source = Path(auth.split('src=', 1)[1].split(',', 1)[0])
        assert sorted(p.name for p in source.iterdir()) == ['auth.json']
        assert not any(str(case) in m for m in mounts)
        assert 'API_KEY' not in ' '.join(command)

        class Process:
            returncode = 0

            def communicate(self, prompt, timeout):
                assert 'independent AssetOpsBench evaluator' in prompt
                kwargs['stdout'].write(json.dumps({'type': 'item.completed',
                    'item': {'type': 'agent_message', 'text': json.dumps(review())}}) + '\n')
                kwargs['stdout'].write('{"type":"turn.completed","usage":{"input_tokens":100}}\n')

        return Process()

    monkeypatch.setattr('benchmark.generated.codex_judge.subprocess.Popen', launch)
    grade = judge_once(case, backend=CodexJudge(output / 'judging', account), output_dir=output)
    assert grade['status'] == 'completed'
    export_success(output, grade, 'gpt-6-astra')
    assert json.loads((output / 'trajectory.json').read_text())['agent']['model_name'] == 'gpt-6-astra'


def test_author_rotates_serially_and_persists_refresh(tmp_path, monkeypatch):
    import subprocess
    from scenarios.generation import runtime
    pool = Pool(tmp_path / 'pool')
    pool.root.mkdir()
    pool.accounts = []
    for i in range(2):
        home = pool.root / str(i)
        home.mkdir()
        email = f'{i}@example.org'
        token = base64.urlsafe_b64encode(json.dumps({'email': email}).encode()).decode().rstrip('=')
        save(home / 'auth.json', {'tokens': {'id_token': f'x.{token}.x'}})
        pool.accounts.append({'id': str(i), 'email': email, 'home': str(home), 'health': 'ready', 'remaining': 90})
    monkeypatch.setattr(pool, 'preflight', lambda: pool.accounts)
    monkeypatch.setattr(pool, 'inspect', lambda *a, **k: {'remaining': 90})
    monkeypatch.setattr(pool, 'reset_if_exhausted', lambda *a: False)
    homes, calls = [], []
    monkeypatch.setattr(runtime, 'configure', lambda destination, auth, kaggle: homes.append(auth))
    destination = tmp_path / 'generation'
    (destination / 'logs').mkdir(parents=True)

    def run(*args, **kwargs):
        calls.append(homes[-1])
        raw = json.loads((homes[-1] / 'auth.json').read_text())
        raw['tokens']['access_token'] = 'fresh-test-value'
        save(homes[-1] / 'auth.json', raw)
        if len(calls) == 1:
            (destination / 'logs/codex-1.stderr').write_text('usage limit reached')
            raise subprocess.CalledProcessError(1, 'codex')

    monkeypatch.setattr(runtime, 'run', run)
    runtime.run_with_pool(destination, pool=pool)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert pool.disabled == {'0'}
    assert all(json.loads((Path(a['home']) / 'auth.json').read_text())['tokens']['access_token'] ==
               'fresh-test-value' for a in pool.accounts)


def test_preflight_excludes_assigned_accounts(tmp_path, monkeypatch):
    pool = Pool(tmp_path)
    accounts = [{'id': 'one', 'email': 'available@example.org', 'home': str(tmp_path / 'one')},
                {'id': 'two', 'email': 'busy@example.org', 'home': str(tmp_path / 'two')}]
    for account in accounts:
        Path(account['home']).mkdir()
    monkeypatch.setattr(pool, 'discover', lambda: setattr(pool, 'accounts', accounts))
    monkeypatch.setattr('agent.codex_accounts.hosted_catalog', lambda: {
        'available@example.org': {'id': 'one'}, 'busy@example.org': {'execution_id': 'running'}})
    monkeypatch.setattr(pool, 'inspect', lambda *a, **k: {'remaining': 80})
    result = pool.preflight()
    assert [a['health'] for a in result] == ['ready', 'busy']


def test_discovery_never_copies_excluded_login(tmp_path):
    imports = tmp_path / 'imports'
    imports.mkdir()
    for i, email in enumerate(['quentin.nolan@the.aurafarming.company',
                              'naomi.wright@the.aurafarming.company',
                              'micah.test@example.org', 'mika.test@example.org',
                              'someone@example.org']):
        token = base64.urlsafe_b64encode(json.dumps({'email': email}).encode()).decode().rstrip('=')
        save(imports / f'{i}.auth.json', {'tokens': {'id_token': f'x.{token}.x'}})
    accounts = Pool(tmp_path).discover()
    assert [a['email'] for a in accounts] == ['someone@example.org']
    assert len(list((tmp_path / 'homes').glob('*/auth.json'))) == 1


def test_exclusions_are_case_insensitive_and_keep_other_quentin_accounts():
    assert is_excluded('Naomi.Wright@the.aurafarming.company')
    assert is_excluded('Mika@example.org')
    assert is_excluded('Micah_Test@example.org')
    assert not is_excluded('quentin.radcliffe@the.aurafarming.company')


def test_lease_rechecks_exclusion_before_account_access(tmp_path, monkeypatch):
    pool = Pool(tmp_path)
    pool.accounts = [{'id': 'excluded', 'email': 'mika@example.org',
                      'remaining': 100, 'health': 'ready'}]
    monkeypatch.setattr(pool, '_lock', lambda account: pytest.fail('Excluded account accessed'))
    with pytest.raises(RuntimeError, match='No usable'):
        with pool.lease():
            pytest.fail('Excluded account leased')


def test_two_sessions_per_account_exclude_serial_writers(tmp_path, monkeypatch):
    pool = Pool(tmp_path, sessions_per_account=2)
    account = {'id': 'a', 'home': str(tmp_path), 'health': 'ready', 'remaining': 90}
    pool.accounts = [account]
    monkeypatch.setattr(pool, 'inspect', lambda *a, **k: {'remaining': 90})
    with pool.lease() as first, pool.lease() as second:
        assert first['id'] == second['id']
        assert pool._session_lock(account) is None
        assert pool._lock(account) is None
        assert pool.reset_if_exhausted(account) is None
    serial = Pool(tmp_path)._lock(account)
    assert serial is not None
    assert pool._session_lock(account) is None
    serial.close()
    assert pool.active == {'a': 0}


def test_concurrent_refresh_does_not_overwrite_new_credentials(tmp_path):
    token = base64.urlsafe_b64encode(json.dumps({'email': 'test@example.org'}).encode()).decode().rstrip('=')
    original = {'tokens': {'id_token': f'x.{token}.x', 'access_token': 'original'}}
    newer = {'tokens': {**original['tokens'], 'access_token': 'newer'}}
    stale = {'tokens': {**original['tokens'], 'access_token': 'stale-session'}}
    account = {'home': str(tmp_path), 'email': 'test@example.org'}
    save(tmp_path / 'auth.json', original)
    assert merge_credentials(account, original, newer)
    assert not merge_credentials(account, original, stale)
    assert json.loads((tmp_path / 'auth.json').read_text()) == newer


def test_fourteen_sessions_keep_five_distinct_accounts_and_resume_cleanly(case, tmp_path, monkeypatch):
    import shutil
    cases = [case]
    for i in range(1, 8):
        target = case.parent / f'case-{i}'
        shutil.copytree(case, target)
        cases.append(target)
    pool = Pool(tmp_path / 'pool', sessions_per_account=2)
    for i in range(7):
        home = pool.root / str(i)
        home.mkdir(parents=True)
        pool.accounts.append({'id': str(i), 'home': str(home), 'remaining': 90, 'health': 'ready'})
    monkeypatch.setattr(pool, 'preflight', lambda: pool.accounts)
    monkeypatch.setattr(pool, 'inspect', lambda *a, **k: {'remaining': 90})

    class TrackingJudge(FakeJudge):
        guard = threading.Lock()
        active, peak, per_account, account_peak = 0, 0, {}, {}

        def generate(self, prompt, temperature=0):
            cls = type(self)
            key = self.account['id']
            with cls.guard:
                cls.active += 1
                cls.peak = max(cls.peak, cls.active)
                cls.per_account[key] = cls.per_account.get(key, 0) + 1
                cls.account_peak[key] = max(cls.account_peak.get(key, 0), cls.per_account[key])
            try:
                time.sleep(.08)
                return super().generate(prompt, temperature)
            finally:
                with cls.guard:
                    cls.active -= 1
                    cls.per_account[key] -= 1

    results = judge_cases(cases, pool=pool, jobs=14, backend_factory=TrackingJudge, export=fake_export)
    assert all(r['status'] == 'completed' for r in results)
    assert 7 < TrackingJudge.peak <= 14
    assert max(TrackingJudge.account_peak.values()) == 2
    for target in cases:
        grades = [json.loads(p.read_text()) for p in target.glob('judging/repeats/*/judge.json')]
        assert len(grades) == len({g['account_ref'] for g in grades}) == 5
        assert len(list(target.glob('judging/repeats/*/trajectory.json'))) == 5
        assert not list(target.glob('judging/inflight/*'))
    count = len(TrackingJudge.calls)
    judge_cases(cases, pool=pool, jobs=14, backend_factory=TrackingJudge, export=fake_export)
    assert len(TrackingJudge.calls) == count


def test_rolling_queue_admits_new_case_before_previous_case_finishes(case):
    import shutil
    second = case.parent / 'second'
    shutil.copytree(case, second)
    (second / 'scenario.json').write_text(json.dumps({
        'id': '2', 'text': 'SECOND question', 'characteristic_form': 'Read sensor'}))
    first_started = threading.Event()
    slow_finished = threading.Event()
    overlapped = []
    emitted = []

    class RollingJudge(FakeJudge):
        def generate(self, prompt, temperature=0):
            if 'SECOND question' in prompt:
                overlapped.append(not slow_finished.is_set())
            elif self.account['id'] == '0':
                first_started.set()
                time.sleep(.4)
                slow_finished.set()
            return super().generate(prompt, temperature)

    def source():
        if first_started.is_set() and not emitted:
            emitted.append(second)
            return [second]
        return []

    results = judge_cases([case], pool=FakePool(12), jobs=6,
                         backend_factory=RollingJudge, export=fake_export,
                         case_source=source, source_done=lambda: bool(emitted))
    assert len(results) == 2 and all(r['status'] == 'completed' for r in results)
    assert any(overlapped), 'New case waited for the slow previous judgment'
    for target in (case, second):
        grades = [json.loads(p.read_text()) for p in target.glob('judging/repeats/*/judge.json')]
        assert len(grades) == len({g['account_ref'] for g in grades}) == 5


def test_later_case_finishes_all_five_judgments_while_first_case_is_still_waiting(case):
    import shutil
    second = case.parent / 'second'
    shutil.copytree(case, second)
    (second / 'scenario.json').write_text(json.dumps({
        'id': '2', 'text': 'SECOND question', 'characteristic_form': 'Read sensor'}))
    second_finished = threading.Event()
    completed = []

    class SlowFirstJudge(FakeJudge):
        def generate(self, prompt, temperature=0):
            if 'SECOND question' not in prompt and self.account['id'] == '0':
                assert second_finished.wait(3), 'A five-judge barrier blocked the later case'
            return super().generate(prompt, temperature)

    def on_complete(target, result):
        assert result['status'] == 'completed'
        completed.append(target)
        if target == second:
            second_finished.set()

    # Five slots, fewer than both cases require together: released individual
    # slots must service the next case without waiting for all five in the first.
    results = judge_cases([case, second], pool=FakePool(7), jobs=5,
                         backend_factory=SlowFirstJudge, export=fake_export,
                         on_complete=on_complete)
    assert all(result['status'] == 'completed' for result in results)
    assert completed == [second, case]
    for target in (case, second):
        grades = [json.loads(p.read_text()) for p in target.glob('judging/repeats/*/judge.json')]
        assert len(grades) == len({g['account_ref'] for g in grades}) == 5
        assert len(list(target.glob('judging/repeats/*/trajectory.json'))) == 5
