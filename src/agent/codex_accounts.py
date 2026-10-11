"""Private subscription pool shared by serial authoring and parallel judging.

Credentials are imported out of band from the owner's Everett account. Nothing
in this module purchases credits or changes the desktop application's login.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from uuid import uuid4

ROOT = Path.home() / '.config/assetopsbench/codex-pool'
BINARY = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex'
EXCLUDED = {'quentin.nolan@the.aurafarming.company', 'naomi.wright@the.aurafarming.company'}
EXCLUDED_NAMES = {'naomi', 'mika', 'micah'}
MODEL, REASONING, TIER = 'gpt-6-astra', 'xhigh', 'fast'


def is_excluded(email):
    email = email.strip().lower()
    first_name = re.split(r'[._+\-\s]', email.split('@', 1)[0])[0]
    return email in EXCLUDED or first_name in EXCLUDED_NAMES


def hosted_catalog():
    """Use only Sagar's personal hosted key, never a Doppler fallback."""
    from dotenv import dotenv_values
    config = Path.home() / '.config/everett/hosted.env'
    if config.stat().st_mode & 0o777 != 0o600:
        raise RuntimeError('Personal Everett configuration must have mode 0600')
    values = dotenv_values(config)
    if not values.get('EVERETT_API_URL') or not values.get('EVERETT_API_KEY'):
        raise RuntimeError('Personal Everett credentials are unavailable')
    env = os.environ.copy()
    env.update({key: values[key] for key in ('EVERETT_API_URL', 'EVERETT_API_KEY')})
    result = subprocess.run(['uv', 'run', 'everett', 'subscription', 'list', '--json'],
        cwd=Path.home() / 'Documents/everett', env=env, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Everett subscription catalog is unavailable')
    return {item['label'].lower(): item for item in json.loads(result.stdout)['items']
            if not is_excluded(item['label'])}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    temporary.replace(path)


def identity(path):
    raw = json.loads(Path(path).read_text())
    token = raw['tokens']['id_token'].split('.')[1]
    claims = json.loads(base64.urlsafe_b64decode(token + '=' * (-len(token) % 4)))
    return claims['email'].lower(), raw


@contextmanager
def credential_lock(home):
    """Serialize metadata refreshes and credential writes, not model sessions."""
    with (Path(home) / 'credentials.lock').open('a+') as handle:
        os.chmod(handle.name, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def credential_snapshot(account):
    with credential_lock(account['home']):
        email, raw = identity(Path(account['home']) / 'auth.json')
        if email != account['email']:
            raise RuntimeError('Subscription login identity changed unexpectedly')
        return raw


def merge_credentials(account, original, refreshed):
    """Never let a finishing session overwrite another session's refresh."""
    source = Path(account['home']) / 'auth.json'
    with credential_lock(account['home']):
        email, current = identity(source)
        if email != account['email']:
            raise RuntimeError('Subscription login identity changed unexpectedly')
        if refreshed != original and current.get('tokens') == original.get('tokens'):
            save(source, refreshed)
            return True
        return False


class SessionLease:
    def __init__(self, gate, slot):
        self.gate, self.slot = gate, slot

    def close(self):
        self.slot.close()
        self.gate.close()


def environment(home):
    env = {k: v for k, v in os.environ.items() if not any(
        marker in k for marker in ('API_KEY', 'AUTH_TOKEN', 'ACCESS_TOKEN'))}
    env.update(CODEX_HOME=str(home))
    return env


@contextmanager
def rpc(home):
    """Only account/model metadata calls; creating this session runs no model."""
    process = subprocess.Popen([BINARY, '-c', 'cli_auth_credentials_store="file"',
                                'app-server', '--stdio'], env=environment(home),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    messages = queue.Queue()

    def read():
        for line in process.stdout:
            try:
                messages.put(json.loads(line))
            except ValueError:
                pass
        messages.put(None)

    threading.Thread(target=read, daemon=True).start()
    sequence = 0

    def call(method, params=None):
        nonlocal sequence
        sequence += 1
        request = {'id': sequence, 'method': method}
        if params is not None:
            request['params'] = params
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        end = time.monotonic() + 30
        while time.monotonic() < end:
            try:
                response = messages.get(timeout=max(.01, end - time.monotonic()))
            except queue.Empty:
                break
            if response is None:
                break
            if response.get('id') == sequence:
                if 'error' in response:
                    # Do not echo server errors that could include authentication data.
                    raise RuntimeError(f'Codex metadata request failed: {method}')
                return response['result']
        raise RuntimeError(f'Codex metadata request unavailable: {method}')

    try:
        call('initialize', {'clientInfo': {'name': 'assetops-pool', 'version': '1.0'}})
        process.stdin.write('{"method":"initialized","params":{}}\n')
        process.stdin.flush()
        yield call
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()


def allowance(data):
    core = (data.get('rateLimitsByLimitId') or {}).get('codex') or data.get('rateLimits') or {}
    windows = [{**w, 'remaining': max(0, 100 - w['usedPercent'])}
               for name in ('primary', 'secondary') if (w := core.get(name))
               and w.get('usedPercent') is not None]
    return {'windows': windows, 'remaining': min((w['remaining'] for w in windows), default=None),
            'resets': (data.get('rateLimitResetCredits') or {}).get('availableCount'),
            'plan': core.get('planType')}


class Pool:
    def __init__(self, root=ROOT, *, model=MODEL, reasoning=REASONING, tier=TIER,
                 sessions_per_account=1):
        if sessions_per_account < 1:
            raise ValueError('Sessions per account must be positive')
        self.root = Path(root).expanduser().resolve()
        self.model, self.reasoning, self.tier = model, reasoning, tier
        self.sessions_per_account = sessions_per_account
        self.accounts = []
        self.disabled = set()
        self.guard = threading.Lock()
        self.active = {}

    def discover(self):
        """Only explicitly imported credentials, never the active app account."""
        accounts = {}
        for source in sorted((self.root / 'imports').glob('*.auth.json')):
            email, raw = identity(source)
            if is_excluded(email):
                continue
            key = hashlib.sha256(email.encode()).hexdigest()[:20]
            home = self.root / 'homes' / key
            home.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = home / 'auth.json'
            # Retain newer local refreshes. No active login is overwritten.
            if not target.exists():
                save(target, raw)
            elif identity(target)[0] != email:
                raise ValueError('Private pool identity mismatch')
            accounts[key] = {'id': key, 'email': email, 'home': str(home)}
        self.accounts = list(accounts.values())
        return self.accounts

    def _lock(self, account):
        handle = (Path(account['home']) / 'lease.lock').open('a+')
        os.chmod(handle.name, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
            return None

    def _session_lock(self, account):
        if self.sessions_per_account == 1:
            return self._lock(account)
        gate = (Path(account['home']) / 'lease.lock').open('a+')
        os.chmod(gate.name, 0o600)
        try:
            # Shared session gates exclude serial authoring, maintenance and resets.
            fcntl.flock(gate, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            gate.close()
            return None
        for index in range(self.sessions_per_account):
            slot = (Path(account['home']) / f'session-{index}.lock').open('a+')
            os.chmod(slot.name, 0o600)
            try:
                fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return SessionLease(gate, slot)
            except BlockingIOError:
                slot.close()
        gate.close()
        return None

    def inspect(self, account, *, check_model=True):
        with credential_lock(account['home']):
            return self._inspect(account, check_model=check_model)

    def _inspect(self, account, *, check_model=True):
        with rpc(account['home']) as call:
            info = call('account/read', {'refreshToken': False}).get('account') or {}
            if info.get('email', '').lower() != account['email'] or info.get('type') != 'chatgpt':
                raise ValueError('Expected subscription login identity')
            data = allowance(call('account/rateLimits/read'))
            if check_model:
                models = call('model/list', {'includeHidden': True}).get('data', [])
                selected = next((m for m in models if m.get('id') == self.model or m.get('model') == self.model), None)
                if not selected or self.reasoning not in {x['reasoningEffort'] for x in selected.get('supportedReasoningEfforts', [])}:
                    raise ValueError('Requested model/reasoning is unavailable')
                tiers = set(selected.get('additionalSpeedTiers', [])) | {x['id'] for x in selected.get('serviceTiers', [])}
                if self.tier and self.tier not in tiers and not (self.tier == 'fast' and 'priority' in tiers):
                    raise ValueError('Requested speed tier is unavailable')
            data['checked_at'] = time.time()
            save(Path(account['home']) / 'usage.json', data)
            return data

    def preflight(self):
        self.discover()
        catalog = hosted_catalog()

        def check(account):
            hosted = catalog.get(account['email'])
            # A hosted stale-token probe may lag a newer, locally refreshed login.
            # Validate that login below; other hosted errors remain unavailable.
            if (not hosted or hosted.get('needs_login') or
                    hosted.get('usage_error') not in (None, 'auth_stale')):
                return {**account, 'health': 'unavailable', 'error_type': 'HostedLoginUnavailable'}
            if hosted.get('execution_id') or hosted.get('thread_id'):
                return {**account, 'health': 'busy', 'busy_reason': 'hosted_assignment'}
            lock = self._lock(account)
            if lock is None:
                return {**account, 'health': 'busy', 'busy_reason': 'local_lease'}
            try:
                usage = self.inspect(account)
                return {**account, **usage, 'health': 'ready' if usage['remaining'] and usage['remaining'] > 0 else 'exhausted'}
            except Exception as exc:
                return {**account, 'health': 'unavailable', 'error_type': type(exc).__name__}
            finally:
                lock.close()

        with ThreadPoolExecutor(max_workers=4) as executor:
            self.accounts = list(executor.map(check, self.accounts))
        save(self.root / 'status.json', {'at': time.time(), 'accounts': self.accounts,
                                      'excluded': sorted(EXCLUDED), 'excluded_names': sorted(EXCLUDED_NAMES)})
        return self.accounts

    def reset_if_exhausted(self, account):
        """Redeem only an existing reset at 0%, under this account's lease.

        An uncertain RPC keeps the same idempotency key for the next attempt.
        This method is never called by preflight.
        """
        maintenance = self._lock(account) if self.sessions_per_account > 1 else None
        if self.sessions_per_account > 1 and maintenance is None:
            # Another live session must finish before an account-wide reset.
            return None
        try:
            with credential_lock(account['home']):
                return self._reset_if_exhausted(account)
        finally:
            if maintenance is not None:
                maintenance.close()

    def _reset_if_exhausted(self, account):
        path = Path(account['home']) / 'reset.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        with rpc(account['home']) as call:
            usage = allowance(call('account/rateLimits/read'))
            if usage['remaining'] is None:
                return False
            if usage['remaining'] > 0:
                state.pop('pending', None)
                state['awaiting_clear'] = False
                save(path, state)
                return True
            if not state.get('pending'):
                if state.get('awaiting_clear') or not usage['resets']:
                    return False
                state['pending'] = str(uuid4())
                save(path, state)
            response = call('account/rateLimitResetCredit/consume', {'idempotencyKey': state['pending']})
            outcome = response.get('outcome')
            if outcome not in {'reset', 'alreadyRedeemed', 'nothingToReset', 'noCredit'}:
                raise RuntimeError('Unknown reset outcome; retaining idempotency key')
            state.update(pending=None, awaiting_clear=outcome in {'reset', 'alreadyRedeemed'}, last_outcome=outcome)
            save(path, state)
            usage = allowance(call('account/rateLimits/read'))
            return usage['remaining'] is not None and usage['remaining'] > 0

    @contextmanager
    def lease(self, *, exclude=(), allow_reset=True, wait_seconds=120):
        deadline = time.monotonic() + wait_seconds
        while True:
            busy = False
            with self.guard:
                candidates = sorted(self.accounts, key=lambda a:
                    (self.active.get(a['id'], 0), -(a.get('remaining') or 0)))
            for account in candidates:
                if is_excluded(account.get('email', '')):
                    continue
                if account['id'] in set(exclude) | self.disabled or account.get('health') not in {'ready', 'exhausted'}:
                    continue
                lock = self._session_lock(account)
                if lock is None:
                    busy = True
                    continue
                with self.guard:
                    self.active[account['id']] = self.active.get(account['id'], 0) + 1
                try:
                    try:
                        # Metadata is shared only; prompts, sessions and auth homes remain separate.
                        # Rechecking before every 40-second judgment serialized admissions.
                        if (account.get('remaining', 0) > 0 and
                                time.time() - account.get('checked_at', 0) < 30):
                            usage = account
                        else:
                            usage = self.inspect(account, check_model=False)
                            account.update(usage, checked_at=time.time())
                        if usage['remaining'] is None or usage['remaining'] <= 0:
                            if self.sessions_per_account > 1:
                                lock.close()
                                lock = None
                                recovery = self.reset_if_exhausted(account) if allow_reset else False
                                busy = busy or recovery is not False
                                continue
                            if not allow_reset or not self.reset_if_exhausted(account):
                                continue
                    except Exception:
                        self.quarantine(account)
                        continue
                    yield account
                    return
                finally:
                    if lock is not None:
                        lock.close()
                    with self.guard:
                        self.active[account['id']] -= 1
            if not busy or time.monotonic() >= deadline:
                raise RuntimeError('No usable distinct Codex subscription is available')
            time.sleep(.2)

    def quarantine(self, account):
        with self.guard:
            self.disabled.add(account['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check'])
    parser.add_argument('--pool', type=Path, default=ROOT)
    args = parser.parse_args()
    rows = Pool(args.pool).preflight()
    print(json.dumps([{k: v for k, v in a.items() if k != 'home'} for a in rows], indent=2))


if __name__ == '__main__':
    main()
