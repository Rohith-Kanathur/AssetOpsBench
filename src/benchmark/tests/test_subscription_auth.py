"""A disposable judge must never consume the host's refresh token."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.generated import auth


@pytest.fixture
def store(tmp_path, monkeypatch):
    config = tmp_path / 'host'
    config.mkdir()
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(config))
    monkeypatch.delenv('CLAUDE_CODE_OAUTH_TOKEN', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(auth.time, 'time', lambda: 1000)
    return config / '.credentials.json'


def credential(expiry, token='access'):
    return {'claudeAiOauth': {'accessToken': token, 'refreshToken': 'refresh-authority', 'expiresAt': expiry}}


def test_valid_access_is_copied_without_refresh_authority(store, tmp_path, monkeypatch):
    original = credential(5_000_000)
    store.write_text(json.dumps(original))
    monkeypatch.setattr(auth.subprocess, 'run', lambda *a, **k: pytest.fail('Unnecessary refresh'))
    target = tmp_path / 'isolated'
    auth.prepare_auth(target, 'claude')
    copied = json.loads((target / '.claude/.credentials.json').read_text())
    assert copied['claudeAiOauth']['accessToken'] == 'access'
    assert 'refreshToken' not in copied['claudeAiOauth']
    assert json.loads(store.read_text()) == original
    assert (target / '.claude/.credentials.json').stat().st_mode & 0o777 == 0o600


def test_expired_token_refreshes_on_host_before_copy(store, tmp_path, monkeypatch):
    store.write_text(json.dumps(credential(500_000)))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'do-not-use')
    calls = []
    def refresh(command, **kwargs):
        calls.append(command)
        assert 'ANTHROPIC_API_KEY' not in kwargs['env']
        assert 'refresh-authority' not in command
        assert kwargs['capture_output'] is True
        store.write_text(json.dumps(credential(5_000_000, 'new-access')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(auth.subprocess, 'run', refresh)
    auth.prepare_auth(tmp_path / 'isolated', 'claude')
    assert len(calls) == 1
    copied = json.loads((tmp_path / 'isolated/.claude/.credentials.json').read_text())
    assert copied['claudeAiOauth']['accessToken'] == 'new-access'
    assert 'refreshToken' not in copied['claudeAiOauth']
    assert json.loads(store.read_text())['claudeAiOauth']['refreshToken'] == 'refresh-authority'


def test_failed_refresh_does_not_copy_expired_credentials(store, tmp_path, monkeypatch):
    store.write_text(json.dumps(credential(500_000)))
    monkeypatch.setattr(auth.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match='claude auth login'):
        auth.prepare_auth(tmp_path / 'isolated', 'claude')
    assert not (tmp_path / 'isolated/.claude/.credentials.json').exists()
