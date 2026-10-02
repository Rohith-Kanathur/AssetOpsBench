import json

from scenarios.generation.harnesses import codex_command
from scenarios.generation.runtime import configure


def test_container_does_not_mount_reference_checkout_or_docker_socket(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    (target / "baseline.json").write_text("{}")
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    path = configure(target, auth, kaggle)
    data = json.loads(path.read_text())
    mounts = data["services"]["agent"]["volumes"]
    assert len(mounts) == 3
    assert all("docker.sock" not in mount for mount in mounts)
    assert mounts[1].endswith("auth.json:ro")
    assert mounts[2].endswith(".kaggle:ro")
    assert "ports" not in data["services"]["database"]
    original = path.read_text()
    configure(target, auth, kaggle)
    assert path.read_text() == original


def test_native_tools_and_mcp_are_available():
    command = codex_command()
    assert 'web_search="live"' in command
    assert "--ignore-user-config" in command
    assert not any("shell_tool=false" in arg for arg in command)
    assert 'mcp_servers.fmsr.args=["-m","servers.fmsr.main"]' in command
    assert command[command.index("--model") + 1] == "gpt-6-astra"
    assert 'model_reasoning_effort="xhigh"' in command
    assert 'service_tier="fast"' in command


def test_explicit_model_settings_override_defaults():
    command = codex_command("gpt-6-sol", "high", "default")
    assert command[command.index("--model") + 1] == "gpt-6-sol"
    assert 'model_reasoning_effort="high"' in command
    assert 'service_tier="default"' in command


def test_existing_database_is_not_reseeded(tmp_path, monkeypatch):
    from scenarios.generation import runtime

    calls = []
    monkeypatch.setattr(runtime, "ensure_image", lambda: None)
    monkeypatch.setattr(runtime, "compose", lambda *args, **kwargs: calls.append(args))
    runtime.start(tmp_path)
    runtime.start(tmp_path)
    seed_calls = [args for args in calls if "couchdb.init_data" in args]
    assert len(seed_calls) == 1
    assert "--reset" not in seed_calls[0]


def test_failed_initialization_is_not_marked_complete(tmp_path, monkeypatch):
    import subprocess
    import pytest
    from scenarios.generation import runtime

    def fail_seed(*args, **kwargs):
        if "couchdb.init_data" in args:
            raise subprocess.CalledProcessError(1, "seed")

    monkeypatch.setattr(runtime, "ensure_image", lambda: None)
    monkeypatch.setattr(runtime, "compose", fail_seed)
    with pytest.raises(subprocess.CalledProcessError):
        runtime.start(tmp_path)
    assert not (tmp_path / "initialized.json").exists()
