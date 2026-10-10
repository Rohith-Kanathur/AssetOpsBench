"""Exercise native process boundaries using fake CLIs, without model calls."""

import importlib
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest

# The minimal execution image mounts this package without the SDK-heavy agent/__init__.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
commands = importlib.import_module("coding_agent.commands")
runner = importlib.import_module("coding_agent.runner")
trajectory = importlib.import_module("coding_agent.trajectory")
zcode = importlib.import_module("coding_agent.zcode")


def test_codex_auth_only_and_http_config(tmp_path):
    source, home, workspace = (tmp_path / part for part in ("source", "home", "workspace"))
    for path in (source, home, workspace):
        path.mkdir()
    (source / "auth.json").write_text('{"tokens":{"access_token":"test-token-123"}}')
    (source / "AGENTS.md").write_text("Unwanted instructions")
    (source / "config.toml").write_text("Bad user settings")
    servers = {"iot": {"type": "http", "url": "http://tools:8100/mcp",
                       "headers": {"Authorization": "Bearer test-secret"}},
               "vibration": {"command": "python", "args": ["-m", "servers.vibration.main"]}}
    command, env = commands.prepare("codex", home, workspace, servers, model="gpt-6-astra",
                                   reasoning_effort="xhigh", service_tier="fast",
                                   auth_home=source, environment={"HOME": str(source)})
    config = tomllib.loads((home / ".codex/config.toml").read_text())
    assert config["mcp_servers"]["iot"]["http_headers"]["Authorization"] == "Bearer test-secret"
    assert config["mcp_servers"]["vibration"]["args"][-1] == "servers.vibration.main"
    assert config["project_doc_max_bytes"] == 0
    assert config["projects"][str(workspace)]["trust_level"] == "untrusted"
    assert (home / ".codex/auth.json").is_file()
    assert not (home / ".codex/AGENTS.md").exists()
    assert "test-secret" not in " ".join(command)
    assert env["CODEX_HOME"] == str(home / ".codex")
    assert (home / ".codex/config.toml").stat().st_mode & 0o777 == 0o600


def test_claude_preserves_subscription_without_settings(tmp_path):
    home, source = tmp_path / "home", tmp_path / "auth"
    home.mkdir()
    source.mkdir()
    (source / ".credentials.json").write_text('{"claudeAiOauth":{"accessToken":"oauth-test-token"}}')
    (source / "settings.json").write_text('{"hooks":{"bad":[]}}')
    command, env = commands.prepare("claude", home, tmp_path, {"iot": {"type": "http", "url": "http://tools/mcp"}},
                                   model="fable", reasoning_effort="high", service_tier="fast",
                                   auth_home=source, environment={"CLAUDE_CODE_OAUTH_TOKEN": "oauth-test-token"})
    assert "--bare" not in command
    assert command[command.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in command
    assert "--disable-slash-commands" in command
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-test-token"
    assert (home / ".claude/.credentials.json").is_file()
    assert not (home / ".claude/settings.json").exists()


def test_zai_uses_api_auth_without_claude_subscription(tmp_path):
    home, source = tmp_path / "home", tmp_path / "auth"
    home.mkdir()
    source.mkdir()
    (source / ".credentials.json").write_text('{"accessToken":"personal-subscription"}')
    command, env = commands.prepare("claude", home, tmp_path, {}, model="zai/glm-5.3",
                                   reasoning_effort="high", service_tier="fast", auth_home=source,
                                   environment={"ZAI_API_KEY": "test-zai-key",
                                                "CLAUDE_CODE_OAUTH_TOKEN": "personal-oauth",
                                                "ANTHROPIC_API_KEY": "personal-api-key"})
    assert command[command.index("--model") + 1] == "glm-5.3"
    assert "test-zai-key" not in command
    assert env["ANTHROPIC_AUTH_TOKEN"] == "test-zai-key"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env and "ANTHROPIC_API_KEY" not in env
    assert not (home / ".claude/.credentials.json").exists()


def test_zai_requires_its_own_key(tmp_path):
    with pytest.raises(ValueError, match="ZAI_API_KEY"):
        commands.prepare("claude", tmp_path, tmp_path, {}, model="zai/glm-5.3",
                         reasoning_effort="high", service_tier="fast", auth_home=None,
                         environment={"CLAUDE_CODE_OAUTH_TOKEN": "personal-oauth"})


def _events(*events):
    return "\n".join(json.dumps(event) for event in events)


def test_native_event_normalization():
    codex = trajectory.parse(_events(
        {"type": "item.completed", "item": {"id": "i1", "type": "mcp_tool_call", "server": "iot",
         "tool": "history", "arguments": {"asset": "T1"}, "result": {"values": [1, 2]}}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Answer"}},
        {"type": "turn.completed", "usage": {"input_tokens": 15}}), "codex")
    assert codex["completed"] and codex["answer"] == "Answer"
    assert codex["trajectory"]["turns"][0]["tool_calls"][0]["name"] == "iot.history"
    claude = trajectory.parse(_events(
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "c1", "content": "file.csv"}]}},
        {"type": "result", "is_error": False, "result": "Saved", "usage": {"output_tokens": 5}}), "claude")
    assert claude["completed"] and claude["answer"] == "Saved"
    assert claude["trajectory"]["turns"][0]["tool_calls"][0]["output"] == "file.csv"


def _fake_cli(tmp_path, body, harness="codex"):
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    program = directory / harness
    program.write_text(f"#!{sys.executable}\n" + body)
    program.chmod(0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    env = {"PATH": str(directory) + os.pathsep + os.environ["PATH"],
           "HOME": str(tmp_path / "unconfigured"), "ASSETOPS_EXECUTION_ISOLATED": "1"}
    return workspace, env


def test_run_redacts_credentials_and_captures_artifact(tmp_path):
    workspace, env = _fake_cli(tmp_path, """import json, os, sys
from pathlib import Path
assert 'Request:' in sys.stdin.read()
Path('export.json').write_text('[1,2]')
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'Saved export.json '+os.environ['MY_API_KEY']}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':3}}))
print(os.environ['MY_API_KEY'], file=sys.stderr)
""")
    env["MY_API_KEY"] = "private-test-token"
    result = runner.run("Export readings", harness="codex", workspace=workspace,
                        mcp_servers={}, output_dir=tmp_path / "results", model="test", environment=env)
    assert result["status"] == "completed"
    assert (workspace / "export.json").read_text() == "[1,2]"
    for path in (tmp_path / "results").iterdir():
        assert "private-test-token" not in path.read_text()
    assert result["answer"] == "Saved export.json [REDACTED]"
    with pytest.raises(ValueError, match="already contains"):
        runner.run("Again", harness="codex", workspace=workspace, mcp_servers={},
                   output_dir=tmp_path / "results", model="test", environment=env)


@pytest.mark.parametrize("body,expected", [
    ("import sys; sys.exit(2)", "exited with status 2"),
    ("print('{}')", "did not report a completed turn"),
    ("import time; time.sleep(10)", "timed out"),
    ("print('{\"type\":\"turn.failed\",\"error\":\"over quota\"}')", "over quota"),
])
def test_run_failure_and_timeout(tmp_path, body, expected):
    workspace, env = _fake_cli(tmp_path, body)
    result = runner.run("Read sensors", harness="codex", workspace=workspace, mcp_servers={},
                        output_dir=tmp_path / "results", model="test", environment=env, timeout=0.2)
    assert result["status"] == "error"
    assert expected in result["error"]
    assert (tmp_path / "results/result.json").is_file()


def test_refuses_unisolated_run(tmp_path):
    with pytest.raises(ValueError, match="isolated container"):
        runner.run("Read sensors", harness="codex", workspace=tmp_path, mcp_servers={},
                   output_dir=tmp_path / "results", model="test", environment={})


def test_missing_cli_reports_launch_error(tmp_path):
    result = runner.run("Read sensors", harness="claude", workspace=tmp_path, mcp_servers={},
                        output_dir=tmp_path / "results", model="test",
                        environment={"ASSETOPS_EXECUTION_ISOLATED": "1", "PATH": str(tmp_path)})
    assert result["status"] == "error"
    assert result["error"].startswith("Could not start claude")


def test_refuses_inherited_instructions(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Influence an answer")
    with pytest.raises(ValueError, match="inherit repository"):
        runner.run("Read sensors", harness="claude", workspace=tmp_path, mcp_servers={},
                   output_dir=tmp_path / "results", model="test",
                   environment={"ASSETOPS_EXECUTION_ISOLATED": "1"})


def test_zcode_native_process_private_config_and_complete_evidence(tmp_path, monkeypatch):
    workspace, env = _fake_cli(tmp_path, """import json, os, sys
from pathlib import Path
assert 'Request:' in sys.argv[sys.argv.index('--prompt')+1]
assert 'ZAI_API_KEY' not in os.environ and 'ANTHROPIC_API_KEY' not in os.environ
assert 'ZCODE_UNTRUSTED_PLUGIN' not in os.environ
assert os.environ['ZCODE_BFS_BINARY'] == '/opt/zcode/runtime-tools/bfs'
assert os.environ['ZCODE_UGREP_BINARY'] == '/opt/zcode/runtime-tools/ugrep'
assert os.environ['ZCODE_RG_BINARY'] == '/opt/zcode/runtime-tools/rg'
provider = Path(os.environ['ZCODE_PERSONAL_PROVIDER_CONFIG_FILE'])
assert provider.stat().st_mode & 0o777 == 0o600
config = json.loads(provider.read_text())['config']
access = config['providerConfigRules']['providerRules'][0]['config']['access']
assert access['type'] == 'zhipu-coding-plan-api-key'
assert config['defaultModelSelection']['modelId'] == 'GLM-5.3'
assert config['defaultModelSelection']['options']['reasoningLevel'] == 'high'
assert not list(Path(os.environ['HOME']).glob('**/auth.json'))
settings = json.loads((Path(os.environ['HOME'])/'.zcode/cli/config.json').read_text())
assert settings['plugins']['enabled'] is False and settings['hooks']['enabled'] is False
assert settings['mcp']['servers']['iot']['url'] == 'http://tools/mcp'
artifacts = Path(os.environ['HOME'])/'.zcode/state/cli/artifacts'
artifacts.mkdir(parents=True)
(artifacts/'t1.json').write_text(json.dumps({'readings':[1,2,3], 'apiKey':access['apiKey']}))
(artifacts/'credentials.json').symlink_to(provider)
Path('readings.csv').write_text('OT\\n42\\n')
for event in [
 {'type':'tool.updated','payload':{'kind':'scheduled','toolCallId':'t1','toolName':'mcp__iot__history','input':{'asset_id':'T1'}}},
 {'type':'tool.updated','payload':{'kind':'result','toolCallId':'t1','result':{'success':True,'content':'42 '+access['apiKey']}}},
 {'type':'tool.updated','payload':{'kind':'scheduled','toolCallId':'t2','toolName':'Bash','input':{'command':'cat readings.csv'}}},
 {'type':'tool.updated','payload':{'kind':'error','toolCallId':'t2','error':{'message':'recoverable'}}},
 {'type':'model.streaming','payload':{'kind':'text_delta','assistantMessageId':'m1','delta':'Saved readings.csv'}},
 {'type':'result','response':'Saved readings.csv','usage':{'inputTokens':25},'projection':{'status':'idle'}}]:
 print(json.dumps(event))
print(access['apiKey'], file=sys.stderr)
""", harness="zcode")
    monkeypatch.setattr(zcode, "NODE", str(tmp_path / "bin/zcode"))
    env.update(ZAI_API_KEY="private-zcode-test-key", ANTHROPIC_API_KEY="personal-anthropic-key",
               ZCODE_STORAGE_DIR="/personal/state", ZCODE_DATA_BASE_DIR="/personal/home",
               ZCODE_UNTRUSTED_PLUGIN="/personal/plugin", ZCODE_RG_BINARY="/personal/rg")
    result = runner.run("Save sensor values", harness="zcode", workspace=workspace,
                        mcp_servers={"iot": {"type": "http", "url": "http://tools/mcp"}},
                        output_dir=tmp_path / "results", model="GLM-5.3", environment=env)
    assert result["status"] == "completed" and result["usage"] == {"inputTokens": 25}
    calls = [call for turn in result["trajectory"]["turns"] for call in turn["tool_calls"]]
    assert calls[0]["input"] == {"asset_id": "T1"}
    assert calls[0]["output"] == {"success": True, "content": "42 [REDACTED]"}
    assert calls[1]["status"] == "error" and calls[1]["output"] == {"message": "recoverable"}
    assert (workspace / "readings.csv").read_text() == "OT\n42\n"
    assert all("private-zcode-test-key" not in p.read_text()
               for p in (tmp_path / "results").rglob("*") if p.is_file())
    assert len(result["native_tool_artifacts"]) == 1
    artifact = result["native_tool_artifacts"][0]
    assert json.loads(artifact["content"]) == {"readings": [1, 2, 3], "apiKey": "[REDACTED]"}
    assert (tmp_path / "results" / artifact["path"]).read_text() == artifact["content"]
    assert len((tmp_path / "results/stdout.jsonl").read_text().splitlines()) == 6


def test_zcode_requires_key_and_rejects_inherited_guidance(tmp_path):
    with pytest.raises(ValueError, match="ZAI_API_KEY"):
        commands.prepare("zcode", tmp_path, tmp_path, {}, model="GLM-5.3", reasoning_effort="high",
                         service_tier="fast", auth_home=None, environment={})
    (tmp_path / "AGENTS.md").write_text("Inherited instructions")
    with pytest.raises(ValueError, match="inherit repository"):
        runner.run("Read", harness="zcode", workspace=tmp_path, output_dir=tmp_path / "results",
                   mcp_servers={}, model="GLM-5.3", environment={"ASSETOPS_EXECUTION_ISOLATED": "1"})
    parsed = trajectory.parse(_events(
        {"type": "turn.failed", "payload": {"error": {"message": "over quota"}}},
        {"type": "result", "response": "Failed", "projection": {"status": "error"}}), "zcode")
    assert not parsed["completed"] and parsed["error"]


def test_zcode_native_error_retains_provider_reason(tmp_path, monkeypatch):
    workspace, env = _fake_cli(tmp_path, """import json
print(json.dumps({'type':'turn.failed','payload':{'error':{'message':'over quota'}}}))
print(json.dumps({'type':'result','response':'Stopped','projection':{'status':'error'}}))
""", harness="zcode")
    monkeypatch.setattr(zcode, "NODE", str(tmp_path / "bin/zcode"))
    env["ZAI_API_KEY"] = "private-plan-test-key"
    result = runner.run("Read sensors", harness="zcode", workspace=workspace,
                        mcp_servers={}, output_dir=tmp_path / "results", model="GLM-5.3", environment=env)
    assert result["status"] == "error" and result["error"] == {"message": "over quota"}
    assert (tmp_path / "results/stdout.jsonl").is_file()
