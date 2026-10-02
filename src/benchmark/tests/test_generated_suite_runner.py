"""Generated suites cannot silently omit negatives or mix model results."""
import json
from pathlib import Path

import pytest

from benchmark import generated_suite_runner as runner


@pytest.fixture
def suite(tmp_path):
    root = tmp_path / "generated"
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"status": "complete", "negative_count": 1,
        "config": {"num_scenarios": 1, "num_negative_scenarios": 1}}))
    for filename, identifier in [("scenarios.json", "positive"), ("negative_scenarios.json", "negative")]:
        (root / filename).write_text(json.dumps([{"id": identifier, "text": f"Question {identifier}"}]))
    return root


def test_both_scenario_kinds_run_and_resume_without_mixing_models(suite, tmp_path, monkeypatch):
    calls = []

    def child(command, *, env, **kwargs):
        sid = command[command.index("--scenario-id") + 1]
        rid = command[command.index("--run-id") + 1]
        model = command[command.index("--model-id") + 1]
        calls.append((model, sid))
        path = Path(env["AGENT_TRAJECTORY_DIR"]) / f"{rid}.json"
        path.write_text(json.dumps({"run_id": rid, "scenario_id": sid, "runner": "openai-agent",
            "model": model, "question": command[-1], "answer": "answer", "trajectory": []}))

    def invoke(command, *, env, record, record_path, **kwargs):
        child(command, env=env)
        record.update(status='completed', execution_start=None, execution_end=None, execution_duration_ms=None)
    monkeypatch.setattr(runner, "invoke", invoke)
    output = tmp_path / "results"
    runner.run_target(suite, output, "glm", "openai", "zai/glm-5.3")
    runner.run_target(suite, output, "glm", "openai", "zai/glm-5.3")
    assert calls == [("zai/glm-5.3", "positive"), ("zai/glm-5.3", "negative")]
    # A trajectory saved before a failed process exit must not masquerade as a completed run.
    measurement=output/'glm/measurements/glm_0002.json'
    data=json.loads(measurement.read_text());data['status']='failed';measurement.write_text(json.dumps(data))
    runner.run_target(suite, output, "glm", "openai", "zai/glm-5.3")
    assert calls[-1]==("zai/glm-5.3","negative")
    assert (output/'glm/failed_trajectories/glm_0002.attempt-1.json').exists()
    with pytest.raises(ValueError, match="different model"):
        runner.run_target(suite, output, "glm", "openai", "gpt-6-astra")
    assert len(calls) == 3


def test_incomplete_or_inconsistent_generation_never_launches_agents(suite, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "invoke", lambda *args, **kwargs: pytest.fail("must not launch"))
    manifest = suite / "run.json"
    data = json.loads(manifest.read_text())
    data["status"] = "running"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete"):
        runner.run_target(suite, tmp_path / "results", "opus", "claude", "claude-opus-5-5")
    data["status"] = "complete"
    manifest.write_text(json.dumps(data))
    (suite / "negative_scenarios.json").write_text("[]")
    with pytest.raises(ValueError, match="counts"):
        runner.run_target(suite, tmp_path / "results", "opus", "claude", "claude-opus-5-5")
    assert not (tmp_path / "results").exists()


def test_quota_recovery_retains_completed_cases_attempts_and_settings(suite,tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'versions',lambda _: {'sdk_version':'fixture','cli_version':'fixture'})
    output=tmp_path/'out'; calls=[]
    def invoke(command, *, env, record, **kwargs):
        calls.append(record['scenario_id'])
        if record['scenario_id']=='negative' and record['attempt']<=3:
            record.update(status='failed',agent_error={'message':"You've hit your session limit"},execution_start=None)
            # The runner finally retains the error from trace events.
            Path(env['AGENT_TRACE_FILE']).parent.mkdir(parents=True,exist_ok=True)
            Path(env['AGENT_TRACE_FILE']).write_text(json.dumps({'kind':'run_error','error':{'message':"You've hit your session limit"}})+'\n')
            raise RuntimeError('quota')
        record.update(status='completed',execution_start=None)
        Path(env['AGENT_TRAJECTORY_DIR'],record['run_id']+'.json').write_text(json.dumps({'run_id':record['run_id'],'scenario_id':record['scenario_id'],'runner':'claude-agent','model':'claude-opus-5-5','question':command[-1],'answer':'OK','trajectory':{'turns':[]}}))
    monkeypatch.setattr(runner,'invoke',invoke)
    for _ in range(3):
        with pytest.raises(RuntimeError,match='quota'):
            runner.run_target(suite,output,'opus','claude','claude-opus-5-5')
    target=output/'opus'; settings=json.loads((target/'settings.json').read_text())
    completed_before=(target/'trajectories/opus_0001.json').read_bytes()
    recovery=tmp_path/'recovery.json'; recovery.write_text(json.dumps({'episode_id':'episode','reason':'manual Claude quota recovery','saved_settings':settings,'scenario_ids':['negative'],'prior_attempts':{'negative':3},'max_new_attempts':3}))
    runner.run_target(suite,output,'opus','claude','claude-opus-5-5',quota_recovery_file=recovery)
    assert calls==['positive','negative','negative','negative','negative']
    assert (target/'trajectories/opus_0001.json').read_bytes()==completed_before
    record=json.loads((target/'measurements/opus_0002.attempt-4.json').read_text())
    assert record['attempt']==4 and record['recovery']['prior_attempts']==3
    assert record['settings']==settings
    runner.run_target(suite,output,'opus','claude','claude-opus-5-5',quota_recovery_file=recovery)
    assert len(calls)==5
    monkeypatch.setattr(runner,'versions',lambda _: {'sdk_version':'changed','cli_version':'fixture'})
    with pytest.raises(ValueError,match='Current runtime'):
        runner.run_target(suite,output,'opus','claude','claude-opus-5-5',quota_recovery_file=recovery)


def test_recovery_rejects_non_quota_evidence_and_reordered_cases(tmp_path):
    target=tmp_path/'target'; (target/'measurements').mkdir(parents=True)
    settings={'agent':'claude'}; (target/'settings.json').write_text(json.dumps(settings))
    rows=[type('Row',(),{'id':sid}) for sid in ['a','b']]
    path=tmp_path/'recovery.json'
    manifest={'episode_id':'e','reason':'manual Claude quota recovery','saved_settings':settings,'scenario_ids':['b','a'],'prior_attempts':{'a':0,'b':0},'max_new_attempts':3}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='original suite order'):
        runner.validate_quota_recovery(path,settings,rows,target)
    manifest.update(scenario_ids=['a'],prior_attempts={'a':3});path.write_text(json.dumps(manifest))
    (target/'measurements/r.attempt-3.json').write_text(json.dumps({'scenario_id':'a','run_id':'r','attempt':3,'status':'failed','agent_error':'ordinary tool error'}))
    with pytest.raises(ValueError,match='non-quota'):
        runner.validate_quota_recovery(path,settings,rows,target)


@pytest.mark.anyio
async def test_zai_agent_preserves_tool_calls_through_real_sdk(monkeypatch, tmp_path):
    import httpx
    import openai
    from agents import ModelSettings
    from agents.models.interface import ModelTracing
    from agent.openai_agent import runner as agent_runner

    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setenv("GLM_REASONING_EFFORT", "low")
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_TRACE_FILE", str(tmp_path / "events.jsonl"))
    real_client = openai.AsyncOpenAI

    def handle(request):
        assert str(request.url) == "https://api.z.ai/api/paas/v4/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "glm-5.3"
        assert payload["reasoning_effort"] == "low"
        assert payload["thinking"] == {"type": "enabled"}
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "glm-5.3", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                    "function": {"name": "iot_history", "arguments": '{"asset_id":"Transformer 1"}'}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})

    clients = []
    def client(**kwargs):
        clients.append(real_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))))
        return clients[-1]

    monkeypatch.setattr(agent_runner, "AsyncOpenAI", client)
    config = agent_runner._build_run_config("zai/glm-5.3")
    try:
        result = await config.model_provider.get_model("glm-5.3").get_response(
            None, "Fetch transformer history", config.model_settings, [], None, [], ModelTracing.DISABLED)
        tool, = result.output
        assert tool.type == "function_call"
        assert tool.name == "iot_history"
        assert json.loads(tool.arguments) == {"asset_id": "Transformer 1"}
        events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
        completed,=[e for e in events if e['kind']=='model_request_end']
        assert completed['duration_ms']>0
        assert completed['response']['output'][0]['name']=='iot_history'
        assert completed['usage']['input_tokens']==10
        assert 'test-key' not in (tmp_path/'events.jsonl').read_text()
    finally:
        for instance in clients:
            await instance.close()
