"""Exercise failure capture through the real Stirrup client and agent loop."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openai.types.chat import ChatCompletion
import pytest

from agent.stirrup_agent.capture import ExecutionCapture
from agent.stirrup_agent.runner import StirrupAgentRunner, _build_full_summary_logger
from benchmark.generated import stirrup_worker


def completion(arguments, identifier):
    return ChatCompletion.model_validate({
        'id': identifier, 'model': 'capture-test', 'created': 0,
        'object': 'chat.completion',
        'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                     'message': {'role': 'assistant', 'content': None,
                                 'tool_calls': [{'id': identifier+'-call', 'type': 'function',
                                                 'function': {'name': 'inspect', 'arguments': arguments}}]}}],
        'usage': {'prompt_tokens': 15, 'completion_tokens': 5, 'total_tokens': 20}})


@pytest.mark.anyio
async def test_malformed_model_arguments_keep_prior_tools_and_raw_response(tmp_path, monkeypatch):
    monkeypatch.setenv('LITELLM_API_KEY', 'test-key')
    monkeypatch.setenv('LITELLM_BASE_URL', 'https://capture.example/v1')
    client = StirrupAgentRunner(model='litellm_proxy/capture-test')._build_client()
    malformed = '{"code":"' + 'z' * 12000
    create = AsyncMock(side_effect=[
        completion('{"query":"read"}', 'response-1'), completion(malformed, 'response-2')])
    client._client.chat.completions.create = create
    monkeypatch.setattr(StirrupAgentRunner, '_build_client', lambda self: client)
    monkeypatch.setattr(StirrupAgentRunner, '_build_tools', lambda self: [])
    (tmp_path/'mcp.json').write_text('{"mcpServers":{}}')
    (tmp_path/'question.txt').write_text('Inspect the existing data.')
    native = tmp_path/'native'
    args = SimpleNamespace(output=native/'result.json', model='litellm_proxy/capture-test',
        mcp_config=tmp_path/'mcp.json', question_file=tmp_path/'question.txt', workspace=tmp_path,
        max_turns=20, max_output_tokens=8192, reasoning_effort='high', temperature=None, timeout=30)

    assert await stirrup_worker.run(args) == 0  # A complete recorded model failure, not a success score.
    saved = json.loads(args.output.read_text())
    assert saved['termination_reason'] == 'invalid_tool_call'
    assert saved['task_completed'] is False and saved['answer'] == ''
    assert saved['api_calls'] == 2 and len(saved['trajectory']['turns']) == 2
    first, second = [t['tool_calls'][0] for t in saved['trajectory']['turns']]
    assert 'not a valid tool' in first['output']
    assert second['input']['_raw'] == malformed
    assert 'output' not in second and second['execution_status'] == 'not_executed'
    wire = [json.loads(line) for line in (native/'api-responses.jsonl').read_text().splitlines()]
    assert wire[-1]['choices'][0]['message']['tool_calls'][0]['function']['arguments'] == malformed
    assert create.await_count == 2  # No repair/model retry.
    assert list((native/'stirrup-cache').glob('*/state.json'))


def test_capture_stores_full_tool_result_before_display_truncation(tmp_path):
    from stirrup.core.models import ToolMessage
    capture = ExecutionCapture(tmp_path)
    text = 'data-' * 10000 + 'END_OF_UNTRUNCATED_TOOL_RESULT'
    logger = _build_full_summary_logger(capture)
    logger.tool_result(ToolMessage(content=text, tool_call_id='one', name='inspect', success=True))
    assert json.loads((tmp_path/'message-events.jsonl').read_text())['content'] == text


def test_caches_for_identical_questions_are_separate(tmp_path):
    from stirrup.core.cache import CacheManager
    first, second = ExecutionCapture(tmp_path/'first'), ExecutionCapture(tmp_path/'second')
    with first.cache_scope():
        manager = CacheManager()
        path = manager._get_state_file('same-question')
        path.parent.mkdir(parents=True)
        path.write_text('first')
    with second.cache_scope():
        other = CacheManager()._get_state_file('same-question')
        other.parent.mkdir(parents=True)
        other.write_text('second')
        CacheManager().clear_cache('same-question')
    assert path.read_text() == 'first'
    assert path != other


def test_capture_excludes_transport_credentials(tmp_path):
    capture = ExecutionCapture(tmp_path)
    capture.request({'model':'test', 'messages':[], 'api_key':'do-not-save',
                     'extra_headers': {'Authorization':'do-not-save'}})
    assert 'do-not-save' not in (tmp_path/'api-requests.jsonl').read_text()


def test_raw_response_survives_client_parse_failure(tmp_path):
    capture = ExecutionCapture(tmp_path)
    capture.response(completion('{"unfinished":', 'bad'))
    partial = capture.partial_trajectory()
    assert partial['turns'][0]['tool_calls'][0]['input'] == {'_raw': '{"unfinished":'}
    assert 'output' not in partial['turns'][0]['tool_calls'][0]


def test_output_limit_response_is_recorded_without_executing_it(tmp_path):
    capture = ExecutionCapture(tmp_path)
    response = completion('{"query":"unfinished"}', 'capped')
    response.choices[0].finish_reason = 'length'
    capture.response(response)
    assert capture.model_failure() == 'output_token_limit'
    assert 'output' not in capture.partial_trajectory()['turns'][0]['tool_calls'][0]
    assert (tmp_path/'api-responses.jsonl').stat().st_mode & 0o777 == 0o600
