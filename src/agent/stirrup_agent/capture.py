"""Durable, per-execution evidence recorded before display or response parsing."""

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
from unittest.mock import patch

from .trajectory import build_trajectory
from ..models import ToolCall, TurnRecord


class ExecutionCapture:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.messages = []
        self.last_response = None
        self.response_waiting = False
        self.in_summary = False
        self.response_in_summary = False

    def append(self, name, value):
        path = self.directory / name
        # Append before allowing the parser/logger to consume this evidence.
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
                       'a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def message(self, message):
        from stirrup.core.cache import serialize_message
        value = serialize_message(message)
        self.append('message-events.jsonl', value)
        self.messages.append(value)
        if value.get('role') == 'assistant':
            self.response_waiting = False

    def request(self, kwargs):
        # Never persist headers, credentials, client internals or arbitrary kwargs.
        allowed = ('model', 'messages', 'tools', 'temperature', 'reasoning_effort',
                   'max_tokens', 'max_completion_tokens', 'tool_choice')
        self.append('api-requests.jsonl', {k: kwargs[k] for k in allowed if k in kwargs})

    def response(self, response):
        value = response.model_dump(mode='json', include={'id', 'model', 'choices', 'usage', 'created'})
        self.append('api-responses.jsonl', value)
        self.last_response = value
        self.response_waiting = True
        self.response_in_summary = self.in_summary

    @contextmanager
    def cache_scope(self):
        import stirrup.core.cache as cache
        # Stirrup's default cache is keyed only by question. Keep it inside this
        # case so a different model answering the same question cannot clear it.
        with patch.object(cache, 'DEFAULT_CACHE_DIR', self.directory / 'stirrup-cache'):
            yield

    def model_failure(self):
        choices = (self.last_response or {}).get('choices', [])
        if not choices or self.response_in_summary:
            return None
        choice = choices[0]
        if choice.get('finish_reason') in ('length', 'max_tokens'):
            return 'output_token_limit'
        for call in (choice.get('message') or {}).get('tool_calls') or []:
            arguments = (call.get('function') or {}).get('arguments') or '{}'
            try:
                json.loads(arguments)
            except (ValueError, TypeError):
                return 'invalid_tool_call'
        return None

    def partial_trajectory(self):
        from stirrup.core.cache import deserialize_messages
        trajectory = build_trajectory(deserialize_messages(self.messages))
        if self.response_waiting and not self.response_in_summary:
            choices = (self.last_response or {}).get('choices', [])
            if choices:
                message = choices[0].get('message') or {}
                calls = []
                for item in message.get('tool_calls') or []:
                    function = item.get('function') or {}
                    raw = function.get('arguments') or ''
                    try:
                        arguments = json.loads(raw or '{}')
                    except ValueError:
                        arguments = {'_raw': raw}
                    calls.append(ToolCall(name=function.get('name') or 'unknown',
                                          id=item.get('id') or '', input=arguments))
                usage = (self.last_response or {}).get('usage') or {}
                trajectory.turns.append(TurnRecord(index=len(trajectory.turns),
                    text=message.get('content') or '', tool_calls=calls,
                    input_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('completion_tokens', 0)))
        value = asdict(trajectory)
        executed = {m.get('tool_call_id') for m in self.messages if m.get('role') == 'tool'}
        for turn in value['turns']:
            for call in turn['tool_calls']:
                if call['id'] not in executed:
                    # An unexecuted malformed call must not acquire a fake result.
                    call.pop('output', None)
                    call['execution_status'] = 'not_executed'
        return value
