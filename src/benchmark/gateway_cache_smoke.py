"""Small, explicit cache probe through the production Stirrup API client.

This is a transport/cache diagnostic, not a scenario evaluation. It uses only
the personal key file, never environment keys or Doppler, and calls no MCP tools.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from dotenv import dotenv_values

from agent.stirrup_agent.runner import StirrupAgentRunner
from benchmark.generated.auth import private_json
from benchmark.generated.stirrup_worker import UsageRecorder

KEY_FILE = Path.home() / '.config/assetopsbench/vercel.env'
MODELS = ('anthropic/claude-opus-5.5', 'openai/gpt-6.1-sol',
          'google/gemini-3.8-flash', 'spacexai/grok-4.7', 'zai/glm-5.3')


def personal_key(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f'Create the personal key file: {path}')
    if path.stat().st_mode & 0o777 != 0o600:
        raise ValueError('The personal key file must have permissions 0600')
    value = (dotenv_values(path, interpolate=False).get('AI_GATEWAY_API_KEY') or '').strip()
    if not value:
        raise ValueError(f'Set AI_GATEWAY_API_KEY in {path}; no fallback key will be used')
    return value


def fixture(run_id: str) -> str:
    prefix = (
        f'Cache diagnostic {run_id}. These are synthetic calibration records, not benchmark evidence.\n'
        'Do not analyze the records or call any tool. Reply with only the exact marker requested '
        'in the latest user message. The records are present only to test reuse of a long prompt.\n'
    )
    records = []
    for index in range(360):
        records.append(json.dumps({'sample': index, 'asset': f'probe-chiller-{index % 8}',
            'temperature_c': round(20 + index % 31 * .13, 2), 'load_kw': 100 + index % 97,
            'observed': True, 'note': 'Synthetic cache calibration record; no operational conclusion.'},
            separators=(',', ':')))
    return prefix + '\n'.join(records)


async def probe(model: str, prefix: str, output: Path, semaphore: asyncio.Semaphore):
    from stirrup.core.models import SystemMessage, UserMessage
    async with semaphore:
        folder = output / model.replace('/', '--')
        recorder = UsageRecorder(folder / 'api-usage.json')
        runner = StirrupAgentRunner(model='litellm_proxy/' + model, reasoning_effort='low',
                                    max_output_tokens=512)
        client = recorder.install(runner._build_client())
        private_json(folder / 'request-config.json', {
            'model': model, 'gateway': client._kwargs.get('extra_body', {}).get('providerOptions', {}).get('gateway'),
            'session_affinity_enabled': bool(client._kwargs.get('extra_headers', {}).get('x-session-affinity'))})
        # A smoke test must have a small, visible request count; no SDK retries.
        client._client.max_retries = 0
        messages = [SystemMessage(content=prefix)]
        turns = []
        started = time.monotonic()
        for turn in range(1, 4):
            marker = f'CACHE_OK_{turn}'
            messages.append(UserMessage(content=f'Reply with exactly {marker}'))
            record = {'turn': turn, 'expected': marker}
            try:
                response = await asyncio.wait_for(client.generate(messages, {}), timeout=100)
                actual = ''.join(getattr(block, 'text', '') for block in response.blocks).strip()
                record.update(status='completed', answer=actual, correct=actual == marker)
                messages.append(response)
            except Exception as exc:
                # Error type/status only: exception text may contain credentials.
                record.update(status='error', error=type(exc).__name__,
                              http_status=getattr(exc, 'status_code', None))
            turns.append(record)
            private_json(folder / 'turns.json', turns)
            if record['status'] == 'error':
                break
        await client._client.close()
        usage = recorder.summary()
        repeat_calls = recorder.calls[1:]
        cached_repeats = sum(bool((r.get('usage', {}).get('prompt_tokens_details') or {}).get('cached_tokens')
                                 or r.get('usage', {}).get('cache_read_input_tokens')) for r in repeat_calls)
        result = {'model': model, **usage, 'turns': turns,
                  'cache_verified': cached_repeats > 0,
                  'cached_repeat_requests': cached_repeats,
                  'elapsed_seconds': round(time.monotonic() - started, 3)}
        private_json(folder / 'result.json', result)
        print(json.dumps(result), flush=True)
        return result


async def run(args):
    key = personal_key(args.key_file.expanduser())
    # Overwrite the two variables this exact route consumes. Never read a
    # TokenRouter, OpenAI, shared Gateway, or Doppler key as a fallback.
    os.environ['LITELLM_API_KEY'] = key
    os.environ['LITELLM_BASE_URL'] = 'https://ai-gateway.vercel.sh/v1'
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    run_id = uuid4().hex
    prefix = fixture(run_id)
    (args.output / 'prompt.txt').write_text(prefix)
    private_json(args.output / 'config.json', {
        'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Controlled Stirrup API-client cache diagnostic; not a benchmark execution',
        'key_source': str(args.key_file.expanduser()), 'models': MODELS,
        'request_limit_per_model': 3, 'max_output_tokens': 512, 'reasoning_effort': 'low',
        'prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(), 'prefix_characters': len(prefix),
        'gateway_caching': 'auto', 'automatic_retries': 0})
    semaphore = asyncio.Semaphore(2)
    results = await asyncio.gather(*(probe(model, prefix, args.output, semaphore) for model in MODELS))
    known_costs = [r['cost_usd'] for r in results if r['cost_usd'] is not None]
    private_json(args.output / 'summary.json', {'models': results,
        'all_cache_verified': all(r['cache_verified'] for r in results),
        'reported_total_cost_usd': sum(known_costs) if len(known_costs) == len(results) else None})
    lines = ['# Vercel cache smoke test', '',
             'Three changing-tail requests per model through the production Stirrup API client. '
             'Synthetic calibration records; no MCP execution, authoring, or judging.', '',
             '| Model | Requests | Cached follow-ups | Cached input tokens | Cost | Result |',
             '|---|---:|---:|---:|---:|---|']
    for r in results:
        cost = f"${r['cost_usd']:.5f}" if r['cost_usd'] is not None else 'not reported'
        lines.append(f"| {r['model']} | {r['api_calls']} | {r['cached_repeat_requests']}/2 | "
                     f"{r['cache_read_tokens']:,} | {cost} | "
                     f"{'cache verified' if r['cache_verified'] else 'not verified'} |")
    lines.extend(['', 'Each model directory contains the actual API usage and response markers. '
                  'A cache pass means at least one follow-up reported cached input tokens. '
                  'This does not establish the cache-hit rate of a full scenario evaluation.'])
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path, default=KEY_FILE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
