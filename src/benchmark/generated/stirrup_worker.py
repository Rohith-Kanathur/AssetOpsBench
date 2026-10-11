"""Execute one scenario with Stirrup, shared task files, and API usage evidence."""

import argparse
import asyncio
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from agent.stirrup_agent.runner import StirrupAgentRunner
from agent.stirrup_agent.capture import ExecutionCapture
from .auth import private_json


class UsageRecorder:
    """Record real calls, including compaction; never send cache-probe requests."""

    def __init__(self, path: Path, capture=None):
        self.path = path
        self.calls = []
        self.capture = capture

    def wrap(self, create):
        async def measured(**kwargs):
            # Only hash model inputs; credentials and request headers are excluded.
            inputs = {k: kwargs[k] for k in ("model", "messages", "tools", "temperature",
                      "reasoning_effort", "max_tokens", "max_completion_tokens") if k in kwargs}
            row = {"request_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True,
                   default=str).encode()).hexdigest(), "requested_model": kwargs.get("model")}
            started = time.monotonic()
            try:
                if self.capture is not None:
                    self.capture.request(kwargs)
                response = await create(**kwargs)
                if self.capture is not None:
                    self.capture.response(response)
                usage = getattr(response, "usage", None)
                row.update(status="completed", response_model=getattr(response, "model", None),
                           response_id=getattr(response, "id", None),
                           usage=usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {}))
                cost = getattr(response, "_hidden_params", {}).get("response_cost")
                if cost is not None and row["usage"].get("cost") is None:
                    row["usage"]["cost"] = cost
                return response
            except Exception as exc:
                row.update(status="error", error=type(exc).__name__)
                raise
            finally:
                row["latency_seconds"] = round(time.monotonic() - started, 3)
                self.calls.append(row)
                private_json(self.path, self.calls)
        return measured

    def install(self, client):
        from stirrup.clients.chat_completions_client import ChatCompletionsClient
        if isinstance(client, ChatCompletionsClient):
            api = client._client.chat.completions
            api.create = self.wrap(api.create)
        else:
            # This worker has its own process, so native LiteLLM instrumentation
            # cannot affect another case's client or credentials.
            import stirrup.clients.litellm_client as native
            native.acompletion = self.wrap(native.acompletion)
        return client

    def summary(self):
        usages = [row["usage"] for row in self.calls if row.get("status") == "completed"]
        costs = [usage.get("cost") for usage in usages]
        return {
            "api_calls": len(self.calls),
            "api_prompt_tokens": sum(u.get("prompt_tokens", 0) or 0 for u in usages),
            "api_output_tokens": sum(u.get("completion_tokens", 0) or 0 for u in usages),
            "cache_read_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                     or u.get("cache_read_input_tokens", 0) or 0 for u in usages),
            "cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
            "cost_source": "provider-reported" if costs and all(c is not None for c in costs) else "unavailable",
        }


def record_attempt(record, result, max_turns):
    """A bounded agent attempt can finish without successfully solving its task."""
    trajectory = asdict(result.trajectory)
    record.update(answer=result.answer, trajectory=trajectory)
    turns = trajectory.get("turns", [])
    finished = any(call.get("name") == "finish" for turn in turns
                   for call in turn.get("tool_calls", []))
    if len(turns) >= max_turns and (not finished or not result.answer.strip()):
        record.update(status="completed", termination_reason="max_turns")
        if not result.answer.strip():
            record["task_completed"] = False
    elif result.answer.strip():
        record.update(status="completed")
    else:
        raise ValueError("Agent returned no final answer")


async def run(args):
    capture = ExecutionCapture(args.output.parent)
    recorder = UsageRecorder(args.output.parent / "api-usage.json", capture=capture)

    class MeasuredRunner(StirrupAgentRunner):
        def _build_client(self):
            return recorder.install(super()._build_client())

    started = time.monotonic()
    record = {"runner": "stirrup", "model": args.model, "status": "error", "answer": "",
              "stirrup_version": version("stirrup"),
              "trajectory": {}, "execution_capabilities": ["mcp", "general-execution"],
              "settings": {"max_turns": args.max_turns, "max_output_tokens": args.max_output_tokens,
                           "reasoning_effort": args.reasoning_effort, "temperature": args.temperature,
                           "timeout": args.timeout}}
    try:
        servers = json.loads(args.mcp_config.read_text())["mcpServers"]
        runner = MeasuredRunner(model=args.model, server_paths=servers, code_enabled=True,
                                code_backend="docker", workspace_dir=args.workspace,
                                shared_workspace=True, max_turns=args.max_turns,
                                max_output_tokens=args.max_output_tokens,
                                reasoning_effort=args.reasoning_effort, temperature=args.temperature,
                                container_record=args.output.parent / "code-container.json", capture=capture)
        private_json(args.output.parent / "system-prompt.json", {"prompt": runner._build_system_prompt()})
        result = await asyncio.wait_for(runner.run(args.question_file.read_text()), args.timeout)
        record_attempt(record, result, args.max_turns)
    except Exception as exc:
        # Provider exception text can contain headers or credentials.
        record.update(error=type(exc).__name__, timed_out=isinstance(exc, TimeoutError))
        record['trajectory'] = capture.partial_trajectory()
        record['capture'] = {'protocol': 'durable-stirrup-v1',
                             'messages': 'message-events.jsonl',
                             'requests': 'api-requests.jsonl', 'responses': 'api-responses.jsonl',
                             'complete_record': True, 'interrupted': True}
        termination = capture.model_failure()
        if termination is not None:
            record.update(status='completed', termination_reason=termination, task_completed=False)
    else:
        record['capture'] = {'protocol': 'durable-stirrup-v1',
                             'messages': 'message-events.jsonl',
                             'requests': 'api-requests.jsonl', 'responses': 'api-responses.jsonl',
                             'complete_record': True, 'interrupted': False}
    record.update(recorder.summary(), elapsed_seconds=round(time.monotonic() - started, 3))
    private_json(args.output, record)
    return int(record["status"] != "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mcp-config", type=Path, required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--temperature", type=float)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
