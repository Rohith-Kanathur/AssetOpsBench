# Coding agent execution

These adapters run Codex, Claude Code or first-party ZCode once per scenario.
They keep the native agent loop and normalize its answer, tool trajectory and usage.
`completed` means the process completed, not that the scenario passed evaluation.

Run inside the benchmark's isolated container. Give the agent only its question,
declared input files and MCP endpoints. Keep generation artifacts, expected answers,
database administration and server source in the separate tools environment.
`ASSETOPS_EXECUTION_ISOLATED=1` acknowledges that the caller provides this isolation;
it is not a sandbox by itself.

```bash
PYTHONPATH=src/agent python -m coding_agent \
  --harness codex --model gpt-6-astra \
  --workspace /workspace --mcp-config /config/mcp.json \
  --question-file /config/question.txt --output-dir /results
```

MCP configuration uses `{"mcpServers": {"iot": {"type": "http", "url": "http://tools:8100/mcp"}}}`.
Both HTTP and stdio servers work. For Claude use `--harness claude --model <model>`.
For ZCode use `--harness zcode`; its defaults are `GLM-5.3` and `high` reasoning.
The ZCode image builds the [official CLI](https://github.com/zai-org/ZCode).
`ZAI_API_KEY` authenticates its Coding Plan provider through a temporary private
configuration file. It does not copy OpenAI or Claude authentication. Native MCP,
Bash and file tools remain available; web tools, plugins, skills and memory are disabled.
Native full tool-output files are retained alongside JSONL and included in saved judge evidence.
The module can also be imported as `agent.coding_agent` in a full installation.

`--auth-home` supplies a directory containing Codex `auth.json` or Claude
`.credentials.json`; otherwise each CLI's configured auth directory is used.
Claude also accepts `CLAUDE_CODE_OAUTH_TOKEN`. Only credentials are copied into a
temporary home; user settings, hooks, skills and conversation history are excluded.
Claude's `--bare` mode is deliberately avoided because it disables subscription auth.

Each fresh output directory receives `result.json`, `stdout.jsonl` and `stderr.log`.
Known credential values are redacted before saving. The timeout kills the process
group, retaining partial evidence and reporting an execution error.
