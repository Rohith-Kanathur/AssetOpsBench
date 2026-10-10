# Scenario evaluation

The paper uses one Stirrup harness with domain MCP tools and Docker code
execution for every evaluated API model. Each case starts with a fresh database
and a shared task directory. Server source, generator instructions, reference
answers and validation evidence remain outside the agent's workspace.

## Prepare

```bash
uv sync
uv run scenario-generate build
docker build -t assetops-scenario-evaluation:local -f src/benchmark/generated/Dockerfile .
docker build -t assetops-code -f src/agent/stirrup_agent/Dockerfile.code .
```

Generation uses Codex CLI's native login in an isolated local environment:

```bash
uv run scenario-generate run /path/to/generated-chiller --asset Chiller
```

The agent researches academic datasets and domain literature, prepares missing
capabilities, exercises tools on live records, and repairs problems. Automated
checks validate structure, source references and supported operations. A passing
checker alone does not prove every complete scenario succeeds or that a diagnosis
is scientifically valid. Review the exercised evidence before freezing the cohort.

## Execute

Use Doppler's existing development configuration for available API credits:

```bash
doppler run --project cofounder --config dev -- \
  uv run scenario-evaluate /path/to/generated-chiller results/chiller-synthetic \
  --max-turns 30 --max-output-tokens 8192 --timeout 600
```

For a validated human cohort, supply a frozen directory containing `environment/`,
`database/`, `inputs/`, `scenarios.json`, and a `manifest.json` of file hashes:

```bash
doppler run --project cofounder --config dev -- \
  uv run scenario-evaluate /path/to/validated-chiller/snapshot results/chiller-human \
  --snapshot --max-turns 30 --max-output-tokens 8192 --timeout 600
```

Every file under an imported snapshot's `inputs/` is explicitly public to the
agent. Keep expected answers and validator scripts elsewhere. The original
scenario records remain unchanged. Use the same permitted data and tool access
for the human and synthetic comparison.

The default model is Luna. At startup, the command prefers configured TokenRouter
credentials, then the OpenAI-compatible gateway, then a direct OpenAI key.
`AI_GATEWAY_API_KEY` maps to Vercel's endpoint automatically. Override the model
matrix explicitly for the actual study:

```bash
uv run scenario-evaluate /path/to/generated-chiller results/model-comparison \
  --runners '{"stirrup":["tokenrouter/openai/gpt-5.6-luna","litellm_proxy/anthropic/claude-opus-5.5"]}'
```

`TOKENROUTER_API_KEY` / `TOKENROUTER_BASE_URL` configure TokenRouter;
`LITELLM_API_KEY` / `LITELLM_BASE_URL` configure an OpenAI-compatible gateway.
Direct provider names use LiteLLM with that provider's API key. Credentials are
kept in the worker process and are not mounted into the code container.
Selection is explicit and fixed for each run. There is no silent provider or model
switch when credits run out; use a new results directory for a different route.
This keeps provider changes visible in the comparison.

Use `--ids 9` for a bounded case. Completed executions resume; `--retry-failed`
archives failed attempts before retrying. The cohort records the models and
Stirrup settings and rejects changes on resume. No temperature override is applied
unless `--temperature` is supplied; reasoning effort is likewise explicit.

## Judge and report

Fable 5.1 (`claude-fable-5-1`) uses Claude Code subscription authentication in a
separate, read-only session with the full trace and output files. It uses the
existing six-criterion rubric. A strict pass requires all five positive criteria
and no hallucinations. Startup errors, timeouts, missing grades and rubric failures
remain distinct.

```bash
uv run scenario-judge results/chiller-human --jobs 2
```

`README.md` and `cases.csv` contain results, six criteria and execution times.
Stirrup cases also retain `native/api-usage.json`, cache-read tokens and
provider-reported cost. Missing provider billing fields are reported as unknown,
not zero. Cache probes are not inserted into evaluation runs.

Native Codex, Claude Code and ZCode remain available through explicit runner
overrides for startup smoke checks. They are not the default paper harness.
The evaluation command accepts only general-execution runs.

Run artifacts stay in ignored `results/`, `output/` or `generated/`, or outside
this checkout. Never commit credentials, downloaded private data or trajectories.
