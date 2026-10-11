# Scenario evaluation

The paper uses one Stirrup harness with domain MCP tools and Docker code
execution for every evaluated API model. Each case starts with a fresh database
and a private task directory shared only with that case's tools. Server source, generator instructions, reference
answers and validation evidence remain outside the agent's workspace.

Each case has its own Docker network, MCP process, code container, workspace,
cache and trajectories. One CouchDB service serves the concurrent cases, with
separate databases and credentials for each case. An authenticated gateway maps
logical database names to private namespaces, including newly created result
collections; it scopes cookie authentication and database listings too. Server
admin credentials never enter case containers. Writes and work-order counters
remain private, and case databases are deleted when that execution finishes.

The six MCP endpoints share Python imports only within their own case. Frozen
dependencies are installed once into a content-addressed image keyed by the base
image and requirements, instead of installed during every case startup. Source
and data remain the frozen snapshot. `runtime.json` records each case's runtime.
For diagnostics, `ASSETOPS_DATABASE_MODE=dedicated` uses a separate CouchDB per
case; `ASSETOPS_CACHE_RUNTIME_IMAGE=0` restores per-case dependency installation.

Execution and judging have independent rolling queues. Defaults are ceilings of
12 executions and 14 judge sessions; measured host/Docker available memory,
macOS pressure and paging may admit fewer. New workers reserve startup memory,
with 1.5 GiB headroom. Active workers are never killed to change concurrency.
Each individual judgment is a queued job. Finishing one immediately frees its
slot for any eligible case; the scheduler never waits for a case's full set of
five before grading another case. Only that case's final average waits for all
five distinct-account judgments.

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

Keep the personal Vercel AI Gateway key in
`~/.config/assetopsbench/vercel.env`, outside Git, with permissions `0600`:

```dotenv
AI_GATEWAY_API_KEY=your-key
```

Select that file explicitly for execution:

```bash
uv run scenario-evaluate /path/to/generated-chiller results/chiller-synthetic \
  --env-file ~/.config/assetopsbench/vercel.env \
  --max-turns 30 --max-output-tokens 8192 --timeout 600
```

For a validated human cohort, supply a frozen directory containing `environment/`,
`database/`, `inputs/`, `scenarios.json`, and a `manifest.json` of file hashes:

```bash
uv run scenario-evaluate /path/to/validated-chiller/snapshot results/chiller-human \
  --env-file ~/.config/assetopsbench/vercel.env \
  --snapshot --max-turns 30 --max-output-tokens 8192 --timeout 600
```

Every file under an imported snapshot's `inputs/` is explicitly public to the
agent. Keep expected answers and validator scripts elsewhere. The original
scenario records remain unchanged. Use the same permitted data and tool access
for the human and synthetic comparison.

The default model is Luna. At startup, the command prefers an explicitly configured
Vercel key, then TokenRouter, another OpenAI-compatible gateway, or a direct OpenAI key.
An `AI_GATEWAY_API_KEY` in the selected file takes precedence over inherited router
keys and maps to Vercel's endpoint automatically. Override the model
matrix explicitly for the actual study:

```bash
uv run scenario-evaluate /path/to/generated-chiller results/model-comparison \
  --env-file ~/.config/assetopsbench/vercel.env \
  --runners '{"stirrup":["litellm_proxy/openai/gpt-6.1-sol","litellm_proxy/anthropic/claude-opus-5.5"]}'
```

Gateway requests enable automatic caching and carry one opaque session-affinity
value per runner. Gemini 3.8 Flash is routed through `google`: the cache smoke
observed cache reads there, while the default Vertex route reported no hits.
This pin fails rather than silently switching Gemini to another provider.

Run an explicit, small cache diagnostic separately from benchmark executions:

```bash
uv run python -m benchmark.gateway_cache_smoke --output output/cache-smoke
```

It reads only the personal key file and sends three requests to each of the five
study models. The report includes actual cache tokens and billed cost. It does
not run MCP tools, author scenarios, or invoke a judge.

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

GPT-6 Astra (`gpt-6-astra`, `xhigh`, `fast`) judges each saved execution five
times, in fresh Codex sessions on five distinct Everett subscriptions. The
existing six-criterion rubric is unchanged. Scores, criterion values and strict
pass rates are arithmetic means of the five judgments, with individual scores
retained. A strict pass within one judgment requires the first five criteria and
no hallucinations. This follows the five-judgments-per-trajectory procedure in
[AssetOpsBench §5.2](https://arxiv.org/html/2506.03828v4#S5.SS2). Codex reasoning
settings replace the paper's sampling-temperature configuration.

Check availability without inference with `python -m agent.codex_accounts check`.
Credentials stay under `~/.config/assetopsbench/codex-pool`, outside Git. The pool
uses Sagar's personal Everett configuration and excludes Naomi, Mika/Micah, and Quentin Nolan. Accounts
assigned to hosted runs, stale logins and depleted accounts without resets are
unavailable. It never falls back to another Everett API key or an API-model judge.
Only already available resets may be used, at reported zero remaining usage.
Preflight never redeems a reset. No purchase is automated.

`scenario-judge --jobs` bounds judge sessions (default 14), with
`--sessions-per-account 2` by default. `scenario-evaluate --jobs` bounds executions
and `--judge-jobs` bounds its separate judge queue. The dispatcher reserves an
available account before occupying a judge worker and skips blocked cases so
other ready work can proceed. Two isolated sessions per subscription may run
across different executions. Each execution
still requires five distinct accounts. Credential writes are coordinated;
account resets wait for active sessions to finish. Serial authoring remains
exclusive. A failed judgment is abandoned entirely and retried from the original evidence
on a different account, up to three attempts. Completed judgments live under
`judging/repeats/NN/`; failed attempts stay under `judging-failures/` for diagnosis
and are excluded from successful ATIF exports. The rubric is not relaxed for
failed or unavailable judgments.

The judge receives a separate copy with the evaluated model, runner and
human/synthetic origin withheld. It retains every recorded turn, tool call and
result, plus the workspace files; identifying strings in paths and text are
masked. Original scenarios, execution logs and artifacts stay untouched. The
copy is saved under each repetition’s `judging/evidence/`, with a private audit mapping in
`judging/blinding.json` that is not exposed to the judge. A binary artifact that
cannot be safely blinded blocks grading for review. Saved grades are invalidated
when the blinding policy or any evidence file changes.

```bash
uv run --group harbor scenario-judge results/chiller-human --jobs 5 --repeats 5 \
  --output results/chiller-human-codex --export results/chiller-human-clean-atifs
```

`--output` copies only saved execution evidence into a fresh grading directory,
preserving the old scores. `judging-timing.json` records wall time and summed
session time; the latter is a serial-equivalent estimate, not a separately timed
sequential run.

`--export` creates a separate bundle containing only completed, independent ATIF
trajectories, their native Codex events and rubric scores. Diagnostic failures,
previous attempts and credential files are omitted. Use `--jobs 1` in a separate
copy of the same saved cases for a measured serial control; never confuse summed
session durations with a measured sequential run.

`README.md` and `cases.csv` contain results, six criteria and execution times.
Stirrup cases also retain `native/api-usage.json`, cache-read tokens and
provider-reported cost. Missing provider billing fields are reported as unknown,
not zero. Cache probes are not inserted into evaluation runs.

A run that exhausts its turn budget is still a recorded model attempt and is
graded on its actual trajectory, with `termination_reason: max_turns`. If it
also produced no final answer, the record notes `task_completed: false`.
An answer delivered at the budget boundary remains for the judge to assess.
A completed execution record does not imply that the task was solved.
Infrastructure failures remain execution errors.

The run supervisor can use `judge_cases(..., case_source=..., source_done=...)`
to admit newly completed executions continuously. Its shared worker pool keeps
the configured global limit, while each execution reserves five distinct
accounts and each judgment retains its own isolated trajectory. Slow judgments
do not hold up admission of the next case.

Native Codex, Claude Code and ZCode remain available through explicit runner
overrides for startup smoke checks. They are not the default paper harness.
The evaluation command accepts only general-execution runs.

Run artifacts stay in ignored `results/`, `output/` or `generated/`, or outside
this checkout. Never commit credentials, downloaded private data or trajectories.

For separate Harbor tasks/trials and ATIF trajectories for **Codex Astra
creation → the full execution matrix → five Codex Astra judgments**, see
[the Harbor adapter](../harbor/README.md). The existing commands remain available;
`scenario-evaluate --no-judge` also supports saving executions for a separate
judging stage.
