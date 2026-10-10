# Stirrup agent

Stirrup is the shared API-model execution harness for the current evaluation.
It provides the domain MCP tools and a Docker workspace for shell commands,
Python and file reads/writes. Generation separately uses Codex CLI with its native
login; grading separately uses Fable 5.1 through Claude Code.

Use [scenario-evaluate](../src/benchmark/generated/README.md) for frozen cohorts,
fresh databases, per-case isolation, persisted outputs and independent grading.
The standalone command is useful for development:

```bash
uv sync
docker build -t assetops-code -f src/agent/stirrup_agent/Dockerfile.code .
uv run stirrup-agent --model-id tokenrouter/openai/gpt-5.6-luna \
  --show-trajectory "What sensors are available for Chiller 6?"
```

## Credentials and caching

- `tokenrouter/<model>` uses `TOKENROUTER_API_KEY` and `TOKENROUTER_BASE_URL`.
- `litellm_proxy/<model>` uses `LITELLM_API_KEY` and `LITELLM_BASE_URL`.
- Native `<provider>/<model>` names use LiteLLM and that provider's credentials.
- `scenario-evaluate` also accepts `AI_GATEWAY_API_KEY` for Vercel and selects
  configured router credits before a direct OpenAI key.

At Vercel's gateway endpoint, the client requests automatic prompt caching.
Cache reuse depends on the provider and matching prefixes. Cohort runs record
actual provider-reported cache reads and usage; a cached prompt still produces
a new response. Missing billing information is not treated as free usage.

## Tools and workspace

The same six domain MCP servers are available: IoT, FMSR, TSFM, work orders,
vibration and utilities. FMSR reads the recovered failure-mode catalog; an LLM
fallback is not needed to recover the existing modes.

Large MCP responses are saved in the code workspace and replaced in the prompt
with a file reference, so the model can analyze them without copying the payload.
The benchmark mounts the same per-case directory into the code and MCP containers.
Server implementation and reference answers are in separate locations.

`--code-enabled` is the standalone default. `--no-code` is retained for old wiring
checks. `--code-backend local` runs with host permissions and is for trusted local
development; the evaluation uses Docker. `STIRRUP_CODE_IMAGE` overrides the default
`assetops-code` image. `DOCKER_HOST` can select a non-default Docker socket.

## Run limits

The standalone CLI exposes `--max-turns`, `--temperature`, `--reasoning-effort`,
`--workspace-dir`, and `--preserve-workspace`. No temperature or reasoning override
is applied unless supplied. The working context budget is 100,000 tokens and
compaction starts at 75%; complete summaries are retained in the run log.
Cohort runs additionally expose a shared per-call output-token limit and retain
model settings in the cohort manifest.

```bash
uv run pytest src/agent/stirrup_agent/tests -q
```
