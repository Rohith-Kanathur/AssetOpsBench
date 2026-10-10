# Harbor pipeline records

This adapter runs **Codex Astra → the configured execution model matrix → Claude
Code / Fable 5.1** as separate, real Harbor trials. It uses Harbor 0.24.0 and
validates trajectories with Harbor's ATIF v1.8 model. The existing generation,
execution isolation and six-criterion judging rubric remain in use.

The adapter is trusted host orchestration: it invokes the existing Docker
runners on the host. It does not give an evaluated agent the host Docker socket.
Harbor supplies trial lifecycle, timeouts, task checksums, result records and a
separate verifier container. These tasks require the `PipelineAgent` adapter;
they are not generic terminal tasks runnable with an arbitrary Harbor agent.

## Run locally

Docker must be running. Codex subscription authentication is required for
creation, and Claude Code subscription authentication for judging. Configure the
execution providers through the same environment variables as `scenario-evaluate`.
Do not put keys in task files or command-line arguments.

Claude credentials are refreshed on the host when needed. Isolated sessions
receive a temporary access-token copy without the refresh token, so they cannot
rotate and discard the host's refresh credential. If the host session has expired,
run `claude auth login` before retrying.

A bounded smoke test (one newly generated IoT scenario, one execution):

```bash
doppler run --project cofounder --config dev -- \
  uv run --group harbor scenario-harbor run output/harbor-smoke \
  --asset Chiller \
  --seed /absolute/path/to/validated/snapshot \
  --count 1 \
  --runners '{"stirrup":"litellm_proxy/openai/gpt-5.6-luna"}' \
  --max-cases 1 --generation-timeout 1800 --timeout 600
```

Creation defaults to `gpt-6-astra` / `xhigh`; judging defaults to
`claude-fable-5-1`. The execution matrix is required explicitly, so a smoke-test
model cannot silently become the paper's full model set. Use the same runner →
model or list-of-models JSON as `scenario-evaluate`. For example:

```bash
--runners '{"stirrup":["litellm_proxy/openai/gpt-5.6-luna","litellm_proxy/anthropic/claude-opus-5.5"],"codex":"gpt-6-astra"}'
```

Generation defaults to 25 scenarios. Use `--count 25` for a total or `--plan
'{"iot":10,"multiagent":15}'` for domain counts. There are no answerability
labels or quotas. Use `--environment existing` with a prepared seed to author
against the existing data and tools.

To evaluate a validated human-authored cohort without running generation:

```bash
doppler run --project cofounder --config dev -- \
  uv run --group harbor scenario-harbor evaluate output/human-chiller-luna \
  --snapshot /absolute/path/to/validated/human-snapshot \
  --runners '{"stirrup":"litellm_proxy/openai/gpt-5.6-luna"}'
```

The snapshot manifest is checked before execution. Original questions and
characteristic forms are retained. Human and synthetic cohorts use the same
execution and judging stages, fresh environments, and evidence format.

Every selected scenario/model pair receives its own execution task, fresh
execution environment and independent Fable judging task. Execution and judge
failures are retained and do not prevent the remaining model pairs from being
attempted. An invalid generation stops the pipeline before execution.
`--max-cases` only bounds scenarios, not the model matrix. Omit it for the full
requested generation budget. Use a fresh output directory for each run.

To attach an already completed native generation, place it at
`DIRECTORY/generation` and pass `--reuse-generation`. The imported trajectory is
explicitly marked as an external run, not a Harbor-executed trial.

To retry judging saved executions without repeating generation or execution:

```bash
uv run --group harbor scenario-harbor judge output/harbor-smoke
```

Use `--case CASE_NAME` to select one saved execution. Retries create new Harbor
tasks and trials; previous results and native judge logs are retained. The
workflow index marks earlier judging trials as superseded. Matching completed
grades are reused by the existing judge cache.

## Saved evidence

```text
RUN/
  invocation.json                  # command and chosen settings; no credentials
  controller/                      # adapter/pipeline source and dependency lock
  tasks/<stage>/                   # task.toml, instruction, stage spec, verifier
  trials/<stage>/
    config.json, result.json       # written by Harbor itself
    agent/trajectory.json          # validated ATIF; actual recorded interactions
    agent/stage.log, outcome.json
    verifier/reward.json
  generation/                      # native Codex logs, research, checks, artifacts
  evaluation/
    snapshot.json, cohort.json     # frozen inputs, hashes, explicit model matrix
    environment/, database/, inputs/, scenarios.json
    cases/<runner-model-scenario>/
      native/, workspace/, result.json
      judging/events.jsonl         # native Claude Code / Fable trace
      judging/prompt.txt, result.json
      judging/evidence/            # complete judge copy, model/source identifiers masked
      judging/blinding.json        # private audit mapping, never shown to the judge
      judge.json
  trajectory.json                  # ATIF workflow with references to each stage
  evidence-manifest.json           # hashes of selected evidence, excluding auth/config
```

Generation includes one embedded trajectory per attempt, including repairs and
failed attempts. Converted ATIF preserves recorded messages and tool inputs and
outputs. Native event files are retained because conversion does not preserve
every provider-specific event or infer missing inference boundaries. No private
reasoning or missing telemetry is invented.

The saved controller includes the exact judge rubric. The characteristic form
defines success, including any required explanation of insufficient evidence;
generic refusal does not replace the required evidence checks. The same rubric
applies to human-authored and synthetic scenarios.
The judge sees the same blinded evidence format for both cohorts. Original
execution logs and workspace files remain intact alongside the judge-only copy.

`stage_completed` is an operational completion reward, **not benchmark accuracy**.
A judging trial additionally records `benchmark_pass`, taken from the existing
rubric. Keep those separate when analyzing Harbor results. A correctly recorded
failed scenario may still have a completed judging stage.

Validate schema, links and evidence hashes:

```bash
uv run --group harbor scenario-harbor validate output/harbor-smoke
```

A manifest is not a publication ZIP. The local run also contains runtime
configuration outside its evidence allowlist. Review anonymity, source/data
licenses and trace contents before packaging; do not zip the entire run directory.
The saved generator commands contain local paths, so use the CLI to regenerate
on a different machine rather than treating absolute paths as portable.

Format references: [Harbor tasks](https://harborframework.com/docs/tasks),
[ATIF specification](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md).
