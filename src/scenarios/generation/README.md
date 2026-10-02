# Scenario generation

Generate scenarios for an asset class using native Codex, existing MCP tools and
source-backed data. Inspect and reuse the environment first; research, seed data
or add tools only for missing capabilities. Evaluation runners remain separate.

## Run

Configure Docker, Codex and Kaggle once:

```bash
codex login
uv tool install kaggle
kaggle auth login
```

From the repository checkout:

```bash
PYTHONPATH=src python -m scenarios.generation --asset-class Transformer
```

After installing the project, the equivalent command is
`scenario-generate --asset-class Transformer`. Defaults: five scenarios across
IoT, FMSR, TSFM, WO and Vibration; Codex / GPT-6 Astra / `xhigh` / `fast`.

```bash
scenario-generate run /path/to/run --asset-class Chiller --count 2 --domains IoT WO
scenario-generate run /path/to/run --followup "Check the unresolved source claims."
scenario-generate check /path/to/run
scenario-generate stop /path/to/run
```

Override generation settings with `--harness`, `--model`, `--reasoning-effort`
and `--service-tier`. Codex is implemented today; `harnesses.py` contains the
command-builder boundary for future clients. Requested settings and prompt hashes
are saved with each invocation. Unsupported model settings surface as CLI errors.

## Environment

A new run exports the selected checkout's committed server, loader and tool code,
including fixtures, tests and model artifacts. Use `--repository` and `--ref` to
choose another committed environment; defaults are the current checkout and HEAD.
There are no asset-specific exclusions. Uncommitted edits are not exported.

The command builds its Docker image if needed, starts an isolated CouchDB volume,
and loads the repository's normal default manifest. Follow-up sessions reuse the
saved source and database. It does not connect to or modify the host database.
Preparing a different starting environment is an experiment setup choice, outside
the generation prompt. The agent reads only its asset/scope request and normal
instructions in [profile.md](prompts/profile.md) and [generate.md](prompts/generate.md).

The container mounts the generated workspace and read-only Codex/Kaggle credentials.
It has native shell, file editing and web search; the Docker socket, host checkout
and Git history are not mounted. Credentials remain accessible to the native
clients in the container. No credentials belong in prompts or Git.

## Results

The command prints its output directory, under `~/.cache/assetopsbench/generation`
by default. Open `workspace/output/README.md` for scenarios, sources, capabilities
and gaps. Raw data and agent traces stay in this local directory.

`check` verifies source hashes, requested coverage, evidence files, tool discovery
and live asset/sensor/window/work-order references. Inspect generated algorithms
and source interpretation separately before using them. `stop` retains the
volume for review; a follow-up starts it again without reseeding.

Tests: `python -m pytest src/scenarios/generation/tests -q`.
