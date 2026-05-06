# scenarios_profiling

Scenario generation pipeline with PyTorch profiler instrumentation. A drop-in companion to the base `scenarios` package; all pipeline logic is identical; `torch.profiler.record_function` spans have been added to every major stage so execution can be profiled with fine granularity.

Profiled pipeline phases:

| Span | Description |
|------|-------------|
| `phase_0_server_descriptions` | Fetch MCP server tool descriptions |
| `phase_1_build_asset_profile` | LLM-based asset profile construction (retrieval + digest + profile) |
| `phase_2_allocate_budget` | Per-focus scenario budget allocation |
| `phase_3_generate_all_focuses` | All focus generation tasks (outer span) |
| `phase_3_generate_<focus>` | Per-focus generation sub-span |
| `phase_4_generate_validate_multiagent` | Multi-agent scenario generation and validation |

### CLI

```bash
# Normal generation (no profiler overhead)
uv run python -m scenarios_profiling.generator Transformer --num-scenarios 50 --data-in-couchdb --profile --profile-dir profiling_output/exp1_latency_baseline

# Profile the run and export chrome trace
python -m scenarios_profiling.generator Chiller \
    --profile \
    --profile-dir profiling_output/chiller

# Profile with memory tracking
python -m scenarios_profiling.generator Chiller --profile --profile-memory

# Profile with stack traces
python -m scenarios_profiling.generator Transformer --profile --profile-with-stack

# Use semantic scholar instead of arXiv for retrieval
python -m scenarios_profiling.generator Transformer --retriever semantic_scholar

# Skip retrieval with a precomputed research digest
python -m scenarios_profiling.generator Transformer \
    --research-digest path/to/digest.md

# Show step-by-step console output
python -m scenarios_profiling.generator Transformer --show-workflow

# Write prompt/artifact logs to disk
python -m scenarios_profiling.generator Transformer --log
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `asset_name` | — | Asset class name (e.g. `"Chiller"`, `"Wind Turbine"`) |
| `--model-id` | watsonx llama-4-maverick | LiteLLM model identifier |
| `--num-scenarios` | 50 | Total scenarios to generate |
| `--profile` | off | Run under the PyTorch profiler |
| `--profile-dir` | `profiling_output/<asset>` | Directory for profiler output |
| `--profile-memory` | off | Also track memory allocations |
| `--profile-with-stack` | off | Capture stack traces in the profile |
| `--show-workflow` | off | Print step-by-step pipeline progress |
| `--log` | off | Write prompt/artifact logs to disk |
| `--retriever` | `arxiv` | Academic retrieval backend (`arxiv` or `semantic_scholar`) |
| `--research-digest` | — | Path to precomputed research digest (skips retrieval steps) |
| `--data-in-couchdb` | off | Use grounded generation with live CouchDB data |

## Profiler Output

When `--profile` is used, the profiler writes the following to `--profile-dir`:

- `chrome_trace.json` — load in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev) for a visual flame chart
- Key-averages table printed to stdout showing CPU time, self-CPU time, and call counts per span

## Output

Generated scenarios are written to `generated/scenarios/<asset_name>/` as a JSON file.
