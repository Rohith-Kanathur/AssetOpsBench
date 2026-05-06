# scenarios_wandb

Optimized scenario generation pipeline with integrated Weights & Biases experiment tracking. Extends `scenarios_optimization` with real-time W&B logging of timing metrics, cache statistics, and generation summaries.

## W&B Logged Metrics

| Metric | Description |
|--------|-------------|
| `timing/*` | Wall-clock time per pipeline phase |
| `cache/hit_rate` | L1 + L2 cache hit rates for asset profiles |
| `cache/l1_hits`, `cache/l2_hits` | Per-level cache hit counts |
| `generation/total_scenarios` | Number of scenarios successfully generated |
| `generation/mode` | Generation mode (`grounded` or `standard`) |
| Run config | `asset_name`, `num_scenarios`, `model_id`, `batch_size`, `max_concurrent`, `retriever`, cache settings |

## Usage

### Basic run with W&B logging

```bash
python -m scenarios_wandb.generator Chiller \
    --num-scenarios 50 \
    --wandb \
    --wandb-project assetopsbench \
    --wandb-run-name chiller-optimized-50
```

### Profile and log to W&B

```bash
python -m scenarios_wandb.generator Chiller \
    --num-scenarios 50 \
    --profile \
    --profile-dir profiling_output/chiller_wandb \
    --wandb \
    --wandb-project assetopsbench
```

### Invalidate cache before a fresh run

```bash
python -m scenarios_wandb.generator Chiller --invalidate-cache --wandb
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `asset_name` | — | Asset class name (e.g. `"Chiller"`, `"Wind Turbine"`) |
| `--model-id` | watsonx llama-4-maverick | LiteLLM model identifier |
| `--num-scenarios` | 50 | Total scenarios to generate |
| `--batch-size` | 10 | Scenarios requested per LLM call |
| `--max-concurrent` | 3 | Max parallel focus generation tasks |
| `--no-disk-cache` | off | Disable L2 disk cache |
| `--cache-dir` | `.cache/scenarios_optimization` | Root directory for disk cache |
| `--cache-ttl-hours` | 24.0 | Disk cache TTL in hours |
| `--invalidate-cache` | off | Purge cached profile for the given asset before running |
| `--timing-report` | off | Print wall/CPU timing breakdown |
| `--profile` | off | Run under PyTorch profiler |
| `--profile-dir` | `profiling_output/<asset>_optimized` | Profiler output directory |
| `--show-workflow` | off | Print step-by-step pipeline progress |
| `--log` | off | Write prompt/artifact logs to disk |
| `--retriever` | `arxiv` | Academic retrieval backend (`arxiv` or `semantic_scholar`) |
| `--research-digest` | — | Path to precomputed research digest |
| `--data-in-couchdb` | off | Use grounded generation with live CouchDB data |
| `--wandb` | off | Enable W&B logging |
| `--wandb-project` | `assetopsbench` | W&B project name |
| `--wandb-entity` | — | W&B entity/team |
| `--wandb-run-name` | — | W&B run display name |
| `--wandb-tags` | — | Space-separated W&B tags |

## Output

Generated scenarios are written to `generated/scenarios_optimized/<asset_name>/` as a JSON file, identical in format to `scenarios_optimization` output.
