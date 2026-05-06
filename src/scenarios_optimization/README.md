# scenarios_optimization

Performance-optimized scenario generation pipeline for AssetOpsBench. Drop-in replacement for `scenarios_profiling`: significantly faster execution through caching, thread pool offloading, batching and parallelization.

## Optimization Techniques

| Technique | Description |
|-----------|-------------|
| **Two-Level Caching** | Asset profiles are cached in an L1 in-memory LRU cache and an L2 disk-based JSON cache (default TTL: 24 h), so repeat runs for the same asset skip expensive LLM profile-building entirely. |
| **Research Digest Caching** | Precomputed research digests are cached separately so only the final profile-builder LLM step reruns when the digest is fresh but the profile is stale. |
| **Few-Shot Example Caching** | Few-shot examples are kept in an in-memory `_LRUCache` to avoid repeated HuggingFace file I/O on every generation call. |
| **Budget Memoization** | Per-focus token budget allocations are memoized per `(asset_name, total)` pair within the run to avoid redundant computations. |
| **Batch Processing** | Scenario generation is issued in configurable batches (default: 10) instead of a single large LLM call, reducing per-call token usage and tightening the validation feedback loop. |
| **Parallel Focus Generation** | All per-focus generation tasks run concurrently via `asyncio.gather` behind a semaphore (default: 3 concurrent), eliminating the sequential bottleneck of the baseline pipeline. |
| **Thread-Pool Offloading** | Blocking synchronous calls (grounding lookups, file I/O) are offloaded to a `ThreadPoolExecutor` via `run_in_executor` to keep the async event loop unblocked. |

## Usage

```bash
uv run python -m scenarios_optimization.generator Transformer --num-scenarios 50 --data-in-couchdb --invalidate-cache --profile --profile-dir profiling_output/exp1_latency_optimized
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `asset_name` | — | Asset class name (e.g. `"Chiller"`, `"Wind Turbine"`) |
| `--model-id` | watsonx llama-4-maverick | LiteLLM model identifier |
| `--num-scenarios` | 50 | Total scenarios to generate |
| `--batch-size` | 10 | Scenarios requested per LLM call |
| `--max-concurrent` | 3 | Max parallel focus generation tasks |
| `--no-disk-cache` | off | Disable L2 disk cache (L1 stays active) |
| `--cache-dir` | `.cache/scenarios_optimization` | Root directory for disk cache |
| `--cache-ttl-hours` | 24.0 | Disk cache TTL in hours (0 = no expiry) |
| `--timing-report` | off | Print wall/CPU timing breakdown |
| `--profile` | off | Run under PyTorch profiler |
| `--profile-dir` | `profiling_output/<asset>_optimized` | Profiler output directory |
| `--show-workflow` | off | Print step-by-step progress |
| `--log` | off | Write prompt/artifact logs to disk |
| `--retriever` | `arxiv` | Academic retrieval backend (`arxiv` or `semantic_scholar`) |
| `--research-digest` | — | Path to precomputed research digest |
| `--data-in-couchdb` | off | Use grounded generation with live CouchDB data |

## Output

Generated scenarios are written to `generated/scenarios_optimized/<asset_name>/` as a JSON file.
