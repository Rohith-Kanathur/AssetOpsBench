"""scenarios_optimization – performance-optimized scenario generation for AssetOpsBench.

This package is a drop-in performance replacement for ``scenarios_profiling``.
All pipeline logic (constraints, models, prompts, retrieval) is reused unchanged
from ``scenarios_profiling``; only the orchestration layer is replaced with an
optimized implementation.

Optimization Overview
---------------------

A. Caching
   * L1 in-memory thread-safe LRU cache for built ``AssetProfile`` objects.
   * L2 disk-based JSON cache (TTL configurable, default 24 h) that survives
     process restarts.
   * Research digests cached separately so that only the final LLM profile-
     builder step is re-run when the digest is fresh but the profile is stale.
   * Few-shot examples cached in-memory via ``_LRUCache`` to avoid repeated
     HuggingFace file I/O.
   * Budget allocations memoized per (asset_name, total) within the run.

B. Batch Processing
   * Scenario generation is driven in configurable batches
     (``--batch-size``, default 10) rather than requesting the entire count
     in a single LLM call.
   * Reduces per-call token usage and allows tighter feedback loops between
     generation and validation.

D. Parallelization
   * Per-focus generation tasks run concurrently via ``asyncio.gather``.
   * A semaphore (``--max-concurrent``, default 3) prevents rate-limit errors.
   * Blocking synchronous calls (grounding, file I/O) are offloaded to a
     ``ThreadPoolExecutor`` via ``run_in_executor``.

CLI::

    python -m scenarios_optimization.generator Chiller --num-scenarios 50 \\
        --batch-size 10 --max-concurrent 3 --timing-report
"""

from .generator import OptimizedScenarioGeneratorAgent
from .optimization_utils import (
    BatchConfig,
    GLOBAL_TIMING,
    TimingRegistry,
    TwoLevelAssetProfileCache,
    DiskCache,
    ProgressiveTimeout,
    estimate_token_count,
    chunk_list,
    timed_section,
)

__all__ = [
    "OptimizedScenarioGeneratorAgent",
    "BatchConfig",
    "GLOBAL_TIMING",
    "TimingRegistry",
    "TwoLevelAssetProfileCache",
    "DiskCache",
    "ProgressiveTimeout",
    "estimate_token_count",
    "chunk_list",
    "timed_section",
]
