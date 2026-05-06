"""Performance optimization utilities for scenario generation.

This module centralises every reusable optimization primitive used by the
``scenarios_optimization`` pipeline:

Caching
-------

Batch Processing
----------------

Parallelisation
---------------
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Generator, Generic, Iterable, Iterator, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Batch configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """Parameters that control batched scenario generation.

    Attributes
    ----------
    generation_batch_size:
        How many scenarios to ask the LLM to generate in a single prompt.
        Larger values reduce the total number of LLM round-trips but increase
        prompt length and may reduce per-scenario quality.
    validation_batch_size:
        How many scenarios to pass to the validator in one call.
        Matches ``generation_batch_size`` by default.
    max_concurrent_batches:
        Maximum number of focus-group batches that may run in parallel via
        the async semaphore.  Set to 1 to disable parallelism.
    max_workers:
        Thread/process pool size for CPU-bound parallelism.
    use_process_pool:
        Use ``ProcessPoolExecutor`` instead of ``ThreadPoolExecutor`` for
        parallelised LLM calls.  Requires the called functions to be
        picklable.  Keep ``False`` (default) for LLM I/O workloads.
    """

    generation_batch_size: int = 10
    validation_batch_size: int = 10
    max_concurrent_batches: int = 3
    max_workers: int = 4
    use_process_pool: bool = False


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split *items* into successive chunks of at most *chunk_size* elements.

    Optimisation technique: batching amortises per-call overhead (HTTP
    handshakes, JSON parsing, rate-limiter tokens) across multiple items.

    Parameters
    ----------
    items:
        The list to split.
    chunk_size:
        Maximum number of elements per chunk.  Must be >= 1.

    Returns
    -------
    list[list[T]]
        List of chunks.  The last chunk may be shorter than *chunk_size*.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# ---------------------------------------------------------------------------
# Thread-safe LRU in-memory cache
# ---------------------------------------------------------------------------


class _LRUCache(Generic[T]):
    """A thread-safe LRU cache backed by ``collections.OrderedDict``.

    Optimisation technique: repeated runs on the same asset class (e.g.
    during development or batch-evaluation) reuse the expensive LLM-generated
    ``AssetProfile`` without re-running retrieval and synthesis.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> T | None:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]

    def put(self, key: str, value: T) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            self._store[key] = value
            if len(self._store) > self._maxsize:
                # Evict the least-recently-used entry
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
                "maxsize": self._maxsize,
            }


# Module-level shared cache instance (process-wide singleton, thread-safe)
_ASSET_PROFILE_CACHE: _LRUCache[Any] = _LRUCache(maxsize=32)
_RESEARCH_DIGEST_CACHE: _LRUCache[str] = _LRUCache(maxsize=32)
_BUDGET_CACHE: _LRUCache[Any] = _LRUCache(maxsize=64)
_FEWSHOT_CACHE: _LRUCache[list[dict]] = _LRUCache(maxsize=64)


def asset_profile_cache_key(
    asset_name: str,
    retriever: str,
    requested_open_form: bool,
) -> str:
    """Build a deterministic cache key for an asset profile.

    The key is a SHA-256 digest of the normalised inputs so it is safe to use
    as a file-system path component.
    """
    raw = f"{asset_name.strip().lower()}|{retriever}|{requested_open_form}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def research_digest_cache_key(asset_name: str, retriever: str) -> str:
    """Build a cache key for a research digest (retrieval output)."""
    raw = f"{asset_name.strip().lower()}|{retriever}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Disk-based persistent cache
# ---------------------------------------------------------------------------


class DiskCache:
    """JSON-based persistent cache for expensive pipeline artefacts.

    Optimisation technique: disk persistence means that even across process
    restarts (e.g. during development, CI reruns, or incremental benchmarking)
    previously computed asset profiles and research digests are reused.

    The cache is stored as individual JSON files under *cache_dir* with
    filenames derived from the cache key.  Writes are atomic (write to a
    ``.tmp`` file then rename) to prevent corruption.

    Parameters
    ----------
    cache_dir:
        Root directory for cache files.  Created automatically if absent.
    namespace:
        Sub-directory under *cache_dir* used to isolate different object
        types (e.g. ``"asset_profiles"`` vs ``"research_digests"``).
    ttl_seconds:
        Maximum age of a cache entry in seconds.  Entries older than this
        are treated as stale and re-fetched.  ``None`` means entries never
        expire.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/scenarios_optimization",
        namespace: str = "default",
        ttl_seconds: float | None = None,
    ) -> None:
        self._root = Path(cache_dir) / namespace
        self._root.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", key)
        return self._root / f"{safe_key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        with self._lock:
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _log.warning("DiskCache: failed to read %s: %s", path, exc)
                return None
        # TTL check
        if self._ttl is not None:
            written_at = payload.get("_written_at", 0)
            if (time.time() - written_at) > self._ttl:
                _log.debug("DiskCache: stale entry for key %s", key)
                return None
        return payload.get("value")

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        payload = {"_written_at": time.time(), "value": value}
        with self._lock:
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                tmp.replace(path)
            except OSError as exc:
                _log.warning("DiskCache: failed to write %s: %s", path, exc)
                tmp.unlink(missing_ok=True)

    def invalidate(self, key: str) -> None:
        path = self._path(key)
        with self._lock:
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        with self._lock:
            for child in self._root.glob("*.json"):
                child.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Optimized asset profile cache (in-memory + disk two-level cache)
# ---------------------------------------------------------------------------


class TwoLevelAssetProfileCache:
    """Two-level cache: L1 in-memory LRU → L2 disk JSON.

    Optimisation technique: *cache hierarchy*.  Reads first check the fast
    in-memory LRU.  On an L1 miss the disk is probed and the result is
    promoted back to L1.  Writes update both levels.  This mirrors the
    classic CPU cache hierarchy design used in high-performance computing.
    """

    def __init__(
        self,
        memory_maxsize: int = 32,
        cache_dir: str | Path = ".cache/scenarios_optimization",
        ttl_seconds: float | None = 3600 * 24,  # 24 h default TTL
    ) -> None:
        self._l1: _LRUCache[dict] = _LRUCache(maxsize=memory_maxsize)
        self._l2 = DiskCache(cache_dir=cache_dir, namespace="asset_profiles", ttl_seconds=ttl_seconds)

    def get(self, key: str) -> dict | None:
        # L1 check
        val = self._l1.get(key)
        if val is not None:
            return val
        # L2 check (disk)
        val = self._l2.get(key)
        if val is not None:
            # Promote to L1
            self._l1.put(key, val)
            return val
        return None

    def put(self, key: str, value: dict) -> None:
        self._l1.put(key, value)
        self._l2.put(key, value)

    def invalidate(self, key: str) -> None:
        self._l1.invalidate(key)
        self._l2.invalidate(key)

    @property
    def stats(self) -> dict[str, Any]:
        return {"l1": self._l1.stats}


# Module-level singleton two-level cache (shared across agent instances)
_TWO_LEVEL_PROFILE_CACHE = TwoLevelAssetProfileCache()
_TWO_LEVEL_DIGEST_CACHE = DiskCache(namespace="research_digests", ttl_seconds=3600 * 24)


# ---------------------------------------------------------------------------
# Early rejection filter
# ---------------------------------------------------------------------------


class EarlyRejectFilter:
    """Fast deterministic pre-screen for raw LLM-generated scenarios.

    Optimisation technique: *early rejection / fast path*.  Applying cheap
    O(n) string checks *before* the expensive structural ``validate_scenario``
    call (which runs regex pattern matching, SequenceMatcher dedup, etc.)
    eliminates clearly bad scenarios in microseconds rather than milliseconds.

    The filter is intentionally conservative — false negatives (passing bad
    scenarios) are acceptable because the downstream validator will catch them.
    False positives (rejecting good scenarios) hurt recall, so thresholds are
    set loosely.

    Parameters
    ----------
    min_text_length:
        Minimum number of characters in the ``text`` field.
    max_text_length:
        Maximum number of characters in the ``text`` field.
    min_characteristic_form_length:
        Minimum characters in ``characteristic_form``.
    jaccard_dedup_threshold:
        Jaccard similarity threshold on trigrams for fast near-duplicate
        detection.  Lower is stricter; default 0.80 is slightly looser than
        the SequenceMatcher 0.90 used by the full validator.
    """

    _REQUIRED_FIELDS = ("text", "category", "characteristic_form")

    def __init__(
        self,
        min_text_length: int = 30,
        max_text_length: int = 1500,
        min_characteristic_form_length: int = 20,
        jaccard_dedup_threshold: float = 0.80,
    ) -> None:
        self.min_text = min_text_length
        self.max_text = max_text_length
        self.min_cf = min_characteristic_form_length
        self.jac_thresh = jaccard_dedup_threshold
        # Pre-compiled tool-name leak pattern (populated lazily)
        self._tool_pattern: re.Pattern | None = None

    # ------------------------------------------------------------------

    def configure_tool_names(self, all_tool_names: Iterable[str]) -> None:
        """Pre-compile a single regex for all known tool names.

        Optimisation: compile once, match many times — avoids per-scenario
        regex recompilation overhead that occurs in the base validator.
        """
        sorted_names = sorted(set(all_tool_names), key=len, reverse=True)
        if not sorted_names:
            self._tool_pattern = None
            return
        escaped = "|".join(re.escape(name) for name in sorted_names)
        self._tool_pattern = re.compile(
            rf"(?<![a-z0-9_])({escaped})(?![a-z0-9_])",
            re.IGNORECASE,
        )

    def filter(
        self,
        scenarios: list[dict],
        accepted_texts: list[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Split *scenarios* into (passed, rejected) based on fast checks.

        Parameters
        ----------
        scenarios:
            Raw list of scenario dicts from the LLM.
        accepted_texts:
            Texts of already-accepted scenarios for dedup checking.

        Returns
        -------
        (passed, rejected):
            *passed* scenarios cleared all fast checks; *rejected* did not.
        """
        accepted_trigrams = [
            _trigrams(str(t)) for t in (accepted_texts or []) if t
        ]
        passed: list[dict] = []
        rejected: list[dict] = []
        accepted_so_far_trigrams: list[frozenset[str]] = list(accepted_trigrams)

        for scenario in scenarios:
            reasons = self._fast_check(scenario, accepted_so_far_trigrams)
            if reasons:
                _log.debug("EarlyReject: %s", reasons)
                rejected.append(scenario)
            else:
                passed.append(scenario)
                # Add this scenario's text to the rolling accepted set so
                # later scenarios in the same batch are dedup-checked against it.
                accepted_so_far_trigrams.append(
                    _trigrams(str(scenario.get("text", "")))
                )

        return passed, rejected

    # ------------------------------------------------------------------

    def _fast_check(
        self, scenario: dict, accepted_trigrams: list[frozenset[str]]
    ) -> list[str]:
        reasons: list[str] = []

        # 1. Required fields
        for fld in self._REQUIRED_FIELDS:
            val = scenario.get(fld)
            if not isinstance(val, str) or not val.strip():
                reasons.append(f"missing or blank field '{fld}'")

        if reasons:
            # If required fields are missing the remaining checks are moot
            return reasons

        text = scenario["text"].strip()
        cf = scenario["characteristic_form"].strip()

        # 2. Length checks
        if len(text) < self.min_text:
            reasons.append(f"text too short ({len(text)} < {self.min_text} chars)")
        if len(text) > self.max_text:
            reasons.append(f"text too long ({len(text)} > {self.max_text} chars)")
        if len(cf) < self.min_cf:
            reasons.append(f"characteristic_form too short ({len(cf)} < {self.min_cf} chars)")

        # 3. Tool-name leakage in text (pre-compiled single-pass regex)
        if self._tool_pattern and self._tool_pattern.search(text):
            reasons.append("tool name found in scenario text (pre-screen)")

        # 4. Fast near-duplicate check via trigram Jaccard similarity
        if accepted_trigrams:
            tg = _trigrams(text)
            if tg:
                for existing in accepted_trigrams:
                    score = _jaccard(tg, existing)
                    if score >= self.jac_thresh:
                        reasons.append(
                            f"near-duplicate text detected (fast Jaccard={score:.2f})"
                        )
                        break

        return reasons


def _trigrams(text: str) -> frozenset[str]:
    """Compute the set of character-level trigrams for *text*.

    Trigram Jaccard similarity is O(k) in the number of unique trigrams and
    has lower constant factors than SequenceMatcher, making it suitable for
    bulk pre-screening.
    """
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    if len(normalised) < 3:
        return frozenset()
    return frozenset(normalised[i : i + 3] for i in range(len(normalised) - 2))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute the Jaccard coefficient between two trigram sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Async parallelisation helpers
# ---------------------------------------------------------------------------


class AsyncBatchSemaphore:
    """Limits the number of concurrent async tasks in a batch pipeline.

    Optimisation technique: *bounded parallelism*.  Releasing too many
    concurrent LLM requests causes rate-limit errors and head-of-line
    blocking.  A semaphore with a carefully chosen limit maximises throughput
    without tripping rate limits.
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:  # type: ignore[override]
        async with self._sem:
            yield


# Re-export as async context manager helper
AsyncIterator = Iterator  # type alias for annotation only


async def run_in_executor(
    fn: Callable[..., T],
    *args: Any,
    executor: Executor | None = None,
) -> T:
    """Run a synchronous callable *fn* in an executor (thread or process pool).

    Optimisation technique: *offloading blocking I/O*.  Python's async event
    loop is single-threaded; blocking calls (e.g., synchronous LLM HTTP calls,
    disk reads) stall all other coroutines.  Running them in a thread pool
    keeps the event loop responsive.

    Parameters
    ----------
    fn:
        Synchronous callable to execute.
    *args:
        Positional arguments forwarded to *fn*.
    executor:
        Custom executor.  Defaults to the module-level ``ThreadPoolExecutor``.
    """
    loop = asyncio.get_event_loop()
    eff_executor = executor or _get_default_executor()
    return await loop.run_in_executor(eff_executor, fn, *args)


_DEFAULT_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _get_default_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Return (or lazily create) the module-level thread pool executor."""
    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _DEFAULT_EXECUTOR is None:
                _DEFAULT_EXECUTOR = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="scenarios_opt_worker",
                )
    return _DEFAULT_EXECUTOR


def shutdown_executor() -> None:
    """Cleanly shut down the shared thread pool executor."""
    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is not None:
        _DEFAULT_EXECUTOR.shutdown(wait=True)
        _DEFAULT_EXECUTOR = None


# ---------------------------------------------------------------------------
# Timing and observability
# ---------------------------------------------------------------------------


@dataclass
class PhaseTimingRecord:
    """Timing data for a single named pipeline phase."""

    name: str
    wall_seconds: float
    cpu_seconds: float
    start_epoch: float


class TimingRegistry:
    """Thread-safe store for phase timing records.

    Optimisation technique: *observability as a first-class concern*.
    Collecting wall/CPU times for each phase allows bottleneck identification
    without requiring a full profiler.  This is the lightweight equivalent of
    the PyTorch profiler spans used in ``scenarios_profiling``.
    """

    def __init__(self) -> None:
        self._records: list[PhaseTimingRecord] = []
        self._lock = threading.Lock()

    def record(self, rec: PhaseTimingRecord) -> None:
        with self._lock:
            self._records.append(rec)

    def to_dict(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": r.name,
                    "wall_seconds": round(r.wall_seconds, 4),
                    "cpu_seconds": round(r.cpu_seconds, 4),
                    "start_epoch": r.start_epoch,
                }
                for r in self._records
            ]

    def summary(self) -> str:
        rows = self.to_dict()
        if not rows:
            return "(no timing records)"
        lines = ["Phase Timing Summary", "=" * 50]
        for row in sorted(rows, key=lambda r: r["wall_seconds"], reverse=True):
            lines.append(
                f"  {row['name']:<45} {row['wall_seconds']:>8.3f}s wall  "
                f"{row['cpu_seconds']:>8.3f}s cpu"
            )
        return "\n".join(lines)


# Global timing registry instance
GLOBAL_TIMING = TimingRegistry()


@contextlib.asynccontextmanager
async def timed_section(name: str, registry: TimingRegistry | None = None) -> AsyncIterator[None]:  # type: ignore[override]
    """Async context manager that records wall and CPU time for a phase.

    Usage::

        async with timed_section("phase_1_build_profile"):
            profile = await agent.build_asset_profile(...)

    Parameters
    ----------
    name:
        Human-readable phase name.
    registry:
        Target ``TimingRegistry``.  Defaults to ``GLOBAL_TIMING``.
    """
    reg = registry or GLOBAL_TIMING
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    start_epoch = time.time()
    try:
        yield
    finally:
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        rec = PhaseTimingRecord(
            name=name,
            wall_seconds=end_wall - start_wall,
            cpu_seconds=end_cpu - start_cpu,
            start_epoch=start_epoch,
        )
        reg.record(rec)
        _log.debug(
            "Phase '%s' completed in %.3fs wall / %.3fs CPU",
            name,
            rec.wall_seconds,
            rec.cpu_seconds,
        )


# ---------------------------------------------------------------------------
# Progressive timeout
# ---------------------------------------------------------------------------


class ProgressiveTimeout:
    """Returns a progressively increasing timeout for successive LLM attempts.

    Optimisation technique: *progressive backoff with budget control*.  Early
    attempts use a short timeout so that fast failures (e.g. trivially bad
    prompts) are detected quickly.  Later attempts get more time because the
    LLM may need more tokens to produce complex repairs.

    Parameters
    ----------
    base_seconds:
        Timeout for the first attempt.
    multiplier:
        Factor by which the timeout grows on each subsequent attempt.
    max_seconds:
        Hard ceiling on the timeout.
    """

    def __init__(
        self,
        base_seconds: float = 30.0,
        multiplier: float = 1.5,
        max_seconds: float = 120.0,
    ) -> None:
        self._base = base_seconds
        self._mult = multiplier
        self._max = max_seconds

    def for_attempt(self, attempt: int) -> float:
        """Return the timeout in seconds for *attempt* (1-indexed)."""
        timeout = self._base * (self._mult ** (attempt - 1))
        return min(timeout, self._max)


# ---------------------------------------------------------------------------
# Token budget estimator
# ---------------------------------------------------------------------------


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in *text* without a full tokeniser.

    Optimisation technique: *token budget awareness*.  Sending prompts that
    exceed the context window causes hard API errors.  This fast heuristic
    (approx. 1 token per 4 characters of English text, tuned for GPT/Claude
    family models) lets callers truncate or summarise context *before* the API
    call, preventing costly retries.

    The estimate is intentionally conservative (slightly over-counts) to
    avoid under-budgeting.
    """
    # Fast path: count whitespace-separated words and scale
    # A standard rule of thumb is 0.75 words ≈ 1 token for English prose
    word_count = len(text.split())
    char_count = len(text)
    # Weighted blend: words-based estimate is better for prose; chars-based
    # for code/JSON.
    prose_estimate = int(word_count / 0.75)
    code_estimate = char_count // 4
    return max(prose_estimate, code_estimate)


def truncate_to_token_budget(text: str, max_tokens: int, *, suffix: str = "...[truncated]") -> str:
    """Truncate *text* to approximately *max_tokens* tokens.

    Uses ``estimate_token_count`` for speed.  If the text is already within
    budget it is returned unchanged.
    """
    if estimate_token_count(text) <= max_tokens:
        return text
    # Binary search for the right truncation point
    lo, hi = 0, len(text)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if estimate_token_count(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid
    return text[:lo] + suffix


# ---------------------------------------------------------------------------
# Deduplication helpers (vectorised set operations)
# ---------------------------------------------------------------------------


def build_trigram_index(texts: list[str]) -> list[frozenset[str]]:
    """Build a list of trigram sets for *texts* in a single pass.

    Optimisation technique: *vectorised pre-computation*.  Pre-computing all
    trigram sets once avoids recomputation on each pairwise similarity check.
    """
    return [_trigrams(t) for t in texts]


def fast_dedup_filter(
    candidates: list[str],
    accepted: list[str],
    threshold: float = 0.80,
) -> list[str]:
    """Return only those *candidates* that are not near-duplicates of *accepted*.

    Optimisation technique: *set-based deduplication*.  Using frozenset
    intersection/union is O(min(|A|, |B|)) per pair, much faster than the
    O(n²) character-level SequenceMatcher used in the base pipeline.
    """
    accepted_index = build_trigram_index(accepted)
    unique: list[str] = []
    rolling_index = list(accepted_index)

    for text in candidates:
        tg = _trigrams(text)
        if not tg:
            unique.append(text)
            continue
        is_dup = any(_jaccard(tg, existing) >= threshold for existing in rolling_index)
        if not is_dup:
            unique.append(text)
            rolling_index.append(tg)

    return unique


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """JSON-serialize *obj*, falling back to a string representation on error."""
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError) as exc:
        _log.warning("_safe_json_dumps fallback: %s", exc)
        return json.dumps(str(obj))


# ---------------------------------------------------------------------------
# PyTorch profiler — re-exported from scenarios_profiling for fair comparison
# ---------------------------------------------------------------------------
# Re-exporting from scenarios_profiling.profiling_utils means the optimized
# pipeline uses the *exact same* ProfilerConfig, build_profiler, and record()
# implementation as the non-optimized one.  This ensures chrome traces from
# both runs are produced by identical profiler infrastructure so wall-clock
# and CPU-time comparisons are apples-to-apples.

from scenarios_profiling.profiling_utils import (  # noqa: E402, F401
    ProfilerConfig,
    build_profiler,
    record as torch_record,
)

__all__ = [
    # Caching
    "AssetProfileCache",
    "DiskCache",
    "TwoLevelAssetProfileCache",
    "asset_profile_cache_key",
    "research_digest_cache_key",
    # Batching
    "BatchConfig",
    "chunk_list",
    # Early rejection
    "EarlyRejectFilter",
    # Parallelism
    "AsyncBatchSemaphore",
    "run_in_executor",
    "shutdown_executor",
    # Timing
    "GLOBAL_TIMING",
    "PhaseTimingRecord",
    "TimingRegistry",
    "timed_section",
    # Misc
    "ProgressiveTimeout",
    "estimate_token_count",
    "truncate_to_token_budget",
    "build_trigram_index",
    "fast_dedup_filter",
    # PyTorch profiler (re-exported for fair comparison)
    "ProfilerConfig",
    "build_profiler",
    "torch_record",
]
