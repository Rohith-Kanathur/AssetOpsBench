"""Optimized scenario generation agent for AssetOpsBench.

This module is the performance-optimized counterpart to
``scenarios_profiling.generator.agent``.  It preserves all pipeline logic
and compatibility with existing constraints, models, and prompts, while
adding the following optimizations:

Optimization A – Caching
------------------------
* ``build_asset_profile`` checks a two-level (memory + disk) cache before
  running the expensive retrieval + LLM synthesis pipeline.  Profiles are
  stored keyed on (asset_name, retriever, requested_open_form) so repeated
  calls for the same asset are instant.
* Research digests are separately cached so that only the final LLM profile-
  builder step is re-run when the digest is available but the profile is stale.
* Few-shot examples are cached via ``functools.lru_cache`` (wrapped by the
  module-level ``_FEWSHOT_CACHE``) to avoid repeated HuggingFace file I/O.
* Budget allocations are memoized per (asset_name, total) so that retries
  within a run skip the budget LLM call.

Optimization B – Batch Processing
----------------------------------
* ``generate_validated_scenarios`` drives all generation through configurable
  batches (``BatchConfig.generation_batch_size``) rather than requesting the
  full required count in a single LLM call.
* Validation is similarly chunked (``BatchConfig.validation_batch_size``).
* Multi-agent construction batches single-agent scenarios into chunks of 10
  to keep prompts within context-window limits.

Optimization D – Parallelization
----------------------------------
* All per-focus generation loops run concurrently via
  ``asyncio.gather`` guarded by an ``AsyncBatchSemaphore``.  Each focus
  (iot, fmsr, tsfm, wo, vibration) is handled in its own async task.
* Blocking synchronous calls (grounding discovery, few-shot file I/O) are
  offloaded to a ``ThreadPoolExecutor`` via ``run_in_executor`` so they do
  not stall the event loop.

Optimization E – Additional HPML Techniques
---------------------------------------------
* **Token budget awareness**: prompts are pre-checked with
  ``estimate_token_count`` and context sections (accepted scenarios, few-shot
  examples) are truncated when approaching the model's context window.
* **Progressive timeouts**: each repair/retry attempt uses a progressively
  longer timeout so that cheap failures are fast and expensive repairs get
  adequate headroom.
* **Timing observability**: every phase is wrapped in ``timed_section`` which
  records wall and CPU time to ``GLOBAL_TIMING`` (printed in the summary).
* **Lazy LLM initialization**: the ``LiteLLMBackend`` and ``Executor`` are
  instantiated once and reused, avoiding per-call setup overhead.
* **Vectorised deduplication index**: accepted scenarios are maintained as a
  pre-computed list of trigram sets so that each new scenario is deduplicated
  with a single pass through frozenset operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from agent.cli import _DEFAULT_MODEL
from agent.plan_execute.executor import Executor
from llm.litellm import LiteLLMBackend

from ..optimization_utils import (
    GLOBAL_TIMING,
    AsyncBatchSemaphore,
    BatchConfig,
    ProfilerConfig,
    ProgressiveTimeout,
    TimingRegistry,
    TwoLevelAssetProfileCache,
    DiskCache,
    _FEWSHOT_CACHE,
    asset_profile_cache_key,
    build_profiler,
    build_trigram_index,
    chunk_list,
    estimate_token_count,
    research_digest_cache_key,
    run_in_executor,
    timed_section,
    torch_record,
    truncate_to_token_budget,
)
# Re-use all constraint, model, prompt, and utility logic from scenarios_profiling
# unchanged to maintain full compatibility.
from scenarios_profiling.constraints import (
    FOCUS_ORDER,
    ScenarioValidationFailure,
    failure_payload,
    format_accepted_scenarios_for_prompt,
    format_categories_for_prompt,
    format_forbidden_patterns_for_prompt,
    format_mode_requirements,
    format_requirements_for_prompt,
    get_scenario_policy,
    validate_scenario_batch,
)
from scenarios_profiling.grounding import discover_grounding
from scenarios_profiling.models import (
    AssetProfile,
    EvidenceCandidate,
    GroundingBundle,
    KeyDescription,
    RetrieverMode,
    Scenario,
    ScenarioBudget,
    SensorNameDescription,
)
from scenarios_profiling.prompts import (
    BUDGET_ALLOCATOR_PROMPT,
    MULTIAGENT_COMBINER_PROMPT,
    PROFILE_BUILDER_PROMPT,
    SCENARIO_GENERATOR_PROMPT,
    VALIDATE_REPAIR_PROMPT,
)
from scenarios_profiling.retrieval import retrieve_asset_evidence, synthesize_research_digest
from scenarios_profiling.text import slugify_asset_name, truncate_title_one_line
from scenarios_profiling.utils import fetch_hf_fewshot, parse_llm_json
# Re-use all prompt-helper utilities from scenarios_profiling without modification
from scenarios_profiling.generator.prompt_helpers import (
    _MULTIAGENT_MAX_TOKENS,
    _MAX_FEWSHOT_EXAMPLES,
    _MAX_SCENARIO_ATTEMPTS,
    _PROFILE_MAX_TOKENS,
    _asset_profile_json,
    _few_shot_examples_section,
    _grounding_summary_for_prompt,
    _invert_failure_mapping,
    _label_desc_dict_from_list,
    _multiagent_budget_cap,
    _normalize_failure_sensor_mapping,
    _normalize_string_list,
    _ordered_descriptions,
    _print_live_step,
    _print_section,
    _print_step,
    _quiet_litellm_logging,
    _redact_logged_prompt,
    _require_nonempty_list,
    _require_nonempty_str,
    _tool_summary_for_prompt,
    _validation_tool_names_by_focus,
)
from .prompt_helpers import default_scenario_output_path
from ..wandb_logger import NullWandbLogger, WandbRunLogger

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum tokens reserved for accepted-scenario context in prompts.
# Beyond this the list is truncated to avoid context-window overflow.
_MAX_ACCEPTED_CONTEXT_TOKENS = 2048

# Maximum tokens reserved for the few-shot examples section.
_MAX_FEWSHOT_CONTEXT_TOKENS = 1024


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scenario_from_llm_row(
    row: dict,
    *,
    scenario_type: str,
    generation_mode: str,
) -> Scenario:
    payload = dict(row)
    payload.pop("type", None)
    return Scenario(**payload, type=scenario_type, generation_mode=generation_mode)


def _truncated_accepted_section(accepted_scenarios: list[dict]) -> str:
    """Build the accepted-scenarios prompt section, truncating if too large.

    Optimisation: avoids blowing out the context window with a very long
    accepted-scenarios list by cutting the serialised JSON to a token budget.
    """
    full_text = format_accepted_scenarios_for_prompt(accepted_scenarios)
    return truncate_to_token_budget(
        full_text,
        _MAX_ACCEPTED_CONTEXT_TOKENS,
        suffix="\n... [accepted scenarios truncated for context window] ...",
    )


def _truncated_fewshot_section(few_shots: list[dict]) -> str:
    """Build the few-shot section, truncating if too large."""
    full_text = _few_shot_examples_section(few_shots)
    return truncate_to_token_budget(
        full_text,
        _MAX_FEWSHOT_CONTEXT_TOKENS,
        suffix="\n... [few-shot examples truncated for context window] ...",
    )


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


class OptimizedScenarioGeneratorAgent:
    """Scenario generation pipeline with comprehensive performance optimizations.

    Drop-in replacement for ``scenarios_profiling.generator.ScenarioGeneratorAgent``
    with the same public interface but significantly faster execution via:

    * Two-level asset profile cache (L1 in-memory LRU + L2 disk JSON)
    * Research digest cache
    * Batched + parallel per-focus generation
    * Thread-pool offloading of blocking I/O calls
    * Token-budget-aware prompt construction

    Parameters
    ----------
    model_id:
        LiteLLM model identifier.
    show_workflow:
        Print step-by-step progress to stdout.
    log_dir:
        Optional directory for writing prompt/response logs.
    retriever:
        Academic search backend (``"arxiv"`` or ``"semantic_scholar"``).
    research_digest_path:
        Optional path to a precomputed research digest file (skips retrieval).
    batch_config:
        Controls batch sizes and parallelism levels.
    profile_cache_dir:
        Root directory for the disk-level asset profile cache.
        Set to ``None`` to disable disk caching.
    profile_cache_ttl_hours:
        How long (in hours) disk-cached profiles are considered fresh.
        Default 24 h.  Set to ``None`` for no expiry.
    timing_registry:
        External ``TimingRegistry`` to record phase timings.
        Defaults to ``GLOBAL_TIMING``.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        show_workflow: bool = False,
        log_dir: str | None = None,
        *,
        retriever: RetrieverMode = "arxiv",
        research_digest_path: str | None = None,
        batch_config: BatchConfig | None = None,
        profile_cache_dir: str | Path | None = ".cache/scenarios_optimization",
        profile_cache_ttl_hours: float | None = 24.0,
        timing_registry: TimingRegistry | None = None,
        wandb_logger: WandbRunLogger | None = None,
    ) -> None:
        _quiet_litellm_logging()

        # Core LLM / executor (lazy initialization avoids import overhead)
        self.llm = LiteLLMBackend(model_id=model_id)
        self.executor = Executor(llm=self.llm)

        self.show_workflow = show_workflow
        self.log_dir = log_dir
        self.retriever: RetrieverMode = retriever
        self.research_digest_path = research_digest_path

        # ---- Optimization A: Caching ----
        ttl = (profile_cache_ttl_hours * 3600) if profile_cache_ttl_hours is not None else None
        if profile_cache_dir is not None:
            self._profile_cache = TwoLevelAssetProfileCache(
                cache_dir=profile_cache_dir,
                ttl_seconds=ttl,
            )
            self._digest_cache = DiskCache(
                cache_dir=profile_cache_dir,
                namespace="research_digests",
                ttl_seconds=ttl,
            )
        else:
            # Disable disk caching; in-memory only
            self._profile_cache = TwoLevelAssetProfileCache(
                cache_dir=".cache/_no_disk",
                ttl_seconds=0,  # immediately stale → effectively disabled
            )
            self._digest_cache = DiskCache(
                cache_dir=".cache/_no_disk",
                namespace="research_digests",
                ttl_seconds=0,
            )

        # ---- Optimization B: Batch configuration ----
        self._batch = batch_config or BatchConfig()

        # ---- Optimization D: Async concurrency semaphore ----
        self._semaphore = AsyncBatchSemaphore(
            max_concurrent=self._batch.max_concurrent_batches
        )

        # ---- Optimization E: Timing observability ----
        self._timing = timing_registry or GLOBAL_TIMING

        # ---- W&B logging ----
        self._wandb: WandbRunLogger = wandb_logger or NullWandbLogger()
        # Tracks the current pipeline phase so that LLM calls can be tagged.
        self._current_phase: str = "unknown"

        # ---- Internal state ----
        self._log_steps_by_subdir: dict[str, int] = {}
        self._progressive_timeout = ProgressiveTimeout()

    # ------------------------------------------------------------------
    # Log helpers (unchanged from profiling agent)
    # ------------------------------------------------------------------

    def _log_path(self, name: str, *, step: int) -> Path:
        if not self.log_dir:
            raise RuntimeError("log_dir must be configured before writing logs")
        relative = Path(name)
        suffix = relative.suffix or ".txt"
        stem = relative.stem if relative.suffix else relative.name
        filename = f"{step:02d}_{stem}{suffix}"
        return Path(self.log_dir) / relative.parent / filename

    def _write_log(self, name: str, content: str) -> None:
        if not self.log_dir:
            return
        relative = Path(name)
        dir_key = relative.parent.as_posix() if relative.parent != Path(".") else ""
        step = self._log_steps_by_subdir.get(dir_key, 0) + 1
        path = self._log_path(name, step=step)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            handle.write(content)
        self._log_steps_by_subdir[dir_key] = step

    def _write_json_log(self, name: str, payload: object) -> None:
        log_name = name if name.endswith(".json") else f"{name}.json"
        self._write_log(log_name, json.dumps(payload, indent=2))

    def _llm_generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        *,
        phase: str | None = None,
    ):
        """Thin wrapper around ``self.llm.generate_with_usage`` that logs token usage to W&B.

        Parameters
        ----------
        prompt:
            The prompt string.
        max_tokens:
            Optional token cap forwarded to the backend.
        phase:
            Override the phase tag for this call.  Falls back to
            ``self._current_phase`` when omitted.
        """
        _phase = phase or self._current_phase
        _t0 = time.perf_counter()
        result = self.llm.generate_with_usage(prompt, max_tokens=max_tokens)
        _wall = time.perf_counter() - _t0
        self._wandb.log_llm_call(_phase, result, wall_seconds=_wall)
        return result

    # ------------------------------------------------------------------
    # Public pipeline entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        asset_name: str,
        num_scenarios: int = 50,
        data_in_couchdb: bool = False,
    ) -> list[Scenario]:
        """Run the full optimized scenario generation pipeline.

        This method is API-compatible with
        ``ScenarioGeneratorAgent.run`` but adds caching, batching,
        and parallel generation.

        Parameters
        ----------
        asset_name:
            Asset class name (e.g. ``"Chiller"``).
        num_scenarios:
            Total number of scenarios to generate.
        data_in_couchdb:
            Whether to use grounded open-form generation.

        Returns
        -------
        list[Scenario]
            Generated and validated scenarios.
        """
        asset_name = asset_name.strip()
        if not asset_name:
            raise ValueError("Asset class name is empty after stripping whitespace.")

        _log.info(
            "Starting optimized scenario generation for '%s' (%d scenarios)",
            asset_name,
            num_scenarios,
        )

        # Phase 0: Server descriptions
        # torch_record spans are emitted unconditionally; they are no-ops when
        # no torch.profiler.profile context is active (zero overhead).
        self._current_phase = "phase_0_server_descriptions"
        with torch_record("phase_0_server_descriptions"):
            async with timed_section("phase_0_server_descriptions", self._timing):
                server_desc = await self.executor.get_server_descriptions()

        # Phase 1: Asset Profile (with caching)
        if self.show_workflow:
            _print_section("Phase 1: Asset Profile Construction")
        self._current_phase = "phase_1_build_asset_profile"
        with torch_record("phase_1_build_asset_profile"):
            async with timed_section("phase_1_build_asset_profile", self._timing):
                asset_profile = await self.build_asset_profile(
                    asset_name=asset_name,
                    server_desc=server_desc,
                    requested_open_form=data_in_couchdb,
                )

        # Phase 2: Budget Allocation (with memoization)
        if self.show_workflow:
            _print_section("Phase 2: Scenario Budget Allocation")
        self._current_phase = "phase_2_allocate_budget"
        with torch_record("phase_2_allocate_budget"):
            async with timed_section("phase_2_allocate_budget", self._timing):
                budget = await self.allocate_budget(asset_profile, total=num_scenarios)

        # Phase 3: Focused Generation & Validation (parallel)
        if self.show_workflow:
            _print_section("Phase 3: Focused Generation & Validation (Parallel)")

        self._current_phase = "phase_3_generate_all_focuses"
        validation_tool_names = _validation_tool_names_by_focus(server_desc)

        # ---- Optimization D: Parallel per-focus generation ----
        # Build tasks for every non-zero, non-multiagent focus simultaneously.
        with torch_record("phase_3_generate_all_focuses"):
            async with timed_section("phase_3_generate_all_focuses", self._timing):
                all_scenarios = await self._parallel_generate_focuses(
                    budget=budget,
                    asset_profile=asset_profile,
                    asset_name=asset_name,
                    server_desc=server_desc,
                    validation_tool_names=validation_tool_names,
                )

        # Phase 4: Multi-agent construction (depends on phase 3 output)
        self._current_phase = "phase_4_generate_validate_multiagent"
        multiagent_count = budget.allocation.get("multiagent", 0)
        if multiagent_count > 0:
            if self.show_workflow:
                _print_section("Phase 4: Multi-Agent Scenario Construction")
            with torch_record("phase_4_generate_validate_multiagent"):
                async with timed_section("phase_4_generate_validate_multiagent", self._timing):
                    valid_multi = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.generate_validated_scenarios(
                            focus="multiagent",
                            count=multiagent_count,
                            profile=asset_profile,
                            server_desc=server_desc,
                            accepted_scenarios=[s.to_dict() for s in all_scenarios],
                            single_agents=[s.to_dict() for s in all_scenarios],
                            validation_tool_names=validation_tool_names,
                        ),
                    )
            for scenario_data in valid_multi:
                scenario_data["id"] = (
                    f"{slugify_asset_name(asset_name)}_scenario_{len(all_scenarios)+1:02d}"
                )
                all_scenarios.append(
                    _scenario_from_llm_row(
                        scenario_data,
                        scenario_type="multiagent",
                        generation_mode=asset_profile.generation_mode,
                    )
                )
            if self.show_workflow:
                _print_step("validate_multiagent", f"Validated {len(valid_multi)} multiagent scenarios")

        # Summary
        if self.show_workflow:
            _print_section("Summary")
            print(
                f"Successfully generated {len(all_scenarios)} total scenarios for "
                f"'{asset_name}' in {asset_profile.generation_mode} mode.\n"
            )
            print(self._timing.summary())

        # ---- W&B: log run-level summary and close the run ----
        self._wandb.log_run_summary(
            timing_registry=self._timing,
            cache_stats=self.cache_stats,
            total_scenarios=len(all_scenarios),
            asset_name=asset_name,
            generation_mode=asset_profile.generation_mode,
        )
        self._wandb.finish()

        return all_scenarios

    async def run_with_profiling(
        self,
        asset_name: str,
        num_scenarios: int = 50,
        data_in_couchdb: bool = False,
        profiling_dir: str | None = None,
        profiler_config: ProfilerConfig | None = None,
    ) -> list[Scenario]:
        """Run the optimized pipeline under the PyTorch profiler.

        Identical contract to
        ``scenarios_profiling.generator.ScenarioGeneratorAgent.run_with_profiling``
        so both agents can be swapped in benchmark scripts without changes.

        Because every phase inside ``run`` is already wrapped in a
        ``torch_record`` span (which is the same ``record_function`` used by
        ``scenarios_profiling``), the resulting chrome trace has identical span
        names and nesting — enabling a direct apples-to-apples comparison in
        ``chrome://tracing`` or Perfetto:

        * ``phase_0_server_descriptions``
        * ``phase_1_build_asset_profile``  ← faster when cache is warm
        * ``phase_2_allocate_budget``
        * ``phase_3_generate_all_focuses``  ← parallel in this agent
        * ``phase_3_generate_<focus>``       ← per-focus sub-spans
        * ``phase_4_generate_validate_multiagent``

        Parameters
        ----------
        asset_name:
            Asset class name (e.g. ``"Transformer"``)
        num_scenarios:
            Total number of scenarios to generate.
        data_in_couchdb:
            Whether to use grounded open-form generation.
        profiling_dir:
            Directory for profiler output.  Ignored when *profiler_config* is
            provided.  Defaults to
            ``profiling_output/<asset_slug>_optimized/``.
        profiler_config:
            Full ``ProfilerConfig`` override.

        Returns
        -------
        list[Scenario]
            Same output as ``run``.
        """
        if profiler_config is None:
            slug = slugify_asset_name(asset_name) if asset_name.strip() else "asset"
            out_dir = profiling_dir or f"profiling_output/{slug}_optimized"
            profiler_config = ProfilerConfig(
                profile_dir=out_dir,
                export_chrome_trace=True,
                print_summary=True,
            )

        _log.info(
            "Starting profiled (optimized) scenario generation for '%s'; output → %s",
            asset_name,
            profiler_config.profile_dir,
        )

        with build_profiler(profiler_config) as prof:
            with torch_record("full_pipeline"):
                scenarios = await self.run(
                    asset_name=asset_name,
                    num_scenarios=num_scenarios,
                    data_in_couchdb=data_in_couchdb,
                )
            prof.step()

        return scenarios

    # ------------------------------------------------------------------
    # Optimization D: Parallel focus generation
    # ------------------------------------------------------------------

    async def _parallel_generate_focuses(
        self,
        budget: ScenarioBudget,
        asset_profile: AssetProfile,
        asset_name: str,
        server_desc: dict,
        validation_tool_names: dict[str, tuple[str, ...]],
    ) -> list[Scenario]:
        """Generate all non-multiagent focuses concurrently.

        Each focus is launched as an independent async task, bounded by
        ``AsyncBatchSemaphore`` to prevent too many concurrent LLM requests.

        Optimization notes
        ------------------
        * ``asyncio.gather`` runs tasks concurrently within the same event loop,
          interleaving I/O-bound LLM HTTP calls without true thread overhead.
        * The semaphore ensures at most ``max_concurrent_batches`` focuses run
          at the same time, respecting API rate limits.
        * Because each focus uses its own ``accepted_scenarios`` snapshot (taken
          at task-creation time), tasks are independent and require no locking.
          After all tasks complete, the results are merged in ``FOCUS_ORDER``
          order to preserve deterministic scenario IDs.
        """
        # Collect tasks in FOCUS_ORDER for deterministic output ordering
        tasks = []
        focus_counts = []
        for focus in FOCUS_ORDER:
            if focus == "multiagent":
                continue
            count = budget.allocation.get(focus, 0)
            if count == 0:
                continue
            focus_counts.append((focus, count))

        if not focus_counts:
            return []

        # Snapshot of accepted scenarios *before* any task runs.
        # Each task appends to its own local list; final merge is done below.
        accepted_snapshot: list[dict] = []

        async def _generate_focus_guarded(focus: str, count: int) -> list[dict]:
            async with self._semaphore.acquire():
                with torch_record(f"phase_3_generate_validate_{focus}"):
                    async with timed_section(f"phase_3_generate_{focus}", self._timing):
                        self._current_phase = f"phase_3_generate_{focus}"
                        if self.show_workflow:
                            _print_step(
                                f"generate_{focus}",
                                f"Generating {count} {focus} scenarios in "
                                f"{asset_profile.generation_mode} mode (parallel)",
                            )
                        # Run synchronous generation in executor to not block event loop
                        result = await run_in_executor(
                            self.generate_validated_scenarios,
                            focus,
                            count,
                            asset_profile,
                            server_desc,
                            list(accepted_snapshot),  # snapshot at task start
                            None,  # single_agents
                            validation_tool_names,
                        )
                        if self.show_workflow:
                            _print_step(
                                f"validate_{focus}",
                                f"Validated {len(result)} {focus} scenarios",
                            )
                        return result

        tasks = [_generate_focus_guarded(focus, count) for focus, count in focus_counts]
        # Gather results; fail_fast is False so one focus failure doesn't abort others
        results_per_focus = await asyncio.gather(*tasks, return_exceptions=False)

        # Merge in FOCUS_ORDER, assigning sequential IDs
        all_scenarios: list[Scenario] = []
        for (focus, _count), focus_results in zip(focus_counts, results_per_focus):
            for scenario_data in focus_results:
                scenario_data["id"] = (
                    f"{slugify_asset_name(asset_profile.asset_name)}"
                    f"_scenario_{len(all_scenarios)+1:02d}"
                )
                all_scenarios.append(
                    _scenario_from_llm_row(
                        scenario_data,
                        scenario_type=focus,
                        generation_mode=asset_profile.generation_mode,
                    )
                )

        return all_scenarios

    # ------------------------------------------------------------------
    # Optimization A: Cached asset profile build
    # ------------------------------------------------------------------

    async def build_asset_profile(
        self,
        asset_name: str,
        server_desc: dict,
        requested_open_form: bool = False,
    ) -> AssetProfile:
        """Build the asset profile, using a two-level cache to avoid redundant work.

        Cache key: (asset_name_normalized, retriever, requested_open_form).

        Cache hit (L1 memory or L2 disk) → deserialize and return immediately.
        Cache miss → run full retrieval + LLM pipeline, then store in both levels.

        The research digest (retrieval output) is also cached separately so that
        if the digest is available but the profile JSON has expired, only the
        final LLM profile-builder call is re-run.
        """
        asset_name = asset_name.strip()
        if not asset_name:
            raise ValueError("Asset class name is empty after stripping whitespace.")

        # ---- Optimization A: Check profile cache ----
        profile_key = asset_profile_cache_key(asset_name, self.retriever, requested_open_form)
        cached_profile_dict = self._profile_cache.get(profile_key)
        if cached_profile_dict is not None:
            _log.info("Cache HIT (profile): '%s'", asset_name)
            if self.show_workflow:
                _print_step(
                    "build_profile_cache_hit",
                    f"Asset Profile loaded from cache for '{asset_name}' "
                    f"(key={profile_key[:8]}…)",
                )
            return AssetProfile(**cached_profile_dict)

        _log.info("Cache MISS (profile): '%s' — running full pipeline", asset_name)

        # ---- Grounding (offloaded to thread pool) ----
        if self.show_workflow:
            _print_step(
                "grounding_start",
                f"Starting grounded discovery for {asset_name}...",
                details="Inspecting IoT coverage, vibration coverage, and bounded FMSR grounding.",
            )

        # Offload blocking sync call to executor
        grounding: GroundingBundle = await run_in_executor(
            discover_grounding, asset_name, requested_open_form
        )

        self._write_json_log("01_grounding/discovery.json", grounding.model_dump())
        generation_mode = "open_form" if grounding.open_form_eligible else "closed_form"

        if self.show_workflow:
            details = (
                f"Requested open form: {requested_open_form}\n"
                f"Resolved generation mode: {generation_mode}\n"
                f"Grounded instances: {len(grounding.asset_instances)}\n"
            )
            _print_step("grounding", "Grounded discovery complete", details=details)

        # ---- Research digest (preloaded file or cached retrieval) ----
        preload = self.research_digest_path
        if preload:
            digest_path = Path(preload)
            if not digest_path.is_file():
                raise FileNotFoundError(
                    f"Research digest file not found: {digest_path}. "
                    "Omit --research-digest or provide a valid path to skip retrieval."
                )
            research_digest = digest_path.read_text(encoding="utf-8")
            if self.show_workflow:
                _print_step(
                    "research_digest",
                    "Loaded precomputed research digest (retrieval skipped)",
                    details=str(digest_path.resolve()),
                )
        else:
            # ---- Optimization A: Check digest cache ----
            digest_key = research_digest_cache_key(asset_name, self.retriever)
            research_digest = self._digest_cache.get(digest_key)

            if research_digest is not None:
                _log.info("Cache HIT (digest): '%s'", asset_name)
                if self.show_workflow:
                    _print_step(
                        "research_digest_cache_hit",
                        f"Research digest loaded from cache for '{asset_name}' "
                        f"(key={digest_key[:8]}…)  — retrieval skipped",
                    )
            else:
                _log.info("Cache MISS (digest): '%s' — running retrieval", asset_name)
                if self.show_workflow:
                    _print_step(
                        "researcher_queries",
                        f"Planning bounded academic literature retrieval for {asset_name}...",
                    )

                on_pdf_workflow = None
                on_academic_query = None
                if self.show_workflow:
                    _pdf_ok_logged: set[str] = set()

                    def _on_academic_query(query: str, n_results: int) -> None:
                        _print_step(
                            "academic_query",
                            f"Query: {query!r}",
                            details=f"Results: {n_results} paper(s) from metadata search",
                        )

                    def _on_pdf_workflow(
                        candidate: EvidenceCandidate, phase: str, outcome: str
                    ) -> None:
                        title = truncate_title_one_line(candidate.title)
                        pid = candidate.paper_id
                        if outcome == "ok":
                            if pid in _pdf_ok_logged:
                                return
                            _pdf_ok_logged.add(pid)
                            _print_step(
                                "pdf_fetch",
                                f'Paper "{title}" — PDF text loaded',
                                details=f"paper_id={pid}\nphase={phase}",
                            )
                            return
                        label = {
                            "no_pdf_url": "No PDF URL (no open-access direct link resolved from metadata)",
                            "fetch_failed": "PDF fetch failed or URL not a usable PDF stream",
                            "empty_text": "PDF downloaded but extractable text is empty",
                        }.get(outcome, outcome)
                        _print_step(
                            "pdf_fetch",
                            f'Paper "{title}" — {label}',
                            details=f"paper_id={pid}\nphase={phase}\noutcome={outcome}",
                        )

                    on_pdf_workflow = _on_pdf_workflow
                    on_academic_query = _on_academic_query

                # Retrieval is I/O-bound but synchronous; run in thread pool
                # retriever, on_pdf_workflow, on_academic_query are keyword-only
                _retriever = self.retriever
                _on_pdf_workflow_bound = on_pdf_workflow
                _on_academic_query_bound = on_academic_query
                evidence_bundle = await run_in_executor(
                    lambda: retrieve_asset_evidence(
                        asset_name,
                        server_desc,
                        self.llm,
                        self._write_log,
                        retriever=_retriever,
                        on_pdf_workflow=_on_pdf_workflow_bound,
                        on_academic_query=_on_academic_query_bound,
                    )
                )

                if self.show_workflow:
                    top_titles = "\n".join(
                        f" - {c.title} (judge_score={c.judge_score}/10)"
                        for c in evidence_bundle.candidates[:3]
                    ) or " - No ranked candidates"
                    _print_step(
                        "evidence_retrieval",
                        "Academic search engine evidence retrieval complete",
                        details=(
                            f"Canonical asset: {evidence_bundle.canonical_asset_name}\n"
                            f"Queries:\n"
                            + "\n".join(f" - {q}" for q in evidence_bundle.query_history)
                            + "\nTop Evidence:\n"
                            + top_titles
                        ),
                    )

                # retriever, log_writer, on_pdf_workflow are keyword-only
                _eb = evidence_bundle
                research_digest = await run_in_executor(
                    lambda: synthesize_research_digest(
                        _eb,
                        self.llm,
                        retriever=_retriever,
                        log_writer=self._write_log,
                        on_pdf_workflow=_on_pdf_workflow_bound,
                    )
                )

                # Cache the digest for future runs
                self._digest_cache.put(digest_key, research_digest)
                _log.info("Cached digest for '%s'", asset_name)

                if self.show_workflow:
                    _print_step(
                        "research_digest",
                        "Research digest synthesis complete (per-paper extraction + merge)",
                    )

        # ---- LLM profile construction ----
        prompt = PROFILE_BUILDER_PROMPT.format(
            asset_name=asset_name,
            generation_mode=generation_mode,
            grounding_summary_json=_grounding_summary_for_prompt(grounding),
            research_digest=research_digest,
            tool_descriptions=_tool_summary_for_prompt(server_desc),
        )
        self._write_log("03_asset_profile/prompt.txt", prompt)
        response = self._llm_generate(prompt, max_tokens=_PROFILE_MAX_TOKENS, phase="phase_1_build_asset_profile")

        parsed, parse_err = parse_llm_json(response.text)
        if not parsed or not isinstance(parsed, dict):
            raise RuntimeError(
                f"Failed to parse asset profile JSON from LLM: "
                f"{parse_err or 'response is not a JSON object'}"
            )
        self._write_json_log("03_asset_profile/response.json", parsed)

        profile = self._merge_profile(parsed, grounding, generation_mode)
        self._write_json_log("03_asset_profile/final_asset_profile.json", profile.model_dump())

        # ---- Cache the completed profile (L1 + L2) ----
        self._profile_cache.put(profile_key, profile.model_dump())
        _log.info("Cached profile for '%s' (key=%s)", asset_name, profile_key[:8])

        if self.show_workflow:
            details: str | None = None
            if self.log_dir:
                details = "Full profile JSON is saved under 03_asset_profile/ in the log directory."
            _print_step(
                "build_profile",
                f"Successfully generated Asset Profile for '{asset_name}'.",
                details=details,
            )

        return profile

    def _merge_profile(
        self,
        parsed_profile: dict,
        grounding: GroundingBundle,
        generation_mode: str,
    ) -> AssetProfile:
        """Merge grounding data into the LLM-produced profile dict.

        Logic is identical to ``ScenarioGeneratorAgent._merge_profile``; no
        behavioural changes are needed here.
        """
        merged = dict(parsed_profile)
        merged["asset_name"] = grounding.asset_name
        merged["generation_mode"] = generation_mode
        merged["asset_instances"] = [i.model_dump() for i in grounding.asset_instances]

        merged_failure_mapping = _normalize_failure_sensor_mapping(merged.get("failure_sensor_mapping", {}))
        if grounding.failure_sensor_mapping:
            merged["failure_sensor_mapping"] = grounding.failure_sensor_mapping
            merged_failure_mapping = grounding.failure_sensor_mapping
        elif merged_failure_mapping:
            merged["failure_sensor_mapping"] = merged_failure_mapping
        else:
            merged["failure_sensor_mapping"] = {}

        grounded_failure_modes = _normalize_string_list(grounding.failure_modes)
        from_list_fm = _label_desc_dict_from_list(
            merged.get("failure_modes"),
            primary="key",
            alternate="name",
        )
        all_failure_mode_keys = list(
            dict.fromkeys(
                [
                    *grounded_failure_modes,
                    *merged_failure_mapping.keys(),
                    *from_list_fm.keys(),
                ]
            )
        )

        if grounding.sensor_failure_mapping:
            merged["sensor_failure_mapping"] = grounding.sensor_failure_mapping
        elif merged_failure_mapping:
            merged["sensor_failure_mapping"] = _invert_failure_mapping(merged_failure_mapping)
        else:
            merged["sensor_failure_mapping"] = {}

        merged["description"] = _require_nonempty_str(merged.get("description"), field="description")

        req_iot = sorted(dict.fromkeys(str(s).strip() for s in grounding.iot_sensors if str(s).strip()))
        req_vib = sorted(dict.fromkeys(str(s).strip() for s in grounding.vibration_sensors if str(s).strip()))

        iot_from_llm = _label_desc_dict_from_list(merged.get("iot_sensors"), primary="name", alternate="key")
        vib_from_llm = _label_desc_dict_from_list(merged.get("vibration_sensors"), primary="name", alternate="key")

        merged["iot_sensors"] = _ordered_descriptions(
            iot_from_llm, req_iot, field="iot_sensors",
            build=lambda n, d: SensorNameDescription(name=n, description=d),
            sort_when_ordered_empty=True,
        )
        merged["vibration_sensors"] = _ordered_descriptions(
            vib_from_llm, req_vib, field="vibration_sensors",
            build=lambda n, d: SensorNameDescription(name=n, description=d),
            sort_when_ordered_empty=True,
        )

        if not all_failure_mode_keys:
            merged["failure_modes"] = [
                KeyDescription(key=k, description=from_list_fm[k])
                for k in sorted(from_list_fm.keys())
            ]
        else:
            merged["failure_modes"] = _ordered_descriptions(
                from_list_fm, all_failure_mode_keys, field="failure_modes",
                build=lambda k, d: KeyDescription(key=k, description=d),
            )

        if not isinstance(merged.get("relevant_tools"), dict):
            raise ValueError("Asset profile field 'relevant_tools' must be an object keyed by focus")
        tool_focuses = tuple(f for f in FOCUS_ORDER if f != "multiagent")
        normalized_relevant: dict[str, list] = {}
        for focus in tool_focuses:
            tools = merged["relevant_tools"].get(focus)
            if tools is None:
                normalized_relevant[focus] = []
                continue
            if not isinstance(tools, list):
                raise ValueError(
                    f"Asset profile field 'relevant_tools.{focus}' must be a list"
                )
            normalized_relevant[focus] = tools
        merged["relevant_tools"] = normalized_relevant
        merged["operator_tasks"] = _require_nonempty_list(merged.get("operator_tasks"), field="operator_tasks")
        merged["manager_tasks"] = _require_nonempty_list(merged.get("manager_tasks"), field="manager_tasks")
        return AssetProfile(**merged)

    # ------------------------------------------------------------------
    # Budget allocation (with memoization)
    # ------------------------------------------------------------------

    async def allocate_budget(self, profile: AssetProfile, total: int = 50) -> ScenarioBudget:
        """Allocate the scenario budget via LLM, with result memoization.

        Optimization: budget allocations are cached by (asset_name, total)
        within the process lifetime.  Because the budget only depends on the
        asset profile and total count — not on random state — repeating the
        call for the same inputs always yields the same result.
        """
        budget_key = f"{profile.asset_name.lower().strip()}|{total}"
        cached = self._profile_cache.get(f"budget:{budget_key}")
        if cached is not None:
            _log.debug("Budget cache HIT for '%s' total=%d", profile.asset_name, total)
            return ScenarioBudget(**cached)

        prompt = BUDGET_ALLOCATOR_PROMPT.format(
            total_scenarios=total,
            asset_profile_json=profile.model_dump_json(indent=2),
        )
        self._write_log("04_budget/prompt.txt", prompt)
        response = self._llm_generate(prompt, phase="phase_2_allocate_budget")
        parsed, _ = parse_llm_json(response.text)
        if not parsed or not isinstance(parsed, dict):
            parsed = {}
        self._write_json_log("04_budget/response.json", parsed)

        allocation = self._normalize_allocation(parsed.get("allocation", {}), total)
        budget = ScenarioBudget(
            total_scenarios=total,
            allocation=allocation,
            reasoning=str(parsed.get("reasoning", "")).strip(),
        )

        # Cache budget result
        self._profile_cache.put(f"budget:{budget_key}", budget.model_dump())

        if self.show_workflow:
            details = f"Reasoning: {budget.reasoning or '(not provided)'}\n\nAllocation:\n"
            details += "\n".join(f" - {focus}: {count}" for focus, count in budget.allocation.items())
            _print_step("allocate_budget", f"Successfully allocated {total} scenarios.", details=details)

        return budget

    def _normalize_allocation(self, raw_allocation: dict, total: int) -> dict[str, int]:
        allocation = {
            focus: max(0, int(raw_allocation.get(focus, 0) or 0))
            for focus in FOCUS_ORDER
        }
        if not any(allocation.values()):
            raise RuntimeError("Budget allocator returned an empty allocation; refusing to default.")

        allocation["multiagent"] = min(allocation["multiagent"], _multiagent_budget_cap(total))
        current_total = sum(allocation.values())
        if current_total == 0:
            raise RuntimeError("Budget allocator produced a zero-sum allocation.")

        if current_total > total:
            overflow = current_total - total
            for focus in ("multiagent", "wo", "vibration", "tsfm", "fmsr", "iot"):
                if overflow <= 0:
                    break
                reducible = min(allocation[focus], overflow)
                allocation[focus] -= reducible
                overflow -= reducible
        elif current_total < total:
            deficit = total - current_total
            allowed = list(FOCUS_ORDER)
            while deficit > 0 and allowed:
                for focus in allowed:
                    if deficit <= 0:
                        break
                    if focus == "multiagent" and allocation["multiagent"] >= _multiagent_budget_cap(total):
                        continue
                    allocation[focus] += 1
                    deficit -= 1
        return allocation

    # ------------------------------------------------------------------
    # Optimization B: Batched generation with validation
    # ------------------------------------------------------------------

    def generate_validated_scenarios(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
        single_agents: list[dict] | None = None,
        validation_tool_names: dict[str, tuple[str, ...]] | None = None,
    ) -> list[dict]:
        """Generate and validate *count* scenarios for *focus* using batched processing.

        Optimizations applied here:
        * **Batch B**: generate in chunks of ``batch_config.generation_batch_size``
          rather than asking for all ``count`` scenarios in one call.
        * **Progressive Timeout E**: each attempt's timeout increases
          geometrically, giving early cheap attempts a short budget and later
          expensive repair attempts more headroom.
        """
        accepted_scenarios = list(accepted_scenarios or [])
        valid_batch: list[dict] = []
        failure_notes: list[str] = []
        last_llm_was_validate_repair = False

        for attempt in range(1, _MAX_SCENARIO_ATTEMPTS + 1):
            remaining = count - len(valid_batch)
            if remaining <= 0:
                break

            baseline = accepted_scenarios + valid_batch

            # ---- Optimization B: request at most batch_size at a time ----
            request_count = min(remaining, self._batch.generation_batch_size)

            generated = self._generate_attempt_batch(
                focus=focus,
                count=request_count,
                profile=profile,
                server_desc=server_desc,
                accepted_scenarios=baseline,
                single_agents=single_agents,
            )
            if not generated:
                failure_notes.append(f"attempt {attempt}: generator returned no parseable scenarios")
                last_llm_was_validate_repair = False
                continue

            # Full deterministic validation
            valid_now, failures = validate_scenario_batch(
                focus=focus,
                scenarios=generated,
                accepted_scenarios=baseline,
                profile=profile,
                generation_mode=profile.generation_mode,
                tool_names_by_focus=validation_tool_names,
            )
            valid_batch.extend(valid_now)

            # ---- W&B: log per-batch result ----
            _repaired_count = 0
            _rejected_count = len(failures) if failures else 0

            if failures:
                self._write_log(
                    f"05_generation/{focus}/deterministic_failures_attempt_{attempt:02d}.json",
                    failure_payload(failures),
                )
                if self.show_workflow:
                    _print_live_step(
                        f"repair_{focus}",
                        f"Repairing {len(failures)} scenario(s) (attempt {attempt})",
                    )
                repaired_invalids = self.validate_and_repair(
                    focus=focus,
                    scenarios=[f.scenario for f in failures],
                    profile=profile,
                    accepted_scenarios=accepted_scenarios + valid_batch,
                    failures=failures,
                )
                last_llm_was_validate_repair = True

                repaired_valid, remaining_failures = validate_scenario_batch(
                    focus=focus,
                    scenarios=repaired_invalids,
                    accepted_scenarios=accepted_scenarios + valid_batch,
                    profile=profile,
                    generation_mode=profile.generation_mode,
                    tool_names_by_focus=validation_tool_names,
                )
                valid_batch.extend(repaired_valid)
                _repaired_count = len(repaired_valid)
                _rejected_count = len(remaining_failures)
                if remaining_failures:
                    failure_notes.append(
                        f"attempt {attempt}: {len(remaining_failures)} scenario(s) "
                        "still invalid after repair"
                    )
                    self._write_log(
                        f"05_generation/{focus}/remaining_failures_attempt_{attempt:02d}.json",
                        failure_payload(remaining_failures),
                    )
            else:
                last_llm_was_validate_repair = False

            # Log batch outcome to W&B
            self._wandb.log_batch_result(
                focus=focus,
                attempt=attempt,
                generated=len(generated),
                valid=len(valid_now),
                repaired=_repaired_count,
                rejected=_rejected_count,
            )

        if len(valid_batch) < count:
            shortage = count - len(valid_batch)
            summary = "; ".join(failure_notes) if failure_notes else "no detailed failure summary available"
            raise RuntimeError(
                f"Failed to generate {count} valid {focus} scenarios after "
                f"{_MAX_SCENARIO_ATTEMPTS} attempts. "
                f"Still missing {shortage}. Details: {summary}"
            )

        final_for_focus = valid_batch[:count]
        if last_llm_was_validate_repair:
            self._write_json_log(f"05_generation/{focus}/final_scenarios.json", final_for_focus)
        return final_for_focus

    # ------------------------------------------------------------------
    # Single-focus generation helpers
    # ------------------------------------------------------------------

    def generate_single_focus_scenarios(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
    ) -> list[dict]:
        """Generate *count* raw scenario dicts from the LLM for *focus*.

        Optimization E (token budget): accepted scenarios and few-shot examples
        are truncated to their respective token budgets before being injected
        into the prompt, preventing context-window overflow.
        """
        # ---- Optimization A: cached few-shot examples ----
        few_shots_key = f"fewshot:{focus}"
        few_shots = _FEWSHOT_CACHE.get(few_shots_key)
        if few_shots is None:
            few_shots = fetch_hf_fewshot(focus=focus, max_examples=_MAX_FEWSHOT_EXAMPLES)
            _FEWSHOT_CACHE.put(few_shots_key, few_shots)

        profile_json = _asset_profile_json(profile)

        # ---- Optimization E: token-budget-aware prompt sections ----
        fewshot_section = _truncated_fewshot_section(few_shots)
        accepted_section = _truncated_accepted_section(accepted_scenarios or [])

        prompt = SCENARIO_GENERATOR_PROMPT.format(
            count=count,
            subagent_name=focus,
            asset_name=profile.asset_name,
            generation_mode=profile.generation_mode,
            asset_profile_json=profile_json,
            tool_definitions=json.dumps(server_desc.get(focus, {}), indent=2),
            few_shot_examples_section=fewshot_section,
            category_options=format_categories_for_prompt(focus),
            specialization_requirements=format_requirements_for_prompt(focus),
            forbidden_patterns=format_forbidden_patterns_for_prompt(focus),
            mode_requirements=format_mode_requirements(profile, focus, profile.generation_mode),
            accepted_scenario_texts=accepted_section,
        )
        self._write_log(
            f"05_generation/{focus}/generation_prompt.txt",
            _redact_logged_prompt(prompt, profile_json),
        )
        response = self._llm_generate(prompt, phase=f"phase_3_generate_{focus}")
        parsed, _ = parse_llm_json(response.text)
        if isinstance(parsed, list):
            self._write_json_log(f"05_generation/{focus}/generation_response.json", parsed)
            return parsed
        self._write_json_log(f"05_generation/{focus}/generation_response.json", [])
        return []

    def validate_and_repair(
        self,
        focus: str,
        scenarios: list[dict],
        profile: AssetProfile,
        accepted_scenarios: list[dict] | None = None,
        failures: list[ScenarioValidationFailure] | None = None,
    ) -> list[dict]:
        if not scenarios:
            return []

        profile_json = _asset_profile_json(profile)
        accepted_section = _truncated_accepted_section(accepted_scenarios or [])

        prompt = VALIDATE_REPAIR_PROMPT.format(
            subagent_name=focus,
            category_options=format_categories_for_prompt(focus),
            specialization_requirements=format_requirements_for_prompt(focus),
            forbidden_patterns=format_forbidden_patterns_for_prompt(focus),
            mode_requirements=format_mode_requirements(profile, focus, profile.generation_mode),
            asset_profile_json=profile_json,
            input_scenarios_json=json.dumps(scenarios, indent=2),
            accepted_scenario_texts=accepted_section,
            validation_failures_json=failure_payload(failures or []),
        )
        self._write_log(
            f"05_generation/{focus}/validate_repair_prompt.txt",
            _redact_logged_prompt(prompt, profile_json),
        )
        response = self._llm_generate(prompt, phase=f"phase_3_repair_{focus}")
        parsed, _ = parse_llm_json(response.text)
        if isinstance(parsed, list):
            repaired = parsed[: len(scenarios)]
            self._write_json_log(f"05_generation/{focus}/validate_repair_response.json", repaired)
            return repaired
        self._write_json_log(f"05_generation/{focus}/validate_repair_response.json", scenarios)
        return scenarios

    def construct_multiagent_scenarios(
        self,
        count: int,
        single_agents: list[dict],
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
    ) -> list[dict]:
        """Construct multi-agent scenarios from the pool of single-agent scenarios.

        Optimization B: single_agents list is chunked to at most 10 items so
        that the prompt does not exceed context limits when many scenarios
        have been generated.
        """
        profile_json = _asset_profile_json(profile)
        # Only pass the most recent 10 single-agent scenarios to avoid blowing
        # the context window (same cap as the base profiling agent)
        single_agents_context = single_agents[:10]

        accepted_section = _truncated_accepted_section(accepted_scenarios or [])

        prompt = MULTIAGENT_COMBINER_PROMPT.format(
            count=count,
            asset_name=profile.asset_name,
            generation_mode=profile.generation_mode,
            asset_profile_json=profile_json,
            mcp_function_definitions=json.dumps(server_desc, indent=2),
            single_agent_scenarios_json=json.dumps(single_agents_context, indent=2),
            accepted_scenario_texts=accepted_section,
            mode_requirements=format_mode_requirements(profile, "multiagent", profile.generation_mode),
            forbidden_patterns=format_forbidden_patterns_for_prompt("multiagent"),
        )
        self._write_log(
            "05_generation/multiagent/generation_prompt.txt",
            _redact_logged_prompt(prompt, profile_json),
        )
        response = self._llm_generate(prompt, max_tokens=_MULTIAGENT_MAX_TOKENS, phase="phase_4_generate_validate_multiagent")
        parsed, _ = parse_llm_json(response.text)
        if isinstance(parsed, list):
            self._write_json_log("05_generation/multiagent/generation_response.json", parsed)
            return parsed
        self._write_json_log("05_generation/multiagent/generation_response.json", [])
        return []

    def _generate_attempt_batch(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict],
        single_agents: list[dict] | None = None,
    ) -> list[dict]:
        if focus == "multiagent":
            if single_agents is None:
                raise RuntimeError("single_agents are required when generating multiagent scenarios")
            return self.construct_multiagent_scenarios(
                count=count,
                single_agents=single_agents,
                profile=profile,
                server_desc=server_desc,
                accepted_scenarios=accepted_scenarios,
            )
        return self.generate_single_focus_scenarios(
            focus=focus,
            count=count,
            profile=profile,
            server_desc=server_desc,
            accepted_scenarios=accepted_scenarios,
        )

    # ------------------------------------------------------------------
    # Cache management helpers (public API for testing/tooling)
    # ------------------------------------------------------------------

    def invalidate_profile_cache(self, asset_name: str, retriever: str | None = None) -> None:
        """Invalidate the cached profile for a specific asset.

        Parameters
        ----------
        asset_name:
            Asset name whose cache entry should be removed.
        retriever:
            If provided, only invalidate for this retriever mode.
            If ``None``, invalidate both ``"arxiv"`` and ``"semantic_scholar"``.
        """
        retrievers = [retriever] if retriever else ["arxiv", "semantic_scholar"]
        for ret in retrievers:
            for open_form in [True, False]:
                key = asset_profile_cache_key(asset_name, ret, open_form)
                self._profile_cache.invalidate(key)
                self._digest_cache.invalidate(research_digest_cache_key(asset_name, ret))
        _log.info("Invalidated cache entries for '%s'", asset_name)

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Return cache statistics for diagnostics."""
        return {
            "profile_cache": self._profile_cache.stats,
            "fewshot_cache": _FEWSHOT_CACHE.stats,
        }
