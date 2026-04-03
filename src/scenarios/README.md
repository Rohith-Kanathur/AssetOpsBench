# Scenario Generator

Automated LLM-driven pipeline that generates evaluation scenarios for AssetOpsBench. Given an asset name (e.g. `"Smart Grid Transformer"`), it produces a set of richly-typed benchmark scenarios covering IoT queries, failure-mode reasoning, time-series analysis, work-order decisions, and complex multi-agent workflows.

---

## Directory Structure

```
src/scenarios/
├── generator.py      # Main orchestrator — ScenarioGeneratorAgent + CLI entry point
├── models.py         # Pydantic models: AssetProfile, ScenarioBudget, Scenario
├── prompts.py        # All LLM prompt templates (6 prompts)
├── utils.py          # ArXiv fetch + HuggingFace few-shot helpers
├── local/            # Place custom local study PDFs / text files here
└── huggingface/      # HuggingFace dataset integration notes
```

---

## Prerequisites

Install project dependencies (from repo root):

```bash
uv sync
```

Set required environment variables in a `.env` file at the repo root:

```bash
# .env
LITELLM_MODEL=gpt-4o          # or any LiteLLM-compatible model ID
OPENAI_API_KEY=sk-...          # or whichever provider key your model needs
```

> The generator reuses the same `LiteLLMBackend` and `Executor` as `plan-execute`, so any model already working with that CLI will work here.

---

## Running the Generator

The generator is invoked as a Python module from the **repo root**:

```bash
python -m scenarios.generator "<Asset Name>" [options]
```

### Minimum viable run

```bash
python -m scenarios.generator "Smart Grid Transformer"
```

Generates 50 scenarios and writes them to `generated_scenarios.json`.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `asset_name` *(positional)* | — | Name of the physical asset to generate scenarios for (e.g. `"Chiller"`, `"Wind Turbine"`, `"Smart Grid Transformer"`) |
| `--num-scenarios N` | `50` | Total number of scenarios to generate |
| `--output PATH` | `generated_scenarios.json` | Output file path for the resulting JSON |
| `--model-id MODEL` | project default | LiteLLM model ID override (e.g. `gpt-4o`, `claude-3-5-sonnet`) |
| `--show-workflow` | off | Print granular phase-by-phase progress to the terminal |
| `--log` | off | Dump raw prompts + LLM responses to a timestamped `logs/` directory |

### Examples

```bash
# Generate 20 scenarios with verbose workflow output
python -m scenarios.generator "Chiller" --num-scenarios 20 --show-workflow

# Use a specific model and save to a custom path
python -m scenarios.generator "Wind Turbine" --model-id claude-3-5-sonnet-latest --output wind_turbine_scenarios.json

# Full debug run: workflow + raw log files
python -m scenarios.generator "Smart Grid Transformer" --show-workflow --log

# Minimal silent run (for CI/scripting)
python -m scenarios.generator "Pump" --num-scenarios 10 --output pump_eval.json
```

---

## Generation Pipeline

The generator runs 5 sequential phases:

```
Phase 1 → Asset Profile Construction
Phase 2 → Scenario Budget Allocation
Phase 3 → Individual Agent Generation & Validation  (iot / fmsr / tsfm / wo)
Phase 4 → Multi-Agent Scenario Construction
         → Output JSON
```

### Phase 1 — Asset Profile Construction

1. An LLM generates 3 targeted ArXiv search queries using the canonical academic name for the asset.
2. The ArXiv API is queried; PDFs are fetched and the first 5 pages of each are extracted.
3. A `PROFILE_BUILDER_PROMPT` synthesises an `AssetProfile` from the literature and the available MCP tool descriptions.

**Output model (`AssetProfile`):**
```json
{
  "asset_name": "Smart Grid Transformer",
  "description": "...",
  "sensor_mappings": { "oil_temp": "Top-oil temperature sensor" },
  "known_failure_modes": ["insulation breakdown", "partial discharge"],
  "relevant_tools": { "iot": [{"name": "get_sensor_reading", "reason": "..."}] },
  "iso_standards": ["ISO 14224", "IEC 60076"]
}
```

> **Critical:** If this phase fails to parse an `AssetProfile`, the process exits immediately with a fatal error. There is no fallback.

### Phase 2 — Scenario Budget Allocation

An LLM analyses the `AssetProfile` and distributes the `--num-scenarios` budget across 5 subagent categories:

| Category | Focus |
|---|---|
| `iot` | Sensor data queries and telemetry |
| `fmsr` | Failure modes and structural reliability |
| `tsfm` | Time-series analysis and forecasting |
| `wo` | Work-order decision support |
| `multiagent` | Complex multi-step orchestration (capped at 50% of total) |

> **Critical:** If budget allocation fails, the process exits immediately.

### Phase 3 — Individual Agent Generation & Validation

For each subagent with a non-zero budget:
1. **Few-shot examples** are fetched from `ibm-research/AssetOpsBench` on HuggingFace (filtered by `type`).
2. A `SCENARIO_GENERATOR_PROMPT` produces a JSON array of scenario dicts.
3. A `VALIDATE_REPAIR_PROMPT` validates and repairs each scenario for schema correctness and tool alignment.
4. Changed scenarios are diffed and written to log files (if `--log` is enabled).

**Valid `category` values per subagent:** (extracted from HF examples)

| Subagent | Allowed Categories |
|---|---|
| `iot` | Data Query, Knowledge Query |
| `fmsr` | Knowledge Query |
| `tsfm` | Knowledge Query, Anomaly Detection Query, Tuning Query, Inference Query, Complex Query |
| `wo` | Decision Support, Prediction, Knowledge Query |

### Phase 4 — Multi-Agent Scenario Construction

Uses up to 10 previously-generated single-agent scenarios as seed material to construct complex cross-agent workflows (e.g. detect anomaly via IoT → confirm history via FMSR → schedule via WO).

---

## Output Schema

The output JSON is an array of `Scenario` objects:

```json
[
  {
    "id": "smart_grid_transformer_iot_01",
    "type": "iot",
    "text": "What is the current oil temperature reading for transformer T-42?",
    "category": "Data Query",
    "characteristic_form": "The agent should call get_sensor_reading with asset_id='T-42' and sensor='oil_temp', then return the value with its unit."
  }
]
```

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Auto-assigned: `{asset}_{type}_{n:02d}` |
| `type` | `str` | Subagent category: `iot`, `fmsr`, `tsfm`, `wo`, `multiagent` |
| `text` | `str` | The natural-language query presented to the agent |
| `category` | `str` | Scenario category (see table above) |
| `characteristic_form` | `str` | Natural-language description of the expected agent response/tool flow |

---

## Log Files (`--log`)

When `--log` is passed, a timestamped directory is created:

```
logs/gen_<asset_name>_<YYYYMMDD_HHMMSS>/
├── 01_research_queries_prompt.txt
├── 02_research_queries_response.txt
├── 03_arxiv_results.txt
├── 04_asset_profile_prompt.txt
├── 05_asset_profile_response.txt
├── 06_asset_profile_<asset>_json.txt
├── 07_budget_allocation_prompt.txt
├── 08_budget_allocation_response.txt
├── 09_iot_generation_prompt.txt
├── 10_iot_generation_response.txt
├── 11_validate_repair_prompt.txt
├── 12_validate_repair_response.txt
├── 13_iot_validation_changes.txt    ← diffs for changed scenarios
...
```

Logs are numbered sequentially in pipeline order, making it straightforward to trace exactly what the LLM received and returned at each step.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `[FATAL ERROR] Critical failure: Could not construct AssetProfile` | LLM returned unparseable JSON in Phase 1 | Run with `--log` and inspect `asset_profile_response.txt`; try a more capable model |
| `[FATAL ERROR] Critical failure: Could not dynamically allocate scenario budget` | Phase 2 parse failure | Same as above — check `budget_allocation_response.txt` |
| `[WARNING] No scenarios were successfully generated` | All subagent generation rounds returned empty | Check HuggingFace connectivity; run with `--show-workflow --log` |
| ArXiv fetch slow / hanging | ArXiv rate-limiting (3s between requests enforced) | Normal — each query + PDF fetch takes ~6–10s per paper |
| `datasets` ImportError | HuggingFace `datasets` library missing | Run `uv sync` from repo root |
| `pypdf` ImportError | PDF extraction library missing | Run `uv sync` from repo root |

---

## Data Sources

- **ArXiv** — Academic literature fetched live at runtime via `fetch_arxiv_studies()` in `utils.py`. Queries are LLM-generated, respecting ArXiv's 3-second rate limit.
- **HuggingFace** — Few-shot examples loaded from `ibm-research/AssetOpsBench` via `fetch_hf_fewshot()`. If the dataset is unavailable, a mock fallback is used automatically.