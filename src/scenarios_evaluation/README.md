# scenarios_evaluation

Evaluates the quality of generated scenarios against a three-tier scoring scheme and optionally logs results to Weights & Biases.

## Evaluation Scheme

Every scenario is graded across three independent evaluation passes that together produce a composite **quality score out of 100**:

| Pass | Max Score | What it checks |
|------|-----------|----------------|
| **Static** | 20 | Schema completeness, required-field presence, valid category, text length bounds, duplicate detection |
| **LLM Judge** | 30 | Five dimensions judged by an LLM: *clarity*, *answerability*, *difficulty*, *tool_usability*, *characteristic_quality* |
| **Dry-Run** | 50 | The scenario is executed end-to-end via `PlanExecuteRunner`; an LLM then scores *plausible_response*, *used_correct_tools*, and *no_obvious_errors* |

The three weighted scores are summed to produce the final quality score for each scenario.

## Usage

### Standard evaluation (no W&B)

```bash
# Evaluate all scenarios in the default scenarios.json
uv run python src/scenarios_evaluation/eval_scenarios.py

# Evaluate only the first 10 scenarios
uv run python src/scenarios_evaluation/eval_scenarios.py --limit 10

# Write results to a custom path
uv run python src/scenarios_evaluation/eval_scenarios.py --output my_report.json
```

### Evaluation with Weights & Biases logging

```bash
uv run python src/scenarios_evaluation/eval_scenarios_wandb.py \
    --wandb \
    --wandb-project assetopsbench \
    --wandb-run-name transformer-eval-50

# Evaluate first 10 only and log to W&B
uv run python src/scenarios_evaluation/eval_scenarios_wandb.py --limit 10 --wandb
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--scenarios PATH` | `scenarios_evaluation/scenarios.json` | Input scenarios file |
| `--output PATH` | `eval_scenarios_report.json` | Output report path |
| `--model-id MODEL` | `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | LLM used for judging |
| `--limit N` | — | Evaluate only the first N scenarios |
| `--wandb` | off | Enable W&B logging |
| `--wandb-project` | `assetopsbench` | W&B project name |
| `--wandb-entity` | — | W&B entity/team |
| `--wandb-run-name` | — | W&B run display name |
| `--wandb-tags` | — | Space-separated list of W&B tags |

## W&B Logged Metrics

When `--wandb` is enabled, the following are logged:

- **Per-scenario** (`eval/*`): static score, LLM score, dry-run score, quality score, type, category
- **Per-LLM-call** (`llm_call/*`): wall time, prompt/completion/total tokens, phase
- **Run summary** (`summary/*`): mean/min/max quality scores, per-dimension means, type & category distributions, total token usage
- **Tables**: `eval_report_table` (one row per scenario), `llm_calls_table` (one row per LLM call)

## Output

The evaluator writes a JSON report (`eval_scenarios_report.json` by default) with one entry per scenario containing all scores, per-dimension breakdowns, LLM suggestions, and dry-run issues.
