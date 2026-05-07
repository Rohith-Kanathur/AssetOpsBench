# HPML Final Project: [Project Title]

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** [Team Name]
- **Members:**
  - Full Name 1 (UNI) — *role / area of contribution*
  - Full Name 2 (UNI) — *role / area of contribution*
  - Full Name 3 (UNI) — *role / area of contribution*
  - Full Name 4 (UNI) — *role / area of contribution*

## Submission

- **GitHub repository:** [https://github.com/Rohith-Kanathur/AssetOpsBench](https://github.com/Rohith-Kanathur/AssetOpsBench)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx)
- **Experiment-tracking dashboard:** [link to public Wandb / MLflow / TensorBoard / Comet / Neptune dashboard]

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

LLM-based AI agents redefine Industry 4.0 asset operations; integrating perception, reasoning, and action across complex industrial systems. Yet evaluating these agents at scale requires large, high-quality benchmark scenarios that are expensive to produce manually. AssetOpsBench, the first unified framework for industrial asset agent evaluation, initially comes with only 141 scenarios. These are handcrafted by SMEs covering a narrow set of HVAC assets (chillers and AHUs), leaving critical classes such as high-voltage power transformers excluded. Beyond asset-class coverage, scenario creation itself does not scale: every new asset type demands physically plausible, causally consistent, tool-reachable, and standards-compliant scenarios authored by subject-matter experts. This project addresses both gaps by extending AssetOpsBench with a new Smart Grid Transformer asset class and introducing a `ScenarioGeneratorAgent` that automatically generates, repairs, and validates scenarios through asset profiling, budget allocation, and constrained LLM generation stages. The optimizations target **inference** stage by reducing latency and token generation. The primary performance bottleneck with the unoptimized baseline is sequential and redundant LLM calls, time consuming domain literature retreival and blocking I/O that together dominate end-to-end wall time. The optimizations that were applied to address this include: a two-level cache (in-memory LRU + disk JSON), a thread pool offloading blocking I/O operations to run concurrently rather than sequentially, and parallelized scenario generation across focus groups. This reduces end-to-end pipeline time by up to **8×** for 50 scenarios with no measurable degradation in scenario quality (mean quality score: 74.2 ± 1.9 for optimized vs. 73.8 ± 3.0 for the baseline).

---

## 2. Model/Application Description

- **Model:** `meta-llama/llama-3-3-70b-instruct` served via IBM WatsonX, used as the LLM backbone across all five pipeline stages: asset profiling, budget allocation, per-domain scenario generation, validation, and repair. A lightweight supervised regression model (trained on the Mendeley transformer health dataset) is used for the `predict_health_index` tool in the Smart Grid Transformer asset class.
- **Framework:** [LiteLLM]for unified LLM API access across providers; [Model Context Protocol (MCP)] for structured tool interfaces exposing the five industrial agent servers (IoT, FMSR, TSFM, WO, Vibration). PyTorch Profiler is used for per-phase instrumentation. Weights & Biases for experiment tracking.
- **Dataset:** Three data sources: (1) existing AssetOpsBench chiller/AHU scenarios as few-shot style examples, (2) academic literature retrieved at runtime from ArXiv and SemanticScholar for asset profile grounding, and (3) the [Mendeley Transformer Health Dataset](https://data.mendeley.com/datasets/rz75w3fkxy/1) for training the health index predictor.
- **Custom components:** `ScenarioGeneratorAgent` pipeline with a validate-and-repair loop enforcing; four new MCP tools for Smart Grid Transformer diagnostics grounded in IEC 60599 and IEC 60076-7 standards; two-level (L1 in-memory LRU + L2 disk) asset profile cache; AsyncBatchSemaphore for rate-limited parallel focus-group execution, thread pool offloader to offload blocking LLM calls.
- **Hardware:** NVIDIA A100 and H100 GPUs (IBM WatsonX inference cluster)

---

## 3. Final Results Summary

All experiments use the Smart Grid Transformer asset in CouchDB-grounded mode. Times are wall-clock seconds measured via `timed_section` spans and validated against PyTorch Profiler Chrome traces.

### Scalability: Pipeline Time vs. Number of Scenarios

**Baseline (unoptimized)**

| Phase | N=10 | N=25 | N=50 |
|---|---|---|---|
| Get Server Descriptions | 2.40 s | 2.58 s | 2.20 s |
| Build Asset Profile | 448.28 s | 338.90 s | 325.70 s |
| Allocate Scenario Budget | 2.05 s | 3.37 s | 2.34 s |
| Generate & Validate Single-Agent | 11.40 s | 34.79 s | 38.83 s |
| Generate & Validate Multi-Agent | 4.05 s | 20.67 s | 39.32 s |
| **Full Pipeline** | **468.22 s** | **400.35 s** | **408.43 s** |

**Optimized (warm cache)**

| Phase | N=10 | N=25 | N=50 |
|---|---|---|---|
| Get Server Descriptions | 1.56 s | 2.42 s | 2.16 s |
| Build Asset Profile | **0.00 s** | **0.00 s** | **0.00 s** |
| Allocate Scenario Budget | **0.00 s** | **0.00 s** | **0.00 s** |
| Generate & Validate Single-Agent | **8.23 s** | **14.86 s** | **15.95 s** |
| Generate & Validate Multi-Agent | 10.73 s | 17.28 s | 18.11 s |
| **Full Pipeline** | **13.63 s** | **19.88 s** | **50.86 s** |

Speedup: **34×** at N=10, **20×** at N=25, **8×** at N=50. The two-level cache eliminates Phase 1 (Build Asset Profile) entirely on warm runs, phase 1 is the dominant cost in the baseline (80–96% of total time). The decreasing speedup ratio at larger N reflects the time taken to generate scenarios in phase 3.

### Scenario Quality: Baseline vs. Optimized

50 Smart Grid Transformer scenarios generated by each pipeline (CouchDB-grounded, N=50), evaluated across 3 independent runs using the three-stage quality scheme (Static /20 + LLM Judge /30 + Dry-Run /50 = /100 composite). A score ≥ 70 is considered high quality.

| Metric | Baseline | Optimized |
|---|---|---|
| Static Score (/20) | 18.4 ± 0.6 | 19.1 ± 0.3 |
| LLM Judge Score (/30) | 22.2 ± 3.6 | 22.4 ± 1.8 |
| Dry-Run Score (/50) | 33.1 ± 1.3 | 32.7 ± 2.5 |
| **Quality Score (/100)** | **73.8 ± 3.0** | **74.2 ± 1.9** |

The 0.4-point gap is well within run-to-run variance, confirming that the speed optimizations come at no cost to scenario quality. Both pipelines exceed the 70-point high-quality threshold on average.

**Hardware:** NVIDIA A100 / H100 GPUs (IBM WatsonX inference cluster)

**Headline result:** *Two-level caching + parallel focus-group execution + thread-pool offloading reduces end-to-end pipeline time from 408 s to 51 s (8× speedup) for 50 scenarios, with mean scenario quality score unchanged at 74.2 ± 1.9 for optimized vs. 73.8 ± 3.0 for the baseline.*

---

## 4. Repository Structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── deliverables/           # Final report (PDF) and final presentation (PPT/PDF) — same files uploaded to CourseWorks
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
└── src/
    ├── agent/              # LLM agent runners, CLI, and plan-execute orchestration
    ├── couchdb/            # CouchDB setup, Docker Compose, and asset data initialisation
    ├── evaluation/         # Evaluation utilities
    ├── llm/                # LiteLLM wrapper and base LLM abstractions
    ├── observability/      # Tracing, run spans, and file exporters
    ├── scenarios/          # Base scenario models, grounding, retrieval, and prompts
    ├── scenarios_evaluation/   # Three-stage scenario quality evaluator (static + LLM judge + dry-run)
    ├── scenarios_optimization/ # Optimized scenario generation 
    ├── scenarios_profiling/    # PyTorch-profiler-instrumented scenario generation pipeline
    ├── scenarios_testing/      # Scenario generation smoke tests
    ├── scenarios_wandb/        # W&B-integrated optimized scenario generation
    └── servers/            # MCP tool servers (IoT, FMSR, TSFM, WO, Vibration, Utilities)
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# Clone
git clone https://github.com/Rohith-Kanathur/AssetOpsBench.git
cd AssetOpsBench
```

Run from the **repo root**:

```bash
uv sync
```

Activate Virtual Environment
```bash
source .venv/bin/activate   # macOS / Linux
```

Copy `.env.public` to `.env`
```bash
cp .env.public .env
# Then edit .env and set WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL
```

Start CouchDB

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
```

Verify CouchDB is running:

```bash
curl -X GET http://localhost:5984/
```

### B. Experiment Tracking Dashboard

Weights & Biases is used for run-level tracking. Logging is done for scenario generation and evaluation stages. Metrics captured: phase wise timing, llm call metrics, cache hit/miss metrics, valid/invalid scenario count and scenario evaluation scores.
> **🔗 Dashboard:** [https://wandb.ai/rk3443-columbia-university/assetopsbench]
> *Platform used:* [Weights & Biases]

### C. Dataset

Dataset used for Health Index Prediction FMSR tool is available here: https://data.mendeley.com/datasets/rz75w3fkxy/1

The scenarios generated from our pipeline are available on Huggingface: TODO

### D. Training

To reproduce the baseline:

```bash
uv run python -m scenarios_profiling.generator Transformer --num-scenarios 50 --data-in-couchdb --profile --profile-dir profiling_output/exp1_latency_baseline
```

To reproduce the optimized run:

```bash
uv run python -m scenarios_optimization.generator Transformer --num-scenarios 50 --data-in-couchdb --profile --profile-dir profiling_output/exp1_latency_optimized
```

To reproduce the run with wandb logging, run:

```bash
uv run python -m scenarios_wandb.generator Transformer \
  --data-in-couchdb \
  --num-scenarios 50 \
  --wandb \
  --wandb-project assetopsbench \
  --wandb-run-name transformer-50-openform
```

### E. Evaluation

```bash
uv run python src/scenarios_evaluation eval_scenarios_wandb.py --wandb --wandb-project assetopsbench --wandb-run-name transformer-eval-50
```

### F. Profiling

To visualize the profiler traces referenced in the report:
1. Run the pipelines as suggested in section `D. Training`.
2. Profiling traces are available in `profiling_output/`
3. Open the chrome trace json files on `https://ui.perfetto.dev/`

### G. Quickstart: Reproduce the Headline Result


To reproduce the headline result in `Section 3: Final Results Summary`:

Run the baseline once:
```bash
uv run python -m scenarios_profiling.generator Transformer --num-scenarios 50 --data-in-couchdb --profile --profile-dir profiling_output/exp1_latency_baseline
```

Run the optimized variant twice (To see the effect of caching):
```bash
uv run python -m scenarios_optimization.generator Transformer --num-scenarios 50 --data-in-couchdb --profile --profile-dir profiling_output/exp1_latency_optimized
```

Open the chrome traces present in `profiling_output/` on https://ui.perfetto.dev/.


---

## 6. Results and Observations

- **Exp 1 - Scalability (Baseline vs. Optimized, N=10/25/50):** The unoptimized baseline spends 80–96% of its total wall time on Phase 1 (Build Asset Profile: 326–448 s), dominated by sequential academic retrieval, PDF fetching, and LLM synthesis. The optimized pipeline with a warm two-level cache reduces this phase to 0 s, yielding end-to-end speedups of **34×** (N=10), **20×** (N=25), and **8×** (N=50). The decreasing ratio at larger N reflects the growing share of scenario-generation phases, which scale with budget but remain well below baseline even at N=50.

- **Exp 2 - Parallelism (C=1→5 parallel focus groups, N=50):** Increasing C from fully serial (C=1, 87.4 s) to C=5 (47.3 s) delivers a **1.85× speedup**. Phase 3 (Generate & Validate Single-Agent) shows the sharpest gains: 43.9 s → 17.6 s. Returns diminish beyond C=3, suggesting that LLM API rate limits and network latency become the bottleneck at higher concurrency, not local CPU or I/O.

- **Exp 3 - Caching (cold vs. warm cache, N=50):** A single cold run primes the disk cache; all subsequent warm runs skip Phase 1 entirely (247.4 s → 0 s), reducing total pipeline time from 302.5 s to 65.1 s, a **4.6× speedup** with no changes to generation logic. The minor increase in Phase 4 (35.3 s → 45.6 s warm) is within normal LLM API latency variance.

- **Exp 4 - Combined optimizations under cold cache (Baseline vs. Optimized, both cold):** Isolating the non-caching gains; thread-pool I/O offloading and parallel focus execution; the optimized variant still achieves a **1.52× end-to-end speedup** (408.4 s → 269.0 s). Phase 1 improves by 33% (325.7 s → 216.9 s) from concurrent PDF/DB/retrieval requests; Phase 3 improves by **2.65×** (38.8 s → 14.7 s) from parallel focus-group generation.

- **Quality (Baseline vs. Optimized, N=50, n=3 runs):** Speed gains come at zero quality cost. Composite quality scores are statistically indistinguishable: **74.2 ± 1.9** (optimized) vs. **73.8 ± 3.0** (baseline). Both exceed the 70-point high-quality threshold.

---

## 7. Notes

- All source code lives under `src/`. See Section 4 for a directory-level description of each package.
- Profiler Chrome traces from reported experiments will be created under `profiling_output/`. Load any `chrome_trace.json` at [https://ui.perfetto.dev](https://ui.perfetto.dev) for a visual flame chart.
- All secrets (API keys, WatsonX credentials, W&B tokens) are loaded from environment variables. Copy `.env.public` to `.env` and fill in `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, and `WATSONX_URL` before running any pipeline command.
- The health index regression model for the Smart Grid Transformer FMSR tool is trained on the [Mendeley Transformer Health Dataset](https://data.mendeley.com/datasets/rz75w3fkxy/1) (open access).
- LLM inference runs on remote IBM WatsonX infrastructure. GPU utilisation metrics captured by W&B therefore reflect only local host activity and do not measure true model-serving compute cost.
- The W&B project for this work is public: [https://wandb.ai/rk3443-columbia-university/assetopsbench](https://wandb.ai/rk3443-columbia-university/assetopsbench).

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [X] Yes, we used AI assistance as described below.

**Tool(s) used:** *e.g., Claude, Cursor*

**Specific purpose:** **

**Sections affected:** **

**How we verified correctness:** **

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{teamname2026hpml,
  title  = {[Project Title]},
  author = {Last1, First1 and Last2, First2 and Last3, First3},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/<org>/<repo>}
}
```

### Contact

Open a GitHub Issue or email *[team-contact@columbia.edu]*.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
