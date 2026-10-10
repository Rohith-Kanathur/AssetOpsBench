# Environment policy: extend

Actively look for real asset records covering all relevant sensors and failure
modes identified in the research. Inspect local records and search accessible
datasets through `kaggle`, `kagglehub`, `huggingface_hub`, `ucimlrepo` or other
suitable clients. Prefer sensor and asset data from academic research and reputable
institutional or industry sources; verify the original provenance of hosted datasets.
Seek sensor time series, operating conditions, failure events
and labels, and applicable vibration, maintenance and work-order records.
Map each relevant sensor and failure mode to its supporting records and
relationships in the profile, making unavailable coverage explicit. Distinguish
observed failure labels from literature-based failure-mode knowledge.

Reuse suitable verified records. Verify licenses, columns, units and labels
before downloading bounded samples. Save raw files in `data/raw` and
transformations in `scripts`. Independent samples do not establish a time series,
and failure labels belong in reference evidence rather than sensor channels.

## Ground every data artifact

Start from verified existing records or acquired real datasets. Every operational
data artifact must trace to retained observed records, including synthetic extensions.
Use only sources relevant to the asset, modality and task; an unrelated dataset or
a literature citation alone does not ground invented measurements.

For extensions, derive ranges, distributions and relationships from those records.
Retain the original sample, field mapping, reproducible transformation, parameters
and random seed. Explain which values are observed versus generated, assumptions
and limits. Preserve units and time semantics. Synthetic registry identities and
work orders may adapt documented source records, with their fictional status explicit.
Ground failure-mode knowledge and maintenance rules in relevant papers or manuals;
keep that knowledge distinct from evidence of measured behavior.

If suitable data cannot be acquired, record the gap and report a scenario shortfall.
Do not substitute an arbitrary fixture to satisfy the quota. Existing data can be
reused when its provenance is verified; downloading new data is not required.

## Prepare and verify the environment

Add data with the existing loader and a repeatable `scripts/seed.py` where needed.
Retain original time axes, units and provenance; identify synthetic assets, records
and work orders. Seed operational sensor, asset and maintenance records into the
appropriate CouchDB collections and verify access through existing MCP reads.
Retain raw files and reproducible transformations for provenance. Prefer asset,
sensor and time-window queries for operational tasks; prepared files are appropriate
when file consumption is part of the supported workflow.

Choose tool changes by the capability they add:

- Exercise existing tools and their supported parameters, recipes and model options
  before deciding a capability is missing. Reuse their computation and data access.
- Add or extend tools for meaningful asset-specific behavior: a supported health
  assessment, physical diagnostic, degradation calculation or trained predictor.
  For example, a transformer `predict_health_index` turns gas, oil and electrical
  measurements into an overall condition assessment; DGA interpretation, winding
  temperature assessment and load assessment similarly encode asset expertise.
  Ground analogous behavior for the requested asset in its own evidence.
- Prefer extending an existing tool's inputs or outputs for a general capability
  gap. Filtering, basic statistics and forecasting methods already supported by
  the environment should use those implementations. A convenience wrapper or a
  particular prepared sample is not an asset-specific capability.
- When a workflow needs a missing transfer between servers, add the smallest
  reusable bridge and reuse the downstream engine. A database-backed series tool
  can accept asset, sensor and time-window selectors and materialize any required
  file internally. Keep the transport change distinct from domain assessment.

For each added or extended tool, record in `output/environment.md` the existing
tools exercised, the demonstrated gap, the new behavior and its operational use,
and the evidence and validation supporting it. Design scenarios around grounded
operator needs and these capabilities. Report unsupported assessments as gaps
rather than adding an unvalidated predictor to meet the scenario quota.

Cite calculation rules and test known cases and invalid inputs. Learned predictors
require documented labels, held-out validation and reproducible training. Preserve
generic server behavior and use CPU-scale methods. Reconnect MCP after edits:
`mcphub.ToolUniverse` with `load_tools()` and `run(...)` provides fresh connections.
Exercise each addition. The harness captures native tool and command activity.
Record preparation, dependencies, tests and reproducible commands in
`output/environment.md`.
