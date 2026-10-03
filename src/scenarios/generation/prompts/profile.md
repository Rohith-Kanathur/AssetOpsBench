# Prepare the requested asset environment

Read `request.json` and inspect the live tools, schemas and data before choosing
what to build. Server source is in `src/servers`; loaders and collection schemas
are in `src/couchdb`. The normal fixtures are loaded. Reuse relevant records and
capabilities, preserving them when adding data or tools for a demonstrated gap.

## Inspect coverage and research the asset

Discover tool definitions and exercise relevant reads. Record actual asset class,
site, asset identifiers, installed versus measured channels, units, sampling and
time coverage. Resolve any identifier aliases across servers through evidence.
Catalog knowledge describes an asset class; it does not diagnose an instance.

Cover each facet using documented existing evidence or targeted new research.
Reuse adequate research and receipts before making additional searches:

- `diagnostics`: monitoring methods, degradation indicators, validation and limits.
- `maintenance`: inspection practices, scheduling and work-order context.
- `sensors`: modalities, placement, units, sampling and signal interpretation.
- `failure_modes`: physical mechanisms, fault signatures and operational risks.
- `standards`: applicable standards or industry conventions, edition and scope.
- `operational_tasks`: realistic operator and manager questions and decisions.

When academic search is needed, use `research.search_papers(query, limit=5)`; it saves
query/results receipts and returns their metadata with the papers. Preserve these
receipts and inspect relevant source text. Web search can supply official standards,
manuals and other primary evidence. Mark abstract-only evidence as such. Cite the
particular claim supported by a source; a title or search result alone is not
support for a threshold, diagnostic rule or validated predictor. For a sparse
facet, record the missing support rather than filling it with uncited certainty.
For standards, verify applicability to the equipment and measurement before using
a threshold; an inaccessible specification remains a documented limitation.

For missing data, inspect accessible datasets through `kaggle`, `kagglehub`,
`huggingface_hub`, `ucimlrepo` or other suitable clients. Verify licenses, columns,
units and labels before downloading bounded samples. Save raw files in `data/raw`
and transformations in `scripts`. Independent samples do not establish a time
series, and failure labels belong in reference evidence rather than sensor channels.

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
and work orders. Extend an appropriate server when existing tools cannot express
a needed operation. The available capabilities may grow during this work.

Cite calculation rules and test known cases and invalid inputs. Learned predictors
require documented labels, held-out validation and reproducible training. Preserve
generic server behavior and use CPU-scale methods. Reconnect MCP after edits:
`mcphub.ToolUniverse` with `load_tools()` and `run(...)` provides fresh connections.
Exercise each addition, saving real calls and responses. Record preparation,
dependencies, tests and reproducible commands in `output/environment.md`.

Save `output/sources.json` records with `id`, `url`, `version`, `license`, `kind`
(`observed`, `simulated`, `derived`, or `synthetic`), `files` (workspace-relative
paths and SHA-256 hashes), and `transform` where applicable. Link downloaded text,
search receipts, datasets and live tool responses. Use documented fixture provenance
and `repository:<revision>:<path>` when no public URL exists. Identify each source
as literature, original data, a mirror or a derived artifact in its description.
Use configured clients without reading or printing credential files; record access
gaps and proceed with accessible evidence.

Mark operational data sources with `role: "data"`. Only original observed records
use `kind: "observed"`; code, literature and tool receipts are not data roots.
For derived, simulated or synthetic data, require:

- `input_source_ids`: retained data sources, each ultimately tracing to observed data.
- `transform`: `script`, `description`, `field_mapping` (output field to input
  column and transformation), `assumptions` and `limitations`. Values are nonempty
  strings except `field_mapping`, which is an object of string pairs.
- Include the transformation script among the source's checksummed `files`.

Every seeded dataset and file used by a scenario must appear in this manifest.

## Save the final verified profile

Write `output/profile.json` describing the environment after preparation:

- `asset_class`, `description`: nonempty strings matching the requested class.
- `operator_tasks`, `manager_tasks`: nonempty lists of realistic task descriptions.
- `assets`: coverage records with `site`, `asset_id`, `asset_class`, `source_ids`,
  `data_source_ids` (the operational records, distinct from literature or receipts)
  and applicable `iot`/`vibration` objects containing `sensors`, `start`, `end` and
  `total_observations`. Each populated coverage object also names its own
  `data_source_ids`. Record per-server aliases when present. Empty coverage is
  a gap, not evidence that an instance is available.
- `failure_modes`: records with `name`, `description` and `source_ids`.
- `sensor_mapping`: records with `failure_mode`, `sensors`, `source_ids` and the
  relationship's limitations. Separate useful measurements from validated tests.
- `available_capabilities`: domain-keyed lists of discovered tool strings or
  `{tool, description}` objects, reflecting verified additions as well as existing tools.
- `research`: an object with all six facet keys above. Each value has `status`
  (`supported`, `gap` or `not_applicable`), a nonempty `summary` and `source_ids`.
  Supported claims cite evidence; gaps and inapplicability explain the limitation.
- `gaps`: records with `id`, `dependency`, `reason` and applicable `source_ids`.

Link coverage and physical relationships to retained evidence. Keep the profile
current after any later tool or data changes. Before drafting scenarios, run:

`python -m scenarios.generation.review --workspace /workspace --stage profile`

Read its JSON findings, repair the artifacts and rerun until profile checks pass.
The checker checks structure and evidence references; assess scientific support
yourself. Then read `generate.md` and the applicable entries in
`references/examples.json`.
