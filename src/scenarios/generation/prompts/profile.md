# Prepare the requested asset environment

Read `request.json` and inspect the live tools, schemas and data before choosing
what to build. The evaluated agent always has MCP tools, shell/Python and file
access. Evaluation uses the final MCP environment recorded in the profile.
`environment_policy` controls whether preparation may extend its data and tools.
Server source is in `src/servers`; loaders and collection schemas are in
`src/couchdb`. The harness has initialized the starting database. If a prepared
seed was supplied, its original records and hashes are in `data/seed-database/`
and `data/seed-manifest.json`; other public inputs keep their original paths.
Otherwise, the selected checkout's default records are loaded. Use research to
establish the asset's data and capability needs while respecting this policy.

## Inspect coverage and research the asset

Discover tool definitions and exercise relevant reads. Record actual asset class,
site, asset identifiers, installed versus measured channels, units, sampling and
time coverage. Resolve any identifier aliases across servers through evidence.
Catalog knowledge describes an asset class; it does not diagnose an instance.

Always use academic research to inform the asset profile and scenario design,
including when existing data and tools cover the asset. Cover these facets:

- `diagnostics`: monitoring methods, degradation indicators, validation and limits.
- `maintenance`: inspection practices, scheduling and work-order context.
- `sensors`: modalities, placement, units, sampling and signal interpretation.
- `failure_modes`: physical mechanisms, fault signatures and operational risks.
- `standards`: applicable standards or industry conventions, edition and scope.
- `operational_tasks`: realistic operator and manager questions and decisions.

For every generation, use `research.search_papers(query, limit=5)` for targeted
academic searches about the asset class. Read relevant papers and incorporate
their findings before completing the profile. The tool saves queries and results
automatically. Use manuals and official standards as complementary evidence.
If academic search or paper access fails, try other academic sources through web
search and document unresolved access gaps or a lack of relevant results.
Mark abstract-only evidence as such. Cite the particular claim supported by a
source; a title or search result alone is not
support for a threshold, diagnostic rule or validated predictor. For a sparse
facet, record the missing support rather than filling it with uncited certainty.
For standards, verify applicability to the equipment and measurement before using
a threshold; an inaccessible specification remains a documented limitation.

{{environment_guidance}}

Save `output/sources.json` records with `id`, `url`, `version`, `license`, `kind`
(`observed`, `simulated`, `derived`, `synthetic`, or policy-authorized `fixture`), `files` (workspace-relative
paths and SHA-256 hashes), and `transform` where applicable. Link downloaded text,
search receipts, datasets and live tool responses. Use documented fixture provenance
and `repository:<revision>:<path>` when no public URL exists. Identify each source
as literature, original data, a mirror or a derived artifact in its description.
Use configured clients without reading or printing credential files; record access
gaps and proceed with accessible evidence.

Mark operational data sources with `role: "data"`. Apply the selected environment
policy to data roots. Code, literature and tool receipts are not data roots.
For derived, simulated or synthetic data, require:

- `input_source_ids`: retained data sources, each tracing to a data root allowed
  by the selected environment policy.
- `transform`: `script`, `description`, `field_mapping` (output field to input
  column and transformation), `assumptions` and `limitations`. Values are nonempty
  strings except `field_mapping`, which is an object of string pairs.
- Include the transformation script among the source's checksummed `files`.

Every operational dataset and file used by a scenario must appear in this manifest.

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
