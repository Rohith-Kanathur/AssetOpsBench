# Generate grounded scenarios

Begin after the profile checker passes. Read `request.json`, the final profile and
the relevant domains in `references/examples.json`. These are adapted requests
from the existing benchmark corpus, with source paths and row IDs for traceability.
They illustrate operator phrasing and task intent. Rebuild their identifiers,
windows, tools and expected behavior from this environment; their original answers
and assets are not evidence for a new scenario.

## Allocate the requested budget

The budget is one of:

- `scenario_counts`: positive/negative totals. Allocate them across relevant
  domains using the profile and available capabilities; domains may receive zero.
- `scenario_plan`: positive/negative counts for each domain. Preserve these quotas;
  omitted domains and counts mean zero.

Domain keys are `iot`, `fmsr`, `tsfm`, `wo`, `vibration`, and `multiagent`. Save the
chosen per-domain budget in `output/allocation.json`, shaped like `scenario_plan`.
For an explicit plan, this records the supplied plan unchanged. `multiagent`
scenarios combine a coherent workflow across at least two tool domains.

Use existing data and tools; additions require a concrete coverage gap. Positive
tasks must be answerable through the environment. Negative tasks deliberately
request something the evidence or tools cannot support, with the missing dependency
recorded. Keep the positive/negative quotas separate. If a requested quota cannot
be supported, preserve completed work and explain the shortfall in the README;
never relabel a requested positive as negative to fill the count.

## Write operator requests and expected behavior

Use a mix of direct lookups, analysis, recommendations and authorized actions.
Lead with the operator's question or decision. Most positive cases should involve
more than one meaningful output or action, with a relevant constraint or conditional
decision where the task calls for it. Keep requests concise, allowing a few connected
sentences for a complex decision. Put explicit user constraints in the request;
put routine pagination, verification and default fallback mechanics in the rubric.
Keep tool and API names in `characteristic_form`, while operator text names the
real assets, sites, measurements and time windows needed to perform the task.

Keep each primary domain substantive. Catalog discovery is different from applying
a diagnostic or predicting failure. Statistical anomalies do not establish physical
faults; forecast intervals require an actual supported method. Reuse the final
profile's cited relationships, applicable standards and verified capabilities.
If you extend data or tools, update and recheck the profile before using them.

Save `output/scenarios.json` as a list of objects containing:

- `id`: unique integer.
- `type`: a lowercase domain key.
- `text`: concise operator-facing request, without implementation tool names.
- `category`: task category.
- `characteristic_form`: one nonempty string describing expected outputs, reasoning,
  evidence, constraints and exact available `server.tool` references. Discover tool
  names from the final capabilities; keep structured rubric objects out of this field.
- `positive`: boolean, false for an intentionally unsupported case.
- `source_ids`: evidence IDs from `sources.json`.
- `grounding`: `scope` (`asset` or `class`) and applicable `asset_class`, `site`,
  `asset_id`, `sensors`, `start`, `end`, `workorder_ids`, `output_workorder_ids`.
  Asset scope identifies a real instance and resolves each referenced channel and
  interval. Class scope identifies the requested asset class for catalog questions;
  outside FMSR, include `justification` explaining why no instance is required.
  `workorder_ids` contains existing prerequisite orders only; record newly created
  orders separately in `output_workorder_ids` after checking their receipts.
- `missing_evidence`: required for a negative case, a list of objects with
  `dependency`, `reason`, `response_files` and optional `gap_ids` from the profile.
  Each missing dependency links at least one retained response for that scenario.

A negative request should be plausible for an operator. Its rubric explicitly
requires an insufficiency answer explaining the missing evidence or capability.
Establish the limitation from the environment; a demanding horizon or unfamiliar
identifier alone is not proof. Preserve all supported parts of the answer and
avoid implying that unsupported certainty is available.

## Exercise, review and repair

Check every scenario through MCP. Save `output/tool_checks.json` as a list of
`{scenario_id, calls}` records. Each call has `tool` in `server.name` form,
`arguments`, and a workspace-relative `response_file` containing its JSON result.
Retain actual unavailable-data responses. Positive multiagent records must span at
least two domain servers and their rubric must describe the connected workflow.

Exercise database writes on a disposable copy. Add check-record-level
`write_context: {environment: "disposable", evidence_file: "<workspace-relative JSON path>"}`.
The evidence file documents the source database, copy and restore procedure.
For each output work order, save a `wo.get_workorder` read-back with the exact site,
order number and matching asset in that copy. Keep receipt locations and isolation
details in the evidence and rubric; they need not become benchmark administration
instructions in the operator's request. The checker reads these receipts and
does not replay writes. Repeat relevant reads after newly prepared data is loaded.

Write `output/README.md` with short scenario summaries, source links, reused and
added capabilities, requested/produced counts, tests and remaining gaps. Link the
profile, allocation, scenarios, evidence and `environment.md`. Save installed
versions in `output/requirements.txt`. Keep raw downloads and traces outside the
report.

Run `python -m scenarios.generation.review --workspace /workspace --stage all`.
Read the JSON findings, repair the affected artifacts and rerun until applicable
checks pass. Also review realism, scientific support, meaningful difficulty,
duplicate or near-duplicate wording, and whether negative cases are truly unsupported.
Preserve completed work and disclose remaining failures when a gap cannot be fixed.
Finish with verified requested/produced counts and unresolved gaps.
