# Generate grounded scenarios

Begin after the profile checker passes. Read `request.json`, the final profile and
the relevant domains in `references/examples.json`. These are adapted requests
from the existing benchmark corpus, with source paths and row IDs for traceability.
Read its `negative_examples` too: human requests with insufficient data or missing
history. Their adaptation notes identify limitations in the original rubrics.
They illustrate operator phrasing, task intent and concrete reference answers.
Rebuild their identifiers, windows, tools and reference results from this environment;
original answers and assets are not evidence for a new scenario. Replace example
placeholders with verified results. Apply the selected evaluation
mode below to every example; file-oriented examples are not proof of a file tool.

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

Use the profile's grounded asset records and final tools, reusing suitable data
and capabilities within the selected environment policy. Positive
tasks must be fully answerable through the final environment and evaluation
capabilities in `generation_mode`, including intermediate data transfers. Negative
tasks test a real domain evidence gap, with the missing dependency recorded. Keep the positive/negative quotas separate. If a requested quota cannot
be supported, preserve completed work and explain the shortfall in the README;
never relabel a requested positive as negative to fill the count.

## Write operator requests and expected behavior

Each request must use evidence to assess a condition, justify a decision, produce
a forecast for a stated need, or perform an authorized action. State the concrete
question and requested result. Exclude standalone asset, sensor, model or failure-mode
listings and record retrieval; these are supporting steps, including in the reference
examples. A profile or statistics comparison must answer a concrete operational or
analytical question; a checklist of tool outputs is insufficient. Additional lookups
alone do not make a task substantive.

Use relevant constraints or conditional decisions where the task calls for them.
Keep requests concise, allowing a few connected sentences for a complex decision.
Put explicit user constraints in the request;
put routine pagination, verification and default fallback mechanics in the rubric.
Keep tool and API names in `characteristic_form`, while operator text names the
real assets, sites, measurements and time windows needed to perform the task.

Keep each primary domain substantive. Catalog discovery is different from applying
a diagnostic or predicting failure. Statistical anomalies do not establish physical
faults; forecast intervals require an actual supported method. Reuse the final
profile's cited relationships, applicable standards and verified capabilities.
Apply the data access and tool reuse policy in `profile.md` to later extensions;
update and recheck the profile before using them.

Save `output/scenarios.json` as a list of objects containing:

- `id`: unique integer.
- `type`: a lowercase domain key.
- `text`: concise operator-facing request, without implementation tool names.
- `category`: task category.
- `characteristic_form`: one free-form string containing the verified reference
  answer, followed by required reasoning, evidence, constraints and exact available
  `server.tool` references. Discover tool names from the final capabilities.
  Keep structured rubric objects out of this field.
- `positive`: boolean, false for an intentionally unsupported case.
- `source_ids`: evidence IDs from `sources.json`.
- `grounding`: `scope` (`asset` or `class`) and applicable `asset_class`, `site`,
  `asset_id`, `sensors`, `start`, `end`, `workorder_ids`, `output_workorder_ids`.
  Asset scope identifies a real instance and resolves each referenced channel and
  interval. Class scope identifies the requested asset class for catalog questions;
  outside FMSR, include `justification` explaining why no instance is required.
  Asset scope also requires `data_source_ids` covering all operational input
  records/files used by the case. These must satisfy the data-grounding rules in
  `profile.md`, including for synthetic extensions and supported parts of negatives.
  For an absent asset or stream, cite the registry or dataset establishing the gap.
  `workorder_ids` contains existing prerequisite orders only; record newly created
  orders separately in `output_workorder_ids`.
- `execution`: `requires` (a list containing `mcp`, optionally `general-execution`),
  `input_files` (workspace-relative files supplied before evaluation), and
  `output_files` (a list of `{path, created_by}` objects). Empty file lists are
  valid. Output paths are workspace-relative; `created_by` is the exact MCP tool
  or `general-execution`. An MCP producer must create and return the output path.
  These describe the task contract, not a record of execution. Include all file
  deliverables and intermediate files necessary to finish the task, including
  those required only by the rubric.
  Inputs must be task data, not verification exports, solutions, scripts, grading
  material or tool receipts. Required input paths must appear in the operator
  request or be discoverable from the actual tools.
- `missing_evidence`: required for a negative case, a list of objects with
  `kind`, `dependency`, `reason` and optional `gap_ids` from the profile.
  `kind` is one of `missing_data`, `unknown_asset`, `unknown_site`, `missing_sensor`,
  `missing_history`, `unsupported_diagnosis`, `unsupported_model`,
  `insufficient_coverage`, or `conflicting_evidence`.
  Ground each limitation in the cited environment data and profile.

A negative request pursues the same substantive outcomes but lacks required evidence.
It should be plausible for an operator. Its rubric explicitly
requires an insufficiency answer explaining the missing evidence or capability.
Establish the limitation from the environment; a demanding horizon or unfamiliar
identifier alone is not proof. Preserve all supported parts of the answer and
avoid implying that unsupported certainty is available. Vary substantive gaps:
a valid asset at the wrong site, a missing channel or interval, an unsupported
cross-source comparison, or diagnostic certainty without the required evidence.
Missing shell access, a file writer, credentials, runtime crashes or a broken tool
are not negative task concepts. A healthy asset or a query returning no matching
work orders can be a successfully answered positive task; negative means the
requested conclusion or action cannot be supported.

## Ground the reference answer

After exercising each scenario, write the actual expected results into
`characteristic_form`. For deterministic tasks, include computed values, matching
identifiers, dates, rankings and the resulting decision. State units (or unknown
source units), precision/tolerances, interval boundaries and tie rules as applicable.
Input thresholds and instructions to calculate a result are not reference results.

For files, specify the expected columns, row count, time coverage and key values;
retain the full exercised output for review. For writes, describe the verified
postconditions while allowing newly assigned IDs to vary. For variable forecasts
or recommendations, state supported outcomes and acceptance criteria, with exact
values only where the requested method and inputs determine them. Qualitative
tasks need a concrete supported conclusion, not invented numerical precision.
Negative cases include verified supported results and the precise conclusion
that cannot be established.

Use retained data and exercised outputs as evidence; cross-check key calculations
against their source observations. Keep reference answers and exercised solutions
in grading material, separate from the operator request and evaluation inputs.

## Exercise, review and repair

Exercise each scenario using the selected evaluation capabilities. Check the
complete workflow, including intermediate transfers and unavailable-data responses.
Finalize the reference answer from the exercised results before submitting the
scenario. The harness records commands and tool activity automatically; the checker
captures its own live MCP reads. Keep source provenance and actual deliverables, without
writing a parallel execution log or verification receipts.

Exercise database writes on a disposable copy and read back each created work order
to verify its site and asset. Restore the initial task state afterward. Keep
exercised outputs in the workspace; evaluation starts without them. Passing the
checker validates contracts and live grounding, not end-to-end scenario solvability.

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
