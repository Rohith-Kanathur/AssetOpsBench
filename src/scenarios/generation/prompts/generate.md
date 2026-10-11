# Generate grounded scenarios

Begin after the profile checker passes. Read `request.json`, the final profile and
the relevant domains in `references/examples.json`. These are adapted requests
from the existing benchmark corpus, with source paths and row IDs for traceability.
They illustrate operator phrasing, task intent and concrete reference answers.
Use them to learn request and reference structure, not as a task distribution
or a fixed solution template. Do not create new scenarios merely by substituting
asset names and dates in an example.
Rebuild their identifiers, windows, tools and reference results from this environment;
original answers and assets are not evidence for a new scenario. Replace example
placeholders with verified results. Apply the selected evaluation
mode below to every example; file-oriented examples are not proof of a file tool.

## Allocate the requested budget

The budget is one of:

- `scenario_count`: total number of scenarios. Allocate them across relevant
  domains using the profile and available capabilities; domains may receive zero.
- `scenario_plan`: integer counts for each domain. Preserve these counts;
  omitted domains mean zero.

Domain keys are `iot`, `fmsr`, `tsfm`, `wo`, `vibration`, and `multiagent`. Save the
chosen per-domain budget in `output/allocation.json`, shaped like `scenario_plan`.
For an explicit plan, this records the supplied plan unchanged. `multiagent`
scenarios combine a coherent workflow across at least two tool domains.

Use the profile's grounded asset records and final tools, reusing suitable data
and capabilities within the selected environment policy. Scenarios are evaluated
with MCP tools and general code execution, including intermediate data transfers.
If the requested count cannot be supported, preserve completed work and explain
the shortfall in the README.

Consider realistic requests whose resolution is inconclusive, has no affirmative
finding, requires rejecting a faulty premise, or cannot be completed because of
missing evidence. Include such cases when appropriate to the asset and environment,
without targeting a quota or labeling scenarios by answerability. Use live
interaction to establish the supported response and record it in characteristic_form.

## Write operator requests and expected behavior

Derive requests from the target asset's documented operating needs, failure
mechanisms, maintenance practices and available evidence. Explore supported
capabilities before choosing tasks, within the requested domain allocation.
Include varied, realistic tasks at appropriate levels of difficulty. Retrieval,
catalog queries, aggregation, recipe construction, analysis and recommendations
can each be valid tasks when they serve an operator need. Avoid trivial paraphrases
and do not append unrelated analysis or actions merely to increase difficulty.

Keep requests concise, allowing a few connected sentences for a complex decision.
State the operator's goal, asset, relevant time window and requested result.
Do not name tool/API methods, prescribe a call sequence or describe the author's
solution. Let the evaluated agent determine how to solve the task. Include
observations an operator could reasonably provide, but do not reveal conclusions
or intermediate results the evaluated agent is expected to derive.
Keep tool and API names in `characteristic_form`. Put necessary supporting evidence
checks there, scoped to the task's assets, time windows and operational needs.
Review shared rubric text per case; remove unrelated lookups, calculations and
deliverables inherited from other tasks.

Keep each task meaningful for its stated purpose. Distinguish catalog discovery,
recipe construction, diagnosis and prediction; do not automatically turn one into
another. Statistical anomalies do not establish physical faults; forecast intervals
require an actual supported method. Reuse the final
profile's cited relationships, applicable standards and verified capabilities.
Apply the data access and tool reuse policy in `profile.md` to later extensions;
update and recheck the profile before using them.

Save `output/scenarios.json` as a list of objects containing:

- `id`: unique integer.
- `type`: a lowercase domain key.
- `text`: concise operator-facing request, without implementation tool names.
- `category`: task category.
- `characteristic_form`: one free-form string containing the verified reference
  answer, required evidence, reasoning and task-specific constraints. Each requirement
  must support the requested result, a necessary domain check or a verified dependency
  in the environment. Document the verified or preferred workflow separately as a
  reference implementation, including exact available `server.tool` references
  discovered from the final capabilities. Its incidental steps are not additional
  completion requirements. Require particular methods, tools or ordering only where
  the task depends on them, not merely because the author used them.
  Keep structured rubric objects out of this field.
- `source_ids`: evidence IDs from `sources.json`.
- `grounding`: `scope` (`asset` or `class`) and applicable `asset_class`, `site`,
  `asset_id`, `sensors`, `start`, `end`, `workorder_ids`, `output_workorder_ids`.
  Asset scope identifies the instance being investigated and the channels and
  interval to inspect. The live reads may establish that requested evidence is absent. Class scope identifies the requested asset class for catalog questions;
  outside FMSR, include `justification` explaining why no instance is required.
  Asset scope also requires `data_source_ids` covering all operational input
  records/files used by the case. These must satisfy the data-grounding rules in
  `profile.md`, including for synthetic extensions and all available evidence examined.
  For an absent asset or stream, cite the registry or dataset establishing the gap.
  `workorder_ids` contains existing prerequisite orders only; record newly created
  orders separately in `output_workorder_ids`.
- `execution`: `requires` (a list containing `mcp`, optionally `general-execution`),
  `input_files` (workspace-relative files supplied before evaluation), and
  `output_files` (a list of `{path, created_by}` objects). Empty file lists are
  valid. Output paths are workspace-relative; `created_by` is the exact MCP tool
  or `general-execution`. An MCP producer must create and return the output path.
  These describe the task contract, not a record of execution. Include requested
  deliverables and intermediate files required by the task or available tools.
  Separate these requirements from helper files used only by the author's solution.
  Inputs must be task data, not verification exports, solutions, scripts, grading
  material or tool receipts. Required input paths must appear in the operator
  request or be discoverable from the actual tools.
Put any limitation and its supporting observations in `characteristic_form`,
not in a separate JSON field. Explain precisely what can and cannot be concluded.
A missing shell capability, credentials, runtime crash or broken tool is a setup
problem to repair, not evidence of a domain limitation. Healthy conditions and no
matching work orders can be valid findings. Never invent an affirmative answer or
repair the environment just to make a requested conclusion true.

## Ground the reference answer

After exercising each scenario, write the actual expected results into
`characteristic_form`. For deterministic tasks, include computed values, matching
identifiers, dates, rankings and the resulting decision. State units (or unknown
source units), precision/tolerances, interval boundaries and tie rules as applicable.
Distinguish reference calculation precision from required reporting precision.
Choose tolerances justified by the inputs, method and requested decision.
Input thresholds and instructions to calculate a result are not reference results.

For files, specify the required contents, row coverage, time coverage and key
values. Require exact column names, JSON keys or layout only when the task or a
verified consumer in the environment depends on them. The author's serialization
or helper script does not establish that dependency. Retain the exercised output
for review and preserve required source values and timestamps. For writes, describe the verified
postconditions while allowing newly assigned IDs to vary. For variable forecasts
or recommendations, state supported outcomes and acceptance criteria, with exact
values only where the task requirements and inputs determine them. Qualitative
tasks need a concrete supported conclusion, not invented numerical precision.
Where evidence is insufficient, include the verified supported results and the
precise conclusion that cannot be established.

Use retained data and exercised outputs as evidence; cross-check key calculations
against their source observations. Research informs the profile and environment
preparation. Scenario questions and reference answers must concern the assets,
tools and data available in the prepared evaluation environment. Keep study-specific
claims, literature discussion and requests to name or summarize papers out of the
question and characteristic form. Retain research citations in the profile and
source records; express applicable domain principles through the environment's
capabilities, evidence and limitations.
Keep reference answers and exercised solutions in grading material, separate from
the operator request and evaluation inputs.

## Short examples

- A single-asset maintenance review checks the relevant asset and orders. A shared
  rubric must not add an unrelated asset lookup.
- A forecast question states the asset, horizon and intended use. Its reference
  records verified results and necessary evidence without exposing a tool sequence
  in the question or requiring an arbitrary output layout.
- Research informs the asset profile. The scenario concerns available asset data
  and capabilities, not the paper that motivated the task.

## Exercise, review and repair

Exercise each scenario using the selected evaluation capabilities. Check the
complete workflow, including intermediate transfers and unavailable-data responses.
Finalize the reference answer from the exercised results before submitting the
scenario. Review the question and its required results separately from the author's
execution trace. Identify requirements introduced only by implementation choices.
Where useful, exercise another valid approach to check for accidental method or
format restrictions. Investigate disagreements against the source evidence; do not
weaken the reference simply to obtain a successful execution. Do not select or reject
scenarios based on the performance of evaluation models.
The harness records commands and tool activity automatically; the checker
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
duplicate or near-duplicate wording, and whether each reference conclusion or
limitation is supported by the observed evidence. Review the question and
characteristic form together for unrelated lookups, unsupported tool-order or
file-layout restrictions, and claims carried over from authoring research.
Verify that required evidence is accessible in the evaluation environment and
that every required result, check and dependency has a task-specific basis.
Repair mismatches and re-exercise affected steps before finalizing.
Preserve completed work and disclose remaining failures when a gap cannot be fixed.
Finish with verified requested/produced counts and unresolved gaps.
