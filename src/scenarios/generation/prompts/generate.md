# Generate grounded scenarios

After the profile checker passes, read `request.json`, the final profile and
`references/examples.json`. The references preserve original questions and
characteristic forms verbatim. Use them for wording, scope and variety, not as a
fixed template or task distribution. Their historical assets, sites, files and
tools need not exist here. Derive new tasks and answers from the current environment;
do not copy historical answers or merely substitute identifiers in an example.
Follow the environment policy in `profile.md` and the evaluation capabilities below.

## Choose tasks and allocate the budget

Start from varied operator or manager needs appropriate to the target asset,
then check how the environment supports them. Let each need determine complexity
and length; do not choose tasks just because a tool makes the answer convenient.
Do not match a held-out cohort's task mix or target a pass rate. Avoid
near-duplicates, unrelated additions and a repeated checklist of audits, exports
or caveats. Keep retrieval, recipe construction, diagnosis and
prediction distinct where they serve different purposes.

- For `scenario_count`, allocate the total across relevant domains; some may receive zero.
- For `scenario_plan`, preserve the supplied per-domain counts; omitted domains mean zero.

Domain keys are `iot`, `fmsr`, `tsfm`, `wo`, `vibration` and `multiagent`.
`multiagent` tasks combine at least two tool domains in a coherent workflow;
verify this when exercising the task, not by inserting tool names into its answer.
Save the allocation in `output/allocation.json`, shaped like `scenario_plan`.
If the environment cannot support the requested count, retain completed work and
report the shortfall without changing the requested budget.

## Write questions and characteristic forms

Write `text` as a concise operator request: the goal, relevant asset or class,
time window where applicable, and requested result. Include observations an
operator could reasonably supply. Do not name tool/API methods, prescribe a call
sequence, or reveal conclusions and intermediate results the agent must derive.
Leave data-source selection and analysis to the agent. Do not add file paths,
internal fields, algorithms or derived thresholds to steer it toward the author's
solution. Identify a supplied document only when it is part of the operator's
request, not as a retrieval hint added after solving the task.

Write `characteristic_form` as a free-form expected response grounded in exercised
results. Include the facts, reasoning, necessary domain checks and constraints
that support the requested result. Keep its scope aligned with the question.
Accept equivalent valid methods and representations. Do not require an unasked
plot, identifier list or extra assessment solely because it appears in the
reference solution.
The author's implementation does not add completion requirements: keep tool-call
sequences, helper steps, runtime labels and generic grading instructions out of
this field. Preserve implementation evidence in the authoring trace and profile.
Use the detail each task needs rather than a common answer template.

- For deterministic results, give the relevant verified values, identifiers,
  dates or rankings, with units where needed. Instructions to calculate a result
  are not a reference answer. Require precision, interval conventions or tie rules
  only when they affect the result; calculation precision does not dictate reporting
  precision. Allow ordinary rounding that preserves the requested conclusion.
- For requested files, describe the necessary contents and coverage. Require exact
  columns, keys or layouts only when the task or a verified consumer depends on them.
  For database writes, give verified postconditions while allowing newly assigned
  IDs to vary.
- For forecasts, recommendations or qualitative assessments, give supported outcomes
  and task-specific acceptance criteria. Require exact values only where the inputs
  and task determine them. Anomalies alone do not establish physical faults;
  prediction claims and intervals need a supported method.
- A healthy result, no matches, a faulty premise or insufficient evidence can be a
  valid outcome. Include these cases when warranted, without quotas or answerability
  labels. State the supported findings and any precise limitation in the same
  characteristic form. Repair runtime, credential and broken-tool problems; do not
  present them as domain limitations or change data to force an affirmative finding.

Research informs asset preparation and interpretation. Questions and answers must
concern the prepared environment's assets, data and capabilities, not papers used
in authoring. Keep research citations in the profile and source records.

## Save the scenario records

Save `output/scenarios.json` as a list of objects with:

- `id`: unique integer.
- `type`: one of the domain keys above.
- `text`, `characteristic_form`: the question and expected response described above.
- `category`: task category.
- `source_ids`: evidence IDs from `sources.json`.
- `grounding`: `scope` (`asset` or `class`) and applicable `asset_class`, `site`,
  `asset_id`, `sensors`, `start`, `end`, `workorder_ids`, `output_workorder_ids`.
  Asset scope identifies the instance and relevant channels/window, with
  `data_source_ids` covering operational inputs under the data-grounding rules in
  `profile.md`. For absent assets or streams, cite the registry or dataset establishing
  the gap. Class scope identifies the class; outside FMSR, add `justification` for
  requiring no instance. Keep prerequisite `workorder_ids` separate from newly
  created `output_workorder_ids`.
- `execution`: `requires` (a list containing `mcp`, optionally `general-execution`),
  `input_files` (workspace-relative paths), and `output_files` (objects with `path`
  and `created_by`). File lists may be empty. `created_by` is the exact MCP producer
  or `general-execution`; an MCP producer must actually create and return the file.
  Declare requested deliverables and intermediate files required by the task or
  tools, not incidental helper files. Inputs must be task data discoverable through
  the shared workspace or tools; do not add paths to questions merely to guide
  retrieval. Never supply reference answers, completed solutions, generator scripts
  or grading material as evaluation inputs.

Keep source records and the profile current. If preparation changes, follow the
selected environment policy and recheck the profile before using those changes.

## Exercise and review

Exercise every scenario through the evaluation capabilities, including required
file transfers and unavailable-data responses. Cross-check key results against
source observations and finalize the characteristic form from that evidence.
Test database writes on a disposable copy, read back created work orders to verify
site and asset, and restore initial task state. Retain exercised outputs for review;
evaluation starts without them. The harness records commands and tool activity,
and the checker captures live MCP reads; do not write duplicate execution logs
or verification receipts.

First read each question without its reference answer and remove hints inherited
from the author's solution. Review the pair for unsupported claims,
unrelated requirements and restrictions inherited from the author's solution.
Where useful, try another valid approach to expose accidental method or format
requirements. Resolve disagreements against source evidence. Do not select scenarios
or weaken references based on evaluation-model performance.

Write `output/README.md` with brief summaries, source links, requested/produced
counts, reused/added capabilities, tests and unresolved gaps. Link the profile,
allocation, scenarios, evidence and `environment.md`. Record installed versions in
`output/requirements.txt`; keep raw downloads and traces outside the report.

Run `python -m scenarios.generation.review --workspace /workspace --stage all`.
Repair reported problems and re-exercise affected steps until applicable checks
pass. The checker validates contracts and live grounding, not end-to-end solvability
or scientific support; review those yourself. Preserve completed work and disclose
any unresolved gaps in the final counts and report.
