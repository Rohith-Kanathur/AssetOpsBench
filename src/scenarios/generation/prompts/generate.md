# Generate grounded scenarios

Read `request.json` and the saved profile. The budget is one of:

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

Save `output/scenarios.json` as a list of objects containing:

- `id`: unique integer.
- `type`: a lowercase domain key.
- `text`: concise operator-facing request, without implementation tool names.
- `category`: task category.
- `characteristic_form`: expected behavior and concrete tool names.
- `positive`: boolean, false for an intentionally unsupported case.
- `source_ids`: evidence IDs from `sources.json`.
- `grounding`: applicable `site`, `asset_id`, `sensors`, `start`, `end` and
  `workorder_ids`. Resolve every positive reference; explain negative cases in
  `missing_evidence`.

Check every scenario through MCP. Save `output/tool_checks.json` as a list of
`{scenario_id, calls}` records. Each call has `tool` in `server.name` form,
`arguments`, and a workspace-relative `response_file` containing its JSON result.
Retain unavailable-data responses. Verify writes on a disposable copy and document
how to restore it. Repeat read calls after reloading newly prepared data.

Write `output/README.md` with short scenario summaries, source links, reused and
added capabilities, requested/produced counts, tests and remaining gaps. Link the
profile, allocation, scenarios, evidence and `environment.md`. Save installed
versions in `output/requirements.txt`. Keep raw downloads and traces outside the
report. Finish with a concise account of ready scenarios and unresolved gaps.
