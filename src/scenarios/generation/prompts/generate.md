# Generate grounded scenarios

Read `request.json` and the saved profile. Produce the requested number of
scenarios, covering each requested domain. Use the data and tools already present;
further additions require a concrete coverage gap. Positive tasks must be
answerable through the environment. Where evidence is missing, write an explicit
insufficient-data scenario and record the unavailable dependency.

Save `output/scenarios.json` as a list of objects containing:

- `id`: unique integer.
- `type`: a requested domain.
- `text`: concise operator-facing request, without implementation tool names.
- `category`: task category.
- `characteristic_form`: expected behavior and concrete tool names.
- `positive`: boolean, false for an insufficient-data case.
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
added capabilities, positive/negative counts, tests and remaining gaps. Link the
profile, scenarios, evidence and `environment.md`. Save installed versions in
`output/requirements.txt`. Keep raw downloads and traces outside the report.
Finish with a concise account of ready scenarios and unresolved gaps.
