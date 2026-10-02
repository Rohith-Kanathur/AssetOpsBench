# Prepare the requested asset environment

Read `request.json` for the asset class and requested scope. Inspect the existing
MCP tools, schemas and data first. Reuse suitable assets, sensor streams, failure
modes and diagnostics. Add data only for a demonstrated gap. Extend a server only
when its existing tools cannot express the needed operation.

Server source is under `src/servers`; collection schemas and loaders are under
`src/couchdb`. The database connection is configured and the normal repository
fixtures are loaded. Preserve existing records when adding to this environment.

1. Inspect the available evidence. For gaps, search primary literature and
   datasets using web search, `kaggle`, `kagglehub`, `huggingface_hub` or `ucimlrepo`.
   Check licenses, columns, units and labels before downloading bounded samples.
   Store raw files in `data/raw` and transformations in `scripts`.
2. Research missing failure modes and sensor relationships. Save
   `output/profile.json` with `asset_class`, `description`, `failure_modes`,
   `sensor_mapping`, `available_capabilities` and `gaps`. Cite each relationship;
   distinguish relevant measurements from validated diagnostic methods.
3. Prepare missing data with the existing loader and a repeatable `scripts/seed.py`
   when needed. Preserve units and original time axes. Identify synthetic IDs,
   records and work orders. Independent samples do not establish a time series;
   failure labels belong in reference evidence, not ordinary sensor channels.
4. Add missing diagnostics to the appropriate MCP server. Cite calculation rules
   and test known cases and invalid inputs. Learned predictors require documented
   labels, held-out validation and reproducible training. Preserve generic behavior.
5. Refresh MCP sessions after edits and exercise the resulting environment.
   Use `mcphub.ToolUniverse` with `load_tools()` and `run(...)` for fresh connections.
   Save the calls and results. Document setup and any additions in
   `output/environment.md`, including commands to reproduce the environment.

Save `output/sources.json` as records with `id`, `url`, `version`, `license`,
`kind` (`observed`, `simulated`, `derived`, or `synthetic`), `files` (workspace-relative
paths and SHA-256 hashes), and `transform` where applicable. For existing fixtures,
use their documented provenance and a `repository:<revision>:<path>` reference
when no public URL is available. Distinguish original datasets, mirrors and
literature. Preserve unsupported claims as gaps.

Use configured clients without reading or printing credential files. If access
or consent is unavailable, record the gap and use accessible evidence. Keep work
within the requested scope, use CPU-scale methods, and capture dependency versions.
Once the profile, environment and tool checks are ready, follow `generate.md`.
