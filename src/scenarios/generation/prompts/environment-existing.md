# Environment policy: existing

Use the existing asset records, catalogs, models and MCP implementations.
The harness initializes the prepared seed, or the checkout's default data when
no seed was supplied, once.
Keep that input surface fixed throughout preparation and scenario generation.

Continue academic research, reading papers and learning the asset's physics,
failure mechanisms, maintenance practices and task needs. Save literature and
research evidence for the profile and reference answers. Use research to interpret
the supplied records and expose limits; it does not supply new operational data.

Inspect and exercise the existing tools, supported recipes and parameters. Server
and loader source is mounted read-only. Do not add or modify tools, register new
models/features, acquire operational datasets, run a seed/import script, or add
sensor readings, registry entries, failure-mode catalog entries or maintenance
history. Missing data or capabilities remain documented gaps.

Existing benchmark fixtures are valid task inputs even when their original real
world provenance is unavailable. Describe that limitation explicitly; fixture
values do not establish real plant behavior, physical units or validated faults.
Use `kind: "fixture"`, `role: "data"`, a `repository:<revision>:<path>` URL and
checksummed original fixture files from the protected environment baseline.
Only verified original observations may use `kind: "observed"`.

Lossless file representations of existing records are permitted for existing
file-based tools. Preserve values, identifiers, units and timestamps; record a
reproducible transformation with `kind: "derived"` and the fixture data roots.
Declare such files as task inputs and exercise their exact paths. Invented values,
new labels/channels, synthetic histories, interpolation and resampling extend
the input surface and are unavailable under this policy.

Exercise normal task actions through existing MCP tools. Test mutations on a
disposable database copy and restore initial inputs afterward; creating a proposed
work order as a task output differs from loading fabricated service history.
Native TSFM run/result ledgers may retain exercised outputs. They cannot become
prerequisite data for another scenario. All initial input collections must match
the protected baseline when generation finishes.

Map realistic operator and manager needs to the available surface before
allocating counts. Honor an explicit domain plan. When a quota cannot be supported,
retain completed cases and explain the shortfall; keep its polarity and domain.
Record research, input limitations, reused capabilities, exercise commands and
tests in `output/environment.md`. No new tool or operational dataset is needed
to complete preparation under this policy.
