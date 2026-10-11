## Evaluation mode: general execution

The evaluated agent has MCP tools plus shell/Python and file read/write access in a working directory. It can transform returned observations,
calculate additional statistics, join data, and create JSON/CSV files or plots.
Design operational requests that use these capabilities where useful. File output
is optional; include it when it serves the operator's task.

Exercise the complete workflow. Classify a helper that serializes IoT observations
as general execution; an IoT tool returning observations is not a file export.
The evaluated agent receives declared task data, normal tool descriptions and the
operator request, not reference answers, completed solutions or authoring artifacts.
Ground any calculations in retrieved observations and document their limitations.
