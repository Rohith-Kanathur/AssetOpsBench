## Evaluation mode: MCP only

The evaluated agent has MCP tools and ordinary language
reasoning. It has no shell, Python, arbitrary file editing or direct database
access. You, the generator, still have those preparation capabilities.

Every positive task must be fully achievable through the supplied MCP surface.
Trace the complete workflow, including data transfers between servers. Prefer
answers, summaries and server actions when a file is not an operational need.
A server-native file output is valid only when an exercised MCP call creates and
returns it; returning JSON data alone does not create a JSON file. Use prepared
input files when file consumption is part of the supported operational workflow,
and exercise the consuming server with their supplied paths.

Python may invoke MCP during preparation. It must not perform a task
step, compute an otherwise unavailable result, or write a requested deliverable
and then claim the MCP agent can do so. Keep seed/preparation work separate from
evaluated task steps. Adapt file-oriented human examples to the available surface;
a missing general file writer is not a meaningful negative scenario.
