# Current scenario generation and evaluation

1. Codex CLI researches the asset and datasets using native web search, academic
   literature, UCI, Hugging Face, Kaggle and GitHub. It uses its native login.
2. In an isolated environment copy, it prepares grounded data and any missing
   domain tools, then exercises them on the actual records.
3. It drafts operator requests and characteristic forms, checks them, inspects
   live results and repairs inconsistencies. Automated checks support this
   interaction; they do not by themselves certify scientific validity.
4. Freeze the reviewed scenarios, data and tool environment. Keep generator
   instructions, expected answers and validation scripts outside evaluated inputs.
5. Evaluate API models in the same Stirrup harness with MCP tools and Docker
   code execution. Keep prompt, tools, data access and run limits comparable.
6. Grade full saved evidence with Fable 5.1 in a separate read-only Claude Code
   session. Record usage, cache reads, provider-reported cost and failures.

The cohorts are 25 validated human-authored Chiller scenarios, 25 generated
Chiller scenarios, and a separate generated Transformer cohort. Human validity
annotation is not part of the current submission scope. Historical runs with
other harnesses or missing-data cohorts are not the new comparison results.

Use [generation](../src/scenarios/generation/README.md) and
[evaluation](../src/benchmark/generated/README.md) for commands. Existing versus
extended environments remains a meaningful preparation choice. MCP-only is a
legacy compatibility option, not a separate study focus.
