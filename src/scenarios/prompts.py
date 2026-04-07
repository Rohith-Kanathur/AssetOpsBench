PROFILE_BUILDER_PROMPT = """\
You are an expert Reliability Engineer preparing an Asset Profile.

Given the Asset Name, a structured evidence bundle from ArXiv, the available subagents and their tools, and generic knowledge of ISO 55000 / ISO 14224 terms, create an Asset Profile in JSON format.

Asset Name: {asset_name}

Structured Evidence Bundle:
{evidence_bundle_json}

Available Subagent Tools:
{tool_descriptions}

Task:
Produce a JSON matching the following schema exactly.
CRITICAL: Output ONLY the raw JSON object. Do NOT include markdown code blocks, Python code, or any conversational preamble. 

Rules:
- Treat the evidence bundle as the primary source of truth.
- Only include sensor mappings, failure modes, and standards that are clearly supported by the evidence snippets or are directly aligned with the available tools.
- If evidence is sparse, stay conservative and generic rather than inventing highly specific details.
- Avoid ML-architecture concepts unless they are clearly about monitoring or operating the physical asset itself.
- The asset profile must represent the physical industrial asset class itself, not surrounding digital, grid, networking, cyber, or facility infrastructure.

{{
    "asset_name": "{asset_name}",
    "description": "Short summary of the asset (1-2 sentences).",
    "sensor_mappings": {{"sensor_name": "description of what it measures"}},
    "known_failure_modes": ["failure mode 1", "failure mode 2"],
    "relevant_tools": {{"iot": [{{"name": "tool_name", "reason": "why we might use it for this asset"}}]}},
    "iso_standards": ["list 2-3 relevant ISO standards or engineering conventions"]
}}
"""

SCENARIO_GENERATOR_PROMPT = """\
You are an advanced Scenario Architect for AssetOps Bench.
We need you to generate {count} evaluation scenarios specifically tailored for the '{subagent_name}' subagent, focused on the asset: {asset_name}.

Asset Profile Details:
{asset_profile_json}

Available Subagent Tools (these are ALL the available tools for this subagent, but you MUST prioritize the tools explicitly mentioned in the Asset Profile's "relevant_tools" section):
{tool_definitions}

Here are some Few-Shot examples from our benchmark dataset (for style and tone reference):
{few_shot_examples}

Constraints:
1. Emphasize Reasoning and Tool Use: Scenarios should require the agent to perform domain-specific reasoning based on the available tools.
2. Emphasize Data Handling and Forecasting where applicable.
3. Every scenario must be highly specific, having a clear 'text' (the query to the agent), a 'category' (e.g. Knowledge Query, Data Query, Analytical Query), and a 'characteristic_form' (explaining in natural language what the agent should do to answer the query correctly).

Task: Generate a JSON array of {count} scenarios.
CRITICAL: Output ONLY the raw JSON array. Do NOT include markdown code blocks, Python code, or any conversational preamble.

Format exactly (raw JSON only):
[
    {{
        "text": "What IoT sites are available for the {asset_name}?",
        "category": "{example_category}",
        "characteristic_form": "The expected response should be the return value of all sites, either as text or as a reference to a tool call."
    }}
]
"""

VALIDATE_REPAIR_PROMPT = """\
You are the Validator and Repair agent for the AssetOps Bench scenario generator.

You are given a list of JSON scenarios and the corresponding Asset Profile.
Your job is to strictly validate and repair them based on:
1. Do they query realistic tools/data from the available toolset in the profile?
2. Do they fit exactly the JSON schema (text, category, characteristic_form)?
3. Does the characteristic_form accurately articulate what the ideal response/agent flow should look like?

Asset Profile:
{asset_profile_json}

Input Scenarios:
{input_scenarios_json}

Return ONLY the corrected and repaired JSON array. If a scenario is perfectly fine, keep it as is. 
CRITICAL: Do not include any explanation, markdown formatting, or Python code. Output MUST be raw JSON text only.
"""

MULTIAGENT_COMBINER_PROMPT = """\
You are the Multiagent Scenario Combiner.
You generate complex multi-agent workflows that span across multiple subagents (e.g. IoT + WO + TSFM).
We need {count} multi-agent scenarios for {asset_name}.

Asset Profile:
{asset_profile_json}

Available MCP Function Definitions (these are ALL the tools across all subagents, but you MUST prioritize the tools explicitly mentioned in the Asset Profile's "relevant_tools" section):
{mcp_function_definitions}

Here are some valid single-agent scenarios we generated earlier:
{single_agent_scenarios_json}

Task: Use these single-agent pieces to construct cohesive multi-agent scenarios. A multiagent scenario tests Agent Communication and Coordination, Workflow Orchestration, and Decision-Making.
For instance, detecting an anomaly using IoT, confirming a historical failure with FMSR or TSFM, and creating a work order using WO.

Return ONLY a JSON array of length {count}. 
CRITICAL: Do NOT include markdown code blocks, Python code, or any conversational preamble.

[
    {{
        "text": "Detect a temperature anomaly on the chiller, verify its past occurrences, and schedule a work order.",
        "category": "Knowledge Query",
        "characteristic_form": "The expected response should involve first utilizing the IoT tool to confirm the anomaly, then querying the WO tool to fetch history, before finalizing a work order creation."
    }}
]
"""

RETRIEVAL_QUERY_PLAN_PROMPT = """\
You are a bounded ReAct retrieval agent for industrial asset research.

You do not call ArXiv directly. The host executes searches for you and returns metadata summaries.
At each step, decide whether to search again or finish with the best candidates found so far.

Asset Name: {asset_name}
Current Step: {step_number} of {max_steps}
Current Canonical Asset Name: {canonical_asset_name}

Available Subagent Tools:
{tool_descriptions}

Previous Queries:
{previous_queries}

Current Top Judged Results:
{current_results_summary}

Return ONLY a raw JSON object in this shape:
{{
    "action": "search" or "finish",
    "reason": "brief explanation",
    "canonical_asset_name": "canonical industrial equipment term",
    "queries": ["1 to 2 new ArXiv queries if action=search, otherwise []"],
    "selected_ids": ["preferred arxiv ids if action=finish, otherwise []"]
}}

Constraints:
- Keep the process generic for any industrial asset class.
- An industrial asset class is a physical piece of equipment or subsystem that is monitored, maintained, and can fail.
- Examples of industrial asset classes: chiller, air handling unit, pump, motor, power transformer.
- Context words like smart grid, plant, facility, building, line, or substation describe the deployment environment, not the asset class itself.
- If the asset name is ambiguous, resolve it toward the physical equipment noun phrase, not an ML concept and not a broader system context.
- Normalize queries toward the physical equipment itself. For example, "smart grid transformer" should be treated as a transformer or power transformer in a smart-grid context, not as the smart grid itself.
- Queries should focus on physical asset monitoring, diagnostics, failures, maintenance, condition assessment, degradation, or reliability.
- Prefer equipment-specific queries such as "power transformer condition monitoring" or "power transformer fault diagnosis".
- Avoid broad or ambiguous queries like "transformer reliability" or "smart grid transformer monitoring" when a more equipment-specific query is possible.
- Do not drift into smart-grid cyberattacks, communications, networking, markets, data architecture, control systems, or generic ML "Transformer" papers.
- When action is "search", propose 1 to 2 short queries only.
- When action is "finish", prefer the strongest already judged metadata and do not propose new queries.
- Only finish early if the current top results already contain at least two clearly physical-asset-focused papers.
- Do not use advanced boolean syntax.
- Do not invent arxiv ids that are not present in Current Top Judged Results.
- Output ONLY raw JSON.
"""

RETRIEVAL_METADATA_JUDGE_PROMPT = """\
You are a relevance judge for ArXiv metadata retrieved for industrial asset scenario generation.

Your task is to score how useful each metadata entry is for building an asset profile for the physical industrial asset below.
Judge from the title and summary only.

Asset Name: {asset_name}
Canonical Asset Name: {canonical_asset_name}

Metadata Entries:
{metadata_entries_json}

Return ONLY a raw JSON array. One object per metadata entry:
[
  {{
    "arxiv_id": "entry id",
    "score_1_to_10": 8,
    "reason": "short reason"
  }}
]

Scoring guide:
- Core question: would this paper help define the physical asset class itself and its maintenance, monitoring, sensing, degradation, diagnostics, or failure behavior?
- High score when the paper is clearly about the physical asset and would help populate sensor mappings, known failure modes, relevant tools, maintenance workflows, or standards.
- 9-10: directly about the physical asset and clearly useful for condition monitoring, fault diagnosis, degradation, insulation or thermal behavior, reliability, maintenance, standards, or physical measurements
- 6-8: relevant to the physical asset but somewhat indirect, narrower than ideal, or missing strong maintenance or diagnostics detail
- 3-5: mixed relevance, generic system context, or only weakly connected to the physical asset itself
- 1-2: not about the physical asset itself, clearly about a different asset family, or dominated by unrelated ML/system context
- Low-score examples:
  - smart-grid cyberattacks or false-data-injection papers
  - communications or networking papers
  - grid data architecture, market, or control-system papers
  - generic ML "Transformer" architecture papers
  - papers about other asset families such as rotating machinery unless the target asset is actually that equipment
- High-score examples:
  - condition monitoring
  - fault diagnosis
  - degradation, insulation, or thermal behavior
  - reliability, maintenance, or standards
  - sensors and measurements for the physical equipment
- In every reason, explicitly include either the phrase "physical asset focused" or the phrase "not physical asset focused".
- If the paper is off-target, say why in physical-asset terms.

Be strict about generic smart-grid, networking, cybersecurity, market, or ML-architecture papers when the asset is a physical piece of equipment.
Output ONLY raw JSON.
"""

BUDGET_ALLOCATOR_PROMPT = """\
You are an expert Scenario Strategy Consultant for AssetOps Bench.

Given an Asset Profile (derived from technical literature) and a total number of scenarios to generate ({total_scenarios}), your task is to allocate this budget across the following five categories:
1. iot: Focusing on sensor data and basic telemetry.
2. fmsr: Focusing on failure modes and structural reliability.
3. tsfm: Focusing on time-series analysis and technical maintenance.
4. wo: Focusing on actual maintenance execution and work orders.
5. multiagent: Complex, multi-stage workflows involving orchestration of multiple agents.

Asset Profile:
{asset_profile_json}

Allocation Strategy:
- Prioritize agents that have more "relevant_tools" or "known_failure_modes" in the profile.
- If the asset mentions complex standards (ISO 14224, etc.), lean towards analytical categories (tsfm, fmsr).
- It is perfectly acceptable for a category to have 0 scenarios if the Asset Profile doesn't warrant it.
- Cap "multiagent" at a maximum of 50% of the total budget (e.g., max 25 if total is 50).
- The sum of all allocations MUST exactly equal {total_scenarios}.

Output Format:
CRITICAL: Return ONLY a raw JSON object. Do NOT include markdown code blocks, Python code, or any conversational preamble.

{{
    "reasoning": "A brief explanation of why you chose this distribution based on the asset details.",
    "allocation": {{
        "iot": int,
        "fmsr": int,
        "tsfm": int,
        "wo": int,
        "multiagent": int
    }}
}}
"""
