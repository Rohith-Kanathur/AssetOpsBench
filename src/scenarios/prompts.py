PROFILE_BUILDER_PROMPT = """\
You are an expert Reliability Engineer preparing an Asset Profile.

Given the Asset Name, some recent literature/studies from ArXiv, the available subagents and their tools, and generic knowledge of ISO 55000 / ISO 14224 terms, create an Asset Profile in JSON format.

Asset Name: {asset_name}

Recent ArXiv Literature:
{arxiv_literature}

Available Subagent Tools:
{tool_descriptions}

Task:
Produce a JSON matching the following schema exactly.
CRITICAL: Output ONLY the raw JSON object. Do NOT include markdown code blocks, Python code, or any conversational preamble. 

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

RESEARCH_QUERY_GENERATOR_PROMPT = """\
You are a Technical Researcher for AssetOps.

Given an Asset Name, generate exactly 3 high-quality academic search queries for ArXiv.

IMPORTANT:
- First, convert the asset into its **canonical academic term** (e.g., "smart grid transformer" → "power transformer").
- Use terminology commonly found in engineering research papers, NOT operational phrasing.

Focus Areas:
1. Condition monitoring and diagnostics (use terms like "condition monitoring", "diagnostics", "measurement").
2. Faults and failures (use terms like "fault diagnosis", "failure analysis", "failure mechanisms").
3. Reliability and prognostics (use terms like "reliability", "prognostics", "degradation modeling").

Constraints:
- Keep queries SHORT (3–6 keywords max).
- Do NOT include too many AND/OR operators.
- Avoid generic words like "sensor", "maintenance" unless paired with academic terms.
- Prefer specific technical phrases (e.g., "dissolved gas analysis", "partial discharge").

Asset Name: {asset_name}

Output ONLY a JSON array of 3 strings. 
CRITICAL: Do NOT include markdown code blocks, Python code, or any conversational preamble.
"""

BUDGET_ALLOCATOR_PROMPT = """\
You are an expert Scenario Strategy Consultant for AssetOps Bench.

Given an Asset Profile (derived from technical literature) and a total number of scenarios to generate ({{total_scenarios}}), your task is to allocate this budget across the following five categories:
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
