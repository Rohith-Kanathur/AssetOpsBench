"""Prompts for single-agent and multi-agent scenario generation."""

SCENARIO_GENERATOR_PROMPT = """\
You are an advanced industrial agent Scenario Architect for AssetOps Bench.
We need you to generate {count} evaluation scenarios with primary focus '{subagent_name}' for the asset class: {asset_name}.

Generation Mode:
{generation_mode}

Asset Profile (full JSON):
{asset_profile_json}

Available Focus Tools:
{tool_definitions}

{few_shot_examples_section}

When generating scenarios, use the Generation Mode and Mode-specific grounding rules (not the few-shot rows) to decide whether each scenario must embed readings, rules, summaries, or dataset identifiers in the query text.

Suggested category values for this focus:
{category_options}

Primary-focus requirements:
{specialization_requirements}

Avoid these bad patterns:
{forbidden_patterns}

Mode-specific grounding rules:
{mode_requirements}

Already accepted scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Constraints:
1. Every scenario must read like a realistic request from an industrial operator or the operator's manager.
2. The scenario may involve supporting work from other agents, but the main burden should stay on the primary focus '{subagent_name}'.
3. Every scenario must be highly specific, having a clear 'text', a 'category', and a 'characteristic_form'.
4. The characteristic_form must explicitly mention the concrete MCP tool names needed to solve the task.
5. The 'text' field must stay natural operator language only: do not name MCP tools, API or function identifiers, or add parenthetical hints such as "e.g. get_failure_modes tool". Reserve every concrete tool name for 'characteristic_form' only.
6. Closed-form scenarios must embed explicit inline sensor readings in the query text: for each measurement, sensor name (or label), numeric value, and unit (e.g. ppm, %, Hz, mm/s). You may also embed rule text, summaries, or dataset identifiers when the task requires them.
7. Open-form scenarios must use only grounded identifiers present in the Asset Profile.
8. Do not output Unsupported.

Task: Generate a JSON array of {count} scenarios.
CRITICAL: Output ONLY the raw JSON array. Do NOT include markdown code blocks, Python code, or any conversational preamble.

Format exactly (raw JSON only):
[
    {{
        "text": "...",
        "category": "...",
        "characteristic_form": "..."
    }}
]
"""

VALIDATE_REPAIR_PROMPT = """\
You are the Validator and Repair agent for the AssetOps Bench scenario generator.

You are given a list of JSON scenarios and the corresponding Asset Profile.
Your job is to strictly validate and repair them based on:
1. Do they fit exactly the JSON schema (text, category, characteristic_form)?
2. Does the characteristic_form name the concrete MCP tool names needed to solve the task?
3. Does the 'text' field contain only natural operator language, with no MCP tool names, function identifiers, or parenthetical hints like "e.g. get_failure_modes tool"? If any appear in 'text', remove or rewrite them and keep tool names in characteristic_form only.
4. Does each scenario clearly satisfy the prompt rules for primary focus '{subagent_name}' even if it uses supporting cross-agent steps?
5. In closed_form mode, does each scenario text include explicit inline sensor readings (name, value, unit) per Mode-specific grounding rules? Are there duplicates, near-duplicates, or (for open_form) ungrounded identifiers?
6. Fix every deterministic validation failure listed below.

Target Focus:
{subagent_name}

Suggested Categories:
{category_options}

Primary-focus requirements:
{specialization_requirements}

Avoid these bad patterns:
{forbidden_patterns}

Mode-specific grounding rules:
{mode_requirements}

Asset Profile:
{asset_profile_json}

Input Scenarios:
{input_scenarios_json}

Already accepted scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Deterministic validation failures that MUST be fixed:
{validation_failures_json}

Return ONLY the corrected and repaired JSON array. If a scenario is perfectly fine, keep it as is. 
CRITICAL: Do not include any explanation, markdown formatting, or Python code. Output MUST be raw JSON text only.
"""

MULTIAGENT_COMBINER_PROMPT = """\
You are the Multiagent Scenario Combiner.
You generate complex multi-agent workflows that span across multiple focused capabilities (e.g. IoT + WO + TSFM + vibration).
We need {count} multi-agent scenarios for {asset_name}.

Generation Mode:
{generation_mode}

Asset Profile:
{asset_profile_json}

Available MCP Function Definitions (these are ALL the tools across all subagents, but you MUST prioritize the tools explicitly mentioned in the Asset Profile's "relevant_tools" section):
{mcp_function_definitions}

Here are some valid single-agent scenarios we generated earlier:
{single_agent_scenarios_json}

Follow Generation Mode: closed-form runs must inline explicit sensor readings (name, value, unit) in operator-facing text, plus any rule text, summaries, or dataset references the task needs.

Already accepted multiagent scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Mode-specific grounding rules:
{mode_requirements}

Avoid these bad patterns:
{forbidden_patterns}

Task: Use these single-agent pieces to construct cohesive multi-agent scenarios from the point of view of an operator or the operator's manager.
A multiagent scenario tests agent communication, workflow orchestration, and decision-making.
For instance, detecting an anomaly using IoT, confirming a failure signature with FMSR or vibration, projecting impact with TSFM, and planning action with WO.

Requirements:
- The 'text' field must stay natural operator language only: do not name MCP tools, API or function identifiers, or add parenthetical hints such as "e.g. get_failure_modes tool". Put concrete tool names only in characteristic_form.
- Every multiagent characteristic_form must mention at least two distinct namespaces chosen from iot, fmsr, tsfm, wo, and vibration.
- Every multiagent characteristic_form must mention concrete MCP tool names from those namespaces.
- Open-form scenarios must only use grounded identifiers from the Asset Profile.
- Closed-form scenarios must contain explicit inline sensor readings (name, value, unit) in the text, and may include summaries, rule text, or dataset identifiers when needed.
- Do not output Unsupported.

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
