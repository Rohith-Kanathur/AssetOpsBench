import json
import logging
import asyncio
import os
import sys
import argparse
from typing import Any
from datetime import datetime

from dotenv import load_dotenv

from .models import AssetProfile, Scenario, ScenarioBudget
from .utils import fetch_arxiv_studies, fetch_hf_fewshot
from .prompts import (
    PROFILE_BUILDER_PROMPT,
    SCENARIO_GENERATOR_PROMPT,
    VALIDATE_REPAIR_PROMPT,
    MULTIAGENT_COMBINER_PROMPT,
    RESEARCH_QUERY_GENERATOR_PROMPT,
    BUDGET_ALLOCATOR_PROMPT
)

# Reuse existing llm setup
from llm.litellm import LiteLLMBackend
from agent.plan_execute.executor import Executor
from agent.cli import _DEFAULT_MODEL

_log = logging.getLogger(__name__)

def _parse_llm_json(raw: str) -> tuple[Any, str | None]:
    """Helper to parse raw text response containing JSON. Returns (parsed_obj, error_msg)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
        if text.lower().startswith("json"):
            text = text[4:]
            
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        # Fallback to finding first brace/bracket
        start_obj = text.find("{")
        start_arr = text.find("[")
        if start_obj == -1 and start_arr == -1:
            return None, f"No JSON start character found. Error: {str(e)}"
            
        start = start_obj if (start_arr == -1 or (start_obj != -1 and start_obj < start_arr)) else start_arr
        end_char = "}" if start == start_obj else "]"
        end = text.rfind(end_char) + 1
        
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end]), None
            except json.JSONDecodeError as e2:
                return None, f"Failed to parse inner JSON block. Error: {str(e2)}"
    return None, "Unknown parsing error."

def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def _print_step(phase: str, info: str, details: str = None, tool_info: str = None):
    print(f"  [OK ] Step ({phase}): {info}")
    if tool_info:
        print(f"       {tool_info}")
    if details:
        indented = "\n".join("        " + line for line in details.splitlines()) # 8 spaces
        print(indented)

def _format_diff(scenario_id: str, old: dict, new: dict) -> str:
    """Format a visual diff of changed fields between two scenario dicts."""
    lines = [f"  [VALIDATED] Scenario: {scenario_id}"]
    for key in set(old.keys()) | set(new.keys()):
        if old.get(key) != new.get(key):
            v_old = json.dumps(old.get(key))
            v_new = json.dumps(new.get(key))
            # Truncate long values for readability
            if len(v_old) > 60: v_old = v_old[:57] + "..."
            if len(v_new) > 60: v_new = v_new[:57] + "..."
            
            lines.append(f"        {key}: {v_old}")
            lines.append(f"               ↓")
            lines.append(f"               {v_new}")
    return "\n".join(lines)

class ScenarioGeneratorAgent:
    def __init__(self, model_id: str = _DEFAULT_MODEL, show_workflow: bool = False, log_dir: str = None):
        self.llm = LiteLLMBackend(model_id=model_id)
        # We instantiate standard Executor just to get server descriptions easily
        self.executor = Executor(llm=self.llm)
        self.show_workflow = show_workflow
        self.log_dir = log_dir
        self._log_step = 1

    def _write_log(self, name: str, content: str):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"{self._log_step:02d}_{name}.txt")
        with open(path, "w") as f:
            f.write(content)
        self._log_step += 1

    def _handle_parse_failure(self, step_name: str, response: str, error_msg: str | None):
        """Log and optionally print detailed failure information."""
        _log.warning(f"Failed to parse LLM response for '{step_name}'. Error: {error_msg}")
        if self.show_workflow:
            print(f"  [ERR!] Step ({step_name}): LLM output parsing failed.")
            print(f"        Reason: {error_msg}")
            if response:
                print(f"        Raw Response: {response}")
            else:
                print(f"        Raw Response: <empty>")

    async def run(self, asset_name: str, num_scenarios: int = 50) -> list[Scenario]:
        _log.info(f"Starting scenario generation for asset: {asset_name}")
        
        server_desc = await self.executor.get_server_descriptions()
        
        # 1. Build Asset Profile
        if self.show_workflow:
            _print_section("Phase 1: Asset Profile Construction")
        asset_profile = await self.build_asset_profile(asset_name, server_desc)
        
        # 2. Budget Allocation
        if self.show_workflow:
            _print_section("Phase 2: Scenario Budget Allocation")
        budget = await self.allocate_budget(asset_profile, total=num_scenarios)
        
        if self.show_workflow:
            _print_section("Phase 3: Individual Agent - Generation & Validation")
            
        # 3, 4. Subagent generation + Validation
        all_scenarios = []
        for subagent, count in budget.allocation.items():
            if count == 0:
                continue
                
            if subagent == "multiagent":
                continue
                
            if self.show_workflow:
                 _print_section(f"{subagent.upper()} Agent")
            
            _log.info(f"Generating {count} scenarios for subagent: {subagent}")
            
            if self.show_workflow:
                ds_query = f"HuggingFace dataset 'ibm-research/AssetOpsBench' target_type={subagent}"
                _print_step(f"generate_{subagent}", f"Generating {count} scenarios for subagent: {subagent}", tool_info=f"src: {ds_query}")

            scenarios = self.generate_single_agent_scenarios(subagent, count, asset_profile, server_desc)
            
            _log.info(f"Validating {count} scenarios for subagent: {subagent}")
            valid_scenarios = self.validate_and_repair(scenarios, asset_profile)
            
            # Identify changes for logging
            changed_indices = []
            if len(scenarios) == len(valid_scenarios):
                for i, (s1, s2) in enumerate(zip(scenarios, valid_scenarios)):
                    if json.dumps(s1, sort_keys=True) != json.dumps(s2, sort_keys=True):
                        changed_indices.append(i)
            else:
                 # If counts differ, we consider them all potentially changed or just a major structural shift
                 _log.warning(f"Scenario count mismatch after validation: {len(scenarios)} -> {len(valid_scenarios)}")

            subagent_generated = []
            diff_logs = []
            for i, s in enumerate(valid_scenarios):
                s['type'] = subagent
                s['id'] = f"{asset_name.replace(' ', '_').lower()}_{subagent}_{len(all_scenarios)+1:02d}"
                
                # Collect diff for logging if it changed
                if i in changed_indices:
                    diff_logs.append(_format_diff(s['id'], scenarios[i], s))

                try:
                    all_scenarios.append(Scenario(**s))
                    subagent_generated.append(s)
                except Exception as e:
                    _log.warning(f"Skipping incorrectly formatted scenario dict: {s} - Error: {e}")

            if diff_logs:
                self._write_log(f"{subagent}_validation_changes", "\n\n".join(diff_logs))

            if self.show_workflow:
                _print_step(f"validate_{subagent}", f"Validated {len(subagent_generated)} scenarios (Made {len(changed_indices)} validation changes)")

        # 5. Multiagent Combiner
        multiagent_count = budget.allocation.get("multiagent", 0)
        if multiagent_count > 0:
            if self.show_workflow:
                _print_section("Phase 4: Multi-Agent Scenario Construction")
            _log.info(f"Generating {multiagent_count} multiagent scenarios")
            single_dicts = [s.model_dump() for s in all_scenarios]
            
            if self.show_workflow:
                _print_step("generate_multiagent", f"Combining existing scenarios into {multiagent_count} multiagent scenarios.")
                
            multi_scenarios = self.construct_multiagent_scenarios(multiagent_count, single_dicts, asset_profile, server_desc)
            
            _log.info(f"Validating multiagent scenarios")
            valid_multi = self.validate_and_repair(multi_scenarios, asset_profile)
            
            changed_indices_multi = []
            if len(multi_scenarios) == len(valid_multi):
                for i, (s1, s2) in enumerate(zip(multi_scenarios, valid_multi)):
                    if json.dumps(s1, sort_keys=True) != json.dumps(s2, sort_keys=True):
                        changed_indices_multi.append(i)
                        
            multi_generated = []
            diff_logs_multi = []
            for i, s in enumerate(valid_multi):
                s['type'] = "multiagent"
                s['id'] = f"{asset_name.replace(' ', '_').lower()}_multiagent_{len(all_scenarios)+1:02d}"
                
                if i in changed_indices_multi:
                    diff_logs_multi.append(_format_diff(s['id'], multi_scenarios[i], s))
                    
                try:
                    all_scenarios.append(Scenario(**s))
                    multi_generated.append(s)
                except Exception as e:
                    _log.warning(f"Skipping incorrectly formatted multiagent scenario dict: {s} - Error: {e}")

            if diff_logs_multi:
                self._write_log("multiagent_validation_changes", "\n\n".join(diff_logs_multi))

            if self.show_workflow:
                _print_step("validate_multiagent", f"Validated {len(multi_generated)} multiagent scenarios (Made {len(changed_indices_multi)} validation changes)")

        if self.show_workflow:
            _print_section("Summary")
            print(f"Successfully generated {len(all_scenarios)} total scenarios for asset: {asset_name}.\n")

        return all_scenarios

    async def build_asset_profile(self, asset_name: str, server_desc: dict) -> AssetProfile:
        metadata = {}
        
        # 1.1 Generate targeted queries
        query_prompt = RESEARCH_QUERY_GENERATOR_PROMPT.format(asset_name=asset_name)
        self._write_log("research_queries_prompt", query_prompt)
        if self.show_workflow:
            _print_step("researcher_queries", f"Generating ArXiv search strategy for {asset_name}...")
            
        query_response = self.llm.generate(query_prompt)
        self._write_log("research_queries_response", query_response)
        queries, err = _parse_llm_json(query_response)
        
        if not queries or not isinstance(queries, list):
            _log.warning(f"Failed to generate queries, falling back to simple asset name: {asset_name}")
            queries = [f"{asset_name} maintenance sensors", f"{asset_name} failure modes reliability"]

        if self.show_workflow:
            _print_step("arxiv_search_plan", f"Researcher generated {len(queries)} targeted queries", details=json.dumps(queries, indent=2))
            
        # 1.2 Fetch studies using multi-query logic
        studies = fetch_arxiv_studies(queries, metadata_out=metadata)
        
        # Format a summary header with titles and links
        header = f"Asset: {asset_name}\n"
        header += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Total Entries: {len(metadata.get('results_summary', []))}\n\n"
        header += "========================================\n"
        header += "  SOURCES (TITLES & PDF LINKS)\n"
        header += "========================================\n"
        for i, res in enumerate(metadata.get('results_summary', []), 1):
            header += f"{i}. {res['title']}\n   -> {res['url']}\n"
        header += "========================================\n\n"
        
        self._write_log("arxiv_results", header + studies)
        
        if self.show_workflow:
            status_summary = ", ".join(f"HTTP {s}" for s in metadata.get('status_codes', []))
            pdf_urls = metadata.get('pdf_urls', [])
            query_to_pdf = metadata.get('query_to_pdf', {})
            
            details = f"Queries: {len(queries)}\nStatus: {status_summary}\nReturned: {metadata.get('returned_entries', 0)} unique entries"
            if query_to_pdf:
                details += "\n\nQuery Results:"
                for q, urls in query_to_pdf.items():
                    details += f"\n  - Q: \"{q}\""
                    for u in urls:
                        details += f"\n    -> {u}"
            elif pdf_urls:
                # Fallback if the mapping isn't populated for some reason
                details += "\nFetched PDFs:\n" + "\n".join(f" - {u}" for u in pdf_urls)
                
            _print_step("arxiv_search_result", f"ArXiv Research phase complete", details=details)
        
        tools_str = json.dumps(server_desc, indent=2)
        
        prompt = PROFILE_BUILDER_PROMPT.format(
            asset_name=asset_name,
            arxiv_literature=studies,
            tool_descriptions=tools_str
        )
        self._write_log("asset_profile_prompt", prompt)
        
        response = self.llm.generate(prompt)
        self._write_log("asset_profile_response", response)
        parsed, parse_err = _parse_llm_json(response)
        
        if not parsed or not isinstance(parsed, dict):
            self._handle_parse_failure("build_profile", response, parse_err)
            raise RuntimeError(f"Critical failure: Could not construct AssetProfile for '{asset_name}'.")
        else:
            profile = AssetProfile(**parsed)
            
        if self.show_workflow:
            _print_step("build_profile", f"Successfully generated Asset Profile for '{asset_name}'.", profile.model_dump_json(indent=2))
            
        self._write_log(f"asset_profile_{asset_name}_json", profile.model_dump_json(indent=2))
        return profile

    async def allocate_budget(self, profile: AssetProfile, total: int = 50) -> ScenarioBudget:
        prompt = BUDGET_ALLOCATOR_PROMPT.format(
            total_scenarios=total,
            asset_profile_json=profile.model_dump_json()
        )
        self._write_log("budget_allocation_prompt", prompt)
        
        response = self.llm.generate(prompt)
        self._write_log("budget_allocation_response", response)
        parsed, parse_err = _parse_llm_json(response)
        
        if not parsed or not isinstance(parsed, dict) or "allocation" not in parsed:
            self._handle_parse_failure("allocate_budget", response, parse_err)
            raise RuntimeError("Critical failure: Could not dynamically allocate scenario budget.")
        else:
            budget = ScenarioBudget(
                total_scenarios=total,
                allocation=parsed["allocation"],
                reasoning=parsed.get("reasoning", "")
            )
            
        if self.show_workflow:
            details = f"Reasoning: {budget.reasoning}\n\nAllocation:\n"
            details += "\n".join(f" - {k}: {v}" for k, v in budget.allocation.items())
            _print_step("allocate_budget", f"Successfully allocated {total} scenarios across agents.", details=details)
            
        return budget

    def generate_single_agent_scenarios(self, subagent: str, count: int, profile: AssetProfile, server_desc: dict) -> list[dict]:
        few_shots = fetch_hf_fewshot(split="scenarios", target_type=subagent, fallback_if_missing=True)
        few_shots_str = json.dumps(few_shots[:2], indent=2)
        
        category_map = {
            "iot": "Data Query, Knowledge Query",
            "fmsr": "Knowledge Query",
            "tsfm": "Knowledge Query, Anomaly Detection Query, Tuning Query, Inference Query, Complex Query",
            "wo": "Decision Support, Prediction, Knowledge Query"
        }
        category_options = category_map.get(subagent.lower(), "Knowledge Query")
        example_category = category_options.split(",")[0].strip()
        
        subagent_tools = json.dumps(server_desc.get(subagent, {}), indent=2)
        
        prompt = SCENARIO_GENERATOR_PROMPT.format(
            count=count,
            subagent_name=subagent,
            asset_name=profile.asset_name,
            asset_profile_json=profile.model_dump_json(),
            tool_definitions=subagent_tools,
            few_shot_examples=few_shots_str,
            category_options=category_options,
            example_category=example_category
        )
        self._write_log(f"{subagent}_generation_prompt", prompt)
        
        response = self.llm.generate(prompt)
        self._write_log(f"{subagent}_generation_response", response)
        parsed, parse_err = _parse_llm_json(response)
        
        if isinstance(parsed, list):
            return parsed
        self._handle_parse_failure(f"generate_{subagent}", response, parse_err)
        return []

    def validate_and_repair(self, scenarios: list[dict], profile: AssetProfile) -> list[dict]:
        if not scenarios:
            return []
            
        prompt = VALIDATE_REPAIR_PROMPT.format(
            asset_profile_json=profile.model_dump_json(),
            input_scenarios_json=json.dumps(scenarios, indent=2)
        )
        self._write_log("validate_repair_prompt", prompt)
        
        response = self.llm.generate(prompt)
        self._write_log("validate_repair_response", response)
        parsed, parse_err = _parse_llm_json(response)
        
        if isinstance(parsed, list):
            return parsed
        self._handle_parse_failure("validate_repair", response, parse_err)
        return scenarios

    def construct_multiagent_scenarios(self, count: int, single_agents: list[dict], profile: AssetProfile, server_desc: dict) -> list[dict]:
        all_tools = json.dumps(server_desc, indent=2)
        
        prompt = MULTIAGENT_COMBINER_PROMPT.format(
            count=count,
            asset_name=profile.asset_name,
            asset_profile_json=profile.model_dump_json(),
            mcp_function_definitions=all_tools,
            single_agent_scenarios_json=json.dumps(single_agents[:10], indent=2)
        )
        self._write_log("multiagent_combiner_prompt", prompt)
        
        response = self.llm.generate(prompt)
        self._write_log("multiagent_combiner_response", response)
        parsed, parse_err = _parse_llm_json(response)
        
        if isinstance(parsed, list):
            return parsed
        self._handle_parse_failure("multiagent_combiner", response, parse_err)
        return []

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Scenario Generator for AssetOpsBench")
    parser.add_argument("asset_name", help="Name of the asset (e.g. 'Chiller' or 'Wind Turbine')")
    parser.add_argument("--model-id", default=_DEFAULT_MODEL, help="Model ID for LiteLLM")
    parser.add_argument("--output", default="generated_scenarios.json", help="Path to output JSON file")
    parser.add_argument("--num-scenarios", type=int, default=50, help="Total number of scenarios to generate")
    parser.add_argument("--show-workflow", action="store_true", help="Show intermediate pipeline steps in the console")
    parser.add_argument("--log", action="store_true", help="Dump raw prompts and results to a log directory")
    
    args = parser.parse_args()
    
    # We reduce the noise in generic logger when we are controlling console formatting ourselves
    level = logging.WARNING if args.show_workflow else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    
    log_dir = None
    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/gen_{args.asset_name.lower().replace(' ', '_')}_{timestamp}"
        print(f"Logging session to: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    agent = ScenarioGeneratorAgent(model_id=args.model_id, show_workflow=args.show_workflow, log_dir=log_dir)
    
    try:
        final_scenarios = asyncio.run(agent.run(args.asset_name, num_scenarios=args.num_scenarios))
    except (RuntimeError, Exception) as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
    
    # Check if we actually generated anything (for the final summary)
    if not final_scenarios:
        print("\n[WARNING] No scenarios were successfully generated and validated.")
        sys.exit(0) # Not necessarily a crash, but we didn't get results

    with open(args.output, "w") as f:
        json.dump([s.to_dict() for s in final_scenarios], f, indent=2)
        
    # No explicit redundant success print if we already printed a summary in show-workflow
    if not args.show_workflow:
        print(f"Success! Generated {len(final_scenarios)} scenarios at {args.output}")
    else:
        print(f"Scenarios saved to {args.output}")

if __name__ == "__main__":
    main()

