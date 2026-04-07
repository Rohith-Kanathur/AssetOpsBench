import json
import logging
import asyncio
import os
import sys
import argparse
from datetime import datetime

from dotenv import load_dotenv

from .models import AssetProfile, Scenario, ScenarioBudget
from .retrieval import retrieve_asset_evidence
from .utils import fetch_hf_fewshot, parse_llm_json
from .prompts import (
    PROFILE_BUILDER_PROMPT,
    SCENARIO_GENERATOR_PROMPT,
    VALIDATE_REPAIR_PROMPT,
    MULTIAGENT_COMBINER_PROMPT,
    BUDGET_ALLOCATOR_PROMPT,
)

from llm.litellm import LiteLLMBackend
from agent.plan_execute.executor import Executor
from agent.cli import _DEFAULT_MODEL

_log = logging.getLogger(__name__)


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_step(
    phase: str,
    info: str,
    details: str | None = None,
    tool_info: str | None = None,
) -> None:
    print(f"  [OK ] Step ({phase}): {info}")
    if tool_info:
        print(f"       {tool_info}")
    if details:
        indented = "\n".join("        " + line for line in details.splitlines())
        print(indented)


def _format_diff(scenario_id: str, old: dict, new: dict) -> str:
    lines = [f"  [VALIDATED] Scenario: {scenario_id}"]
    for key in set(old.keys()) | set(new.keys()):
        if old.get(key) != new.get(key):
            v_old = json.dumps(old.get(key))
            v_new = json.dumps(new.get(key))
            if len(v_old) > 60:
                v_old = v_old[:57] + "..."
            if len(v_new) > 60:
                v_new = v_new[:57] + "..."

            lines.append(f"        {key}: {v_old}")
            lines.append(f"               ↓")
            lines.append(f"               {v_new}")
    return "\n".join(lines)


class ScenarioGeneratorAgent:
    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        show_workflow: bool = False,
        log_dir: str | None = None,
    ) -> None:
        self.llm = LiteLLMBackend(model_id=model_id)
        self.executor = Executor(llm=self.llm)
        self.show_workflow = show_workflow
        self.log_dir = log_dir
        self._log_step = 1

    def _write_log(self, name: str, content: str) -> None:
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"{self._log_step:02d}_{name}.txt")
        with open(path, "w") as f:
            f.write(content)
        self._log_step += 1

    def _handle_parse_failure(
        self,
        step_name: str,
        response: str,
        error_msg: str | None,
    ) -> None:
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

        if self.show_workflow:
            _print_section("Phase 1: Asset Profile Construction")
        asset_profile = await self.build_asset_profile(asset_name, server_desc)

        if self.show_workflow:
            _print_section("Phase 2: Scenario Budget Allocation")
        budget = await self.allocate_budget(asset_profile, total=num_scenarios)

        if self.show_workflow:
            _print_section("Phase 3: Individual Agent - Generation & Validation")

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
                _print_step(
                    f"generate_{subagent}",
                    f"Generating {count} scenarios for subagent: {subagent}",
                    tool_info=f"src: {ds_query}",
                )

            scenarios = self.generate_single_agent_scenarios(subagent, count, asset_profile, server_desc)

            _log.info(f"Validating {count} scenarios for subagent: {subagent}")
            valid_scenarios = self.validate_and_repair(scenarios, asset_profile)

            changed_indices = []
            if len(scenarios) == len(valid_scenarios):
                for i, (s1, s2) in enumerate(zip(scenarios, valid_scenarios)):
                    if json.dumps(s1, sort_keys=True) != json.dumps(s2, sort_keys=True):
                        changed_indices.append(i)
            else:
                 _log.warning(f"Scenario count mismatch after validation: {len(scenarios)} -> {len(valid_scenarios)}")

            generated_count = 0
            diff_logs = []
            for i, s in enumerate(valid_scenarios):
                s["type"] = subagent
                s["id"] = f"{asset_name.replace(' ', '_').lower()}_{subagent}_{len(all_scenarios)+1:02d}"

                if i in changed_indices:
                    diff_logs.append(_format_diff(s["id"], scenarios[i], s))

                all_scenarios.append(Scenario(**s))
                generated_count += 1

            if diff_logs:
                self._write_log(f"{subagent}_validation_changes", "\n\n".join(diff_logs))

            if self.show_workflow:
                _print_step(
                    f"validate_{subagent}",
                    f"Validated {generated_count} scenarios (Made {len(changed_indices)} validation changes)",
                )

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

            multi_generated = 0
            diff_logs_multi = []
            for i, s in enumerate(valid_multi):
                s["type"] = "multiagent"
                s["id"] = f"{asset_name.replace(' ', '_').lower()}_multiagent_{len(all_scenarios)+1:02d}"

                if i in changed_indices_multi:
                    diff_logs_multi.append(_format_diff(s["id"], multi_scenarios[i], s))

                all_scenarios.append(Scenario(**s))
                multi_generated += 1

            if diff_logs_multi:
                self._write_log("multiagent_validation_changes", "\n\n".join(diff_logs_multi))

            if self.show_workflow:
                _print_step("validate_multiagent", f"Validated {len(multi_generated)} multiagent scenarios (Made {len(changed_indices_multi)} validation changes)")

        if self.show_workflow:
            _print_section("Summary")
            print(f"Successfully generated {len(all_scenarios)} total scenarios for asset: {asset_name}.\n")

        return all_scenarios

    async def build_asset_profile(self, asset_name: str, server_desc: dict) -> AssetProfile:
        if self.show_workflow:
            _print_step("researcher_queries", f"Planning bounded ArXiv retrieval for {asset_name}...")

        evidence_bundle = retrieve_asset_evidence(
            asset_name=asset_name,
            server_desc=server_desc,
            llm=self.llm,
            log_writer=self._write_log,
        )

        if self.show_workflow:
            top_titles = "\n".join(
                f" - {candidate.title} (judge_score={candidate.judge_score}/10)"
                for candidate in evidence_bundle.candidates[:3]
            ) or " - No ranked candidates"
            details = (
                f"Canonical asset: {evidence_bundle.canonical_asset_name}\n"
                f"Steps: {evidence_bundle.diagnostics.steps_run}\n"
                f"Metadata requests: {evidence_bundle.diagnostics.metadata_requests}\n"
                f"PDF requests: {evidence_bundle.diagnostics.pdf_requests}\n"
                f"Cooldown: {evidence_bundle.diagnostics.cooldown_seconds:.1f}s\n"
                f"Queries:\n"
                + "\n".join(f" - {query}" for query in evidence_bundle.query_history)
                + "\nTop Evidence:\n"
                + top_titles
            )
            if evidence_bundle.diagnostics.finish_reason:
                details += f"\nFinish: {evidence_bundle.diagnostics.finish_reason}"
            _print_step("arxiv_search_result", "ArXiv evidence retrieval complete", details=details)

        tools_str = json.dumps(server_desc, indent=2)

        prompt = PROFILE_BUILDER_PROMPT.format(
            asset_name=asset_name,
            evidence_bundle_json=evidence_bundle.model_dump_json(indent=2),
            tool_descriptions=tools_str,
        )
        self._write_log("asset_profile_prompt", prompt)

        response = self.llm.generate(prompt)
        self._write_log("asset_profile_response", response)
        parsed, parse_err = parse_llm_json(response)

        if not parsed or not isinstance(parsed, dict):
            self._handle_parse_failure("build_profile", response, parse_err)
            raise RuntimeError(f"Critical failure: Could not construct AssetProfile for '{asset_name}'.")
        profile = AssetProfile(**parsed)

        if self.show_workflow:
            _print_step("build_profile", f"Successfully generated Asset Profile for '{asset_name}'.", profile.model_dump_json(indent=2))

        self._write_log(f"asset_profile_{asset_name}_json", profile.model_dump_json(indent=2))
        return profile

    async def allocate_budget(self, profile: AssetProfile, total: int = 50) -> ScenarioBudget:
        prompt = BUDGET_ALLOCATOR_PROMPT.format(
            total_scenarios=total,
            asset_profile_json=profile.model_dump_json(),
        )
        self._write_log("budget_allocation_prompt", prompt)

        response = self.llm.generate(prompt)
        self._write_log("budget_allocation_response", response)
        parsed, parse_err = parse_llm_json(response)

        if not parsed or not isinstance(parsed, dict) or "allocation" not in parsed:
            self._handle_parse_failure("allocate_budget", response, parse_err)
            raise RuntimeError("Critical failure: Could not dynamically allocate scenario budget.")
        budget = ScenarioBudget(
            total_scenarios=total,
            allocation=parsed["allocation"],
            reasoning=parsed.get("reasoning", ""),
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
            "wo": "Decision Support, Prediction, Knowledge Query",
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
            example_category=example_category,
        )
        self._write_log(f"{subagent}_generation_prompt", prompt)

        response = self.llm.generate(prompt)
        self._write_log(f"{subagent}_generation_response", response)
        parsed, parse_err = parse_llm_json(response)

        if isinstance(parsed, list):
            return parsed
        self._handle_parse_failure(f"generate_{subagent}", response, parse_err)
        return []

    def validate_and_repair(self, scenarios: list[dict], profile: AssetProfile) -> list[dict]:
        if not scenarios:
            return []

        prompt = VALIDATE_REPAIR_PROMPT.format(
            asset_profile_json=profile.model_dump_json(),
            input_scenarios_json=json.dumps(scenarios, indent=2),
        )
        self._write_log("validate_repair_prompt", prompt)

        response = self.llm.generate(prompt)
        self._write_log("validate_repair_response", response)
        parsed, parse_err = parse_llm_json(response)

        if isinstance(parsed, list):
            if len(parsed) > len(scenarios):
                _log.warning(
                    "validate_and_repair returned %d scenarios but only %d were submitted; truncating extras.",
                    len(parsed),
                    len(scenarios),
                )
                parsed = parsed[:len(scenarios)]
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
            single_agent_scenarios_json=json.dumps(single_agents[:10], indent=2),
        )
        self._write_log("multiagent_combiner_prompt", prompt)

        response = self.llm.generate(prompt)
        self._write_log("multiagent_combiner_response", response)
        parsed, parse_err = parse_llm_json(response)

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
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}")
        sys.exit(1)

    if not final_scenarios:
        print("\n[WARNING] No scenarios were successfully generated and validated.")
        sys.exit(0)

    with open(args.output, "w") as f:
        json.dump([s.to_dict() for s in final_scenarios], f, indent=2)

    if not args.show_workflow:
        print(f"Success! Generated {len(final_scenarios)} scenarios at {args.output}")
    else:
        print(f"Scenarios saved to {args.output}")


if __name__ == "__main__":
    main()
