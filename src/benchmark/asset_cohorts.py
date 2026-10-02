"""Prepare existing and synthetic asset cohorts for the native benchmark harness.

Questions and rubrics from the existing corpus are preserved. Coverage is
recorded, never used to remove cases or relabel them as negative scenarios.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re

from scenarios.config import GeneratorConfig
from scenarios.planning import ScenarioPlan
from scenarios.text import slugify_asset_name

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "src/scenarios/huggingface/scenarios/all_utterance.jsonl"
LANES = {"iot": "iot", "fmsr": "fmsr", "tsfm": "tsfm", "workorder": "wo",
         "wo": "wo", "multiagent": "multiagent", "vibration": "vibration",
         "monitoring rule": "tsfm"}


def existing_rows(asset_class: str, source: Path) -> list[dict]:
    """Select the corpus's explicit entity, with a text fallback for absent entities."""
    content = source.read_text(encoding="utf-8")
    rows = ([json.loads(line) for line in content.splitlines() if line.strip()]
            if source.suffix == ".jsonl" else json.loads(content))
    if not isinstance(rows, list):
        raise ValueError("Scenario source must contain an array or JSONL objects")
    selected = []
    for row in rows:
        entity = str(row.get("entity") or "").strip()
        matches = (slugify_asset_name(entity) == slugify_asset_name(asset_class) if entity
                   else bool(re.search(r"\b" + re.escape(asset_class) + r"\b", row["text"], re.I)))
        if not matches:
            continue
        lane = LANES.get(str(row.get("type", "")).strip().lower())
        if lane is None:
            raise ValueError(f"Unknown scenario type {row.get('type')!r}")
        selected.append({**row, "id": str(row["id"]), "type": lane,
                         "provenance": {"cohort": "existing", "source_id": row["id"],
                                        "source_type": row["type"]}})
    if not selected:
        raise ValueError(f"No existing {asset_class} scenarios in {source}")
    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Existing corpus contains duplicate scenario IDs")
    return selected


def stratified_sample(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Round-robin task lanes for a small pilot, without filtering data coverage."""
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        raise ValueError("limit must be positive")
    buckets = {}
    for row in rows:
        buckets.setdefault(row["type"], []).append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked = []
    while len(picked) < limit:
        for lane in sorted(buckets):
            if buckets[lane] and len(picked) < limit:
                picked.append(buckets[lane].pop())
    return picked


def workorder_inventory(asset_class: str) -> dict:
    """Read work-order support without inventing aliases for IoT asset IDs."""
    from servers.wo.couch import CouchClient
    from servers.wo.workorders import list_workorders

    async def collect():
        client = CouchClient(
            base_url=os.environ.get("COUCHDB_URL", "http://localhost:5984"),
            db=os.environ.get("WO_DBNAME", "workorder"),
            username=os.environ.get("COUCHDB_USERNAME", "admin"),
            password=os.environ.get("COUCHDB_PASSWORD", "password"),
        )
        try:
            return await list_workorders(client, page_size=0)
        finally:
            await client.aclose()

    result = asyncio.run(collect())
    inventory = {"database": os.environ.get("WO_DBNAME", "workorder"),
                 "available": bool(result.get("success")),
                 "asset_match_policy": "Explicit asset class, otherwise asset-number class prefix; identifiers are preserved and do not establish IoT aliases."}
    if not result.get("success"):
        return {**inventory, "error": result.get("error"), "matching_count": None,
                "asset_references": [], "work_orders": []}
    asset_slug = slugify_asset_name(asset_class)
    rows = []
    for row in result["data"]["workorders"]:
        declared = row.get("aob_asset_class") or row.get("asset_class")
        matches = (slugify_asset_name(str(declared)) == asset_slug if declared
                   else bool(re.match(re.escape(asset_slug) + r"(?:_|(?=\d)|$)",
                                      slugify_asset_name(str(row.get("assetnum") or "")))))
        if matches:
            rows.append(row)
    references = sorted({(str(row.get("siteid") or ""), str(row.get("assetnum") or ""))
                         for row in rows})
    return {**inventory, "matching_count": len(rows),
            "asset_references": [{"site_id": site, "asset_num": asset}
                                 for site, asset in references], "work_orders": rows}


def audit_asset(asset_class: str) -> dict:
    """Inventory the actual IoT, vibration, registry and failure-mode sources."""
    from scenarios.grounding import discover_grounding
    from servers.fmsr.main import get_failure_modes
    from servers.iot.main import asset_detail, sensor_coverage

    grounding = discover_grounding(asset_class, requested_open_form=True)
    details = []
    for instance in grounding.asset_instances:
        details.append({"asset_id": instance.asset_id, "site_name": instance.site_name,
                        "registry": asset_detail(instance.site_name, instance.asset_id).model_dump(),
                        "sensor_coverage": sensor_coverage(instance.site_name, instance.asset_id).model_dump()})
    scenario_sources = []
    for source in (DEFAULT_SOURCE, ROOT / "src/scenarios/huggingface/task/rule_monitoring_scenarios.jsonl"):
        if not source.is_file():
            continue
        try:
            matches = existing_rows(asset_class, source)
        except ValueError as exc:
            if str(exc).startswith("No existing "):
                continue
            raise
        source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        mentions = [row for row in source_rows
                    if re.search(r"\b" + re.escape(asset_class) + r"s?\b", row.get("text", ""), re.I)]
        selected_ids = {row["id"] for row in matches}
        scenario_sources.append({"source": str(source.resolve()), "count": len(matches),
                                 "task_type_counts": dict(Counter(row["type"] for row in matches)),
                                 "text_mention_count": len(mentions),
                                 "other_entity_text_mention_ids": [str(row["id"]) for row in mentions
                                                                   if str(row["id"]) not in selected_ids],
                                 "included_in_default_cohort": source == DEFAULT_SOURCE})
    return {"grounding": grounding.model_dump(), "assets": details,
            "failure_mode_lookup": get_failure_modes(asset_class).model_dump(),
            "workorder_inventory": workorder_inventory(asset_class),
            "existing_scenario_sources": scenario_sources,
            "coverage_policy": "Existing questions/rubrics remain unchanged, including unavailable assets/data."}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare(asset_class: str, source: Path, output: Path, *, limit=None, seed=42,
            backend="claude-code", generation_model="claude-opus-5-5",
            agent=None, execution_model=None,
            judge="claude-code/claude-fable-5-1", inventory=None) -> dict:
    all_rows = existing_rows(asset_class, source)
    rows = stratified_sample(all_rows, limit, seed)
    plan = ScenarioPlan.model_validate({lane: {"positive": count}
                                       for lane, count in Counter(r["type"] for r in rows).items()})
    config = GeneratorConfig(asset_name=asset_class, backend=backend, model_id=generation_model,
                             mode="open", scenario_plan=plan, retriever="semantic_scholar",
                             log=True, show_workflow=True)
    output.mkdir(parents=True, exist_ok=False)
    real = output / "existing"
    # The native runner accepts any completed prepared suite; this manifest
    # labels the source explicitly rather than claiming it was generated.
    write_json(real / "run.json", {"suite_kind": "existing", "status": "complete",
                                  "config": {"asset_name": asset_class, "num_scenarios": len(rows),
                                             "num_negative_scenarios": 0},
                                  "scenario_count": len(rows), "negative_count": 0})
    write_json(real / "scenarios.json", rows)
    write_json(output / "all_existing.json", all_rows)
    write_json(output / "generation-config.json", config.resolved_dict())
    manifest = {"asset_class": asset_class, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source": str(source.resolve()), "existing_count": len(all_rows), "selected_count": len(rows),
                "selected_ids": [r["id"] for r in rows], "seed": seed,
                "task_type_counts": dict(Counter(r["type"] for r in rows)),
                "selection": "all" if len(rows) == len(all_rows) else "round-robin task-type pilot",
                "generation_examples": "Existing few-shot policy retained; this is not a held-out corpus.",
                "inventory": inventory}
    write_json(output / "cohorts.json", manifest)
    models = json.loads((ROOT / "benchmarks/generated-comparison.json").read_text())["targets"]
    if execution_model is not None:
        models = [{"name": agent or "openai", "agent": agent or "openai", "model_id": execution_model}]
    comparison = {"suite": str(real.resolve()), "judge": judge,
                  "asset_class": asset_class, "allow_same_model_judge": True,
                  "targets": [{**model, "model_key": model["name"],
                               "name": f"{cohort}-{model['name']}", "cohort": cohort,
                               "suite": str((output / cohort).resolve())}
                              for cohort in ("existing", "synthetic") for model in models]}
    write_json(output / "comparison-config.json", comparison)
    return manifest


def summarize_comparison(cohorts: Path, results: Path, output: Path) -> dict:
    """Compare final case outcomes; keep missing judge measurements unavailable."""
    from benchmark.measurement import summarize_cases

    config = json.loads((cohorts / "comparison-config.json").read_text())
    summary = {"cohorts": {}, "caveat": "Existing scenarios may reference unavailable data. The existing corpus is also used as few-shot context; no held-out or causal quality claim is made."}
    for target in config["targets"]:
        root = results / target["name"]
        settings = json.loads((root / "settings.json").read_text())
        if settings.get("model") != target["model_id"] or settings.get("agent") != target["agent"]:
            raise ValueError("Measured executor does not match cohort configuration")
        rows = json.loads((Path(target["suite"]) / "scenarios.json").read_text())
        lanes = {str(row["id"]): row["type"] for row in rows}
        latest = {}
        for path in (root / "measurements").glob("*.json"):
            record = json.loads(path.read_text())
            sid = str(record["scenario_id"])
            if sid not in lanes:
                raise ValueError(f"Measured scenario {sid!r} does not belong to {target['name']}")
            if sid not in latest or record["attempt"] > latest[sid]["attempt"]:
                latest[sid] = record
        records = list(latest.values())
        cohort = summary["cohorts"].setdefault(target["cohort"], {})
        model_key = target.get("model_key", target["model_id"])
        if model_key in cohort:
            raise ValueError(f"Duplicate model {model_key!r} in {target['cohort']} cohort")
        cohort[model_key] = {
            "agent": target["agent"], "model_id": target["model_id"],
            **summarize_cases(records, assigned_cases=len(rows)),
            "by_task_type": {lane: summarize_cases([r for r in records if lanes.get(str(r["scenario_id"])) == lane],
                                                   assigned_cases=sum(t == lane for t in lanes.values()))
                             for lane in sorted(set(lanes.values()))}}
    write_json(output, summary)
    return summary


def main():
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("asset_class")
    prep.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--limit", type=int)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--env-file", type=Path, default=ROOT / ".env")
    prep.add_argument("--backend", default="claude-code")
    prep.add_argument("--generation-model", default="claude-opus-5-5")
    prep.add_argument("--agent", choices=("claude", "openai"))
    prep.add_argument("--execution-model", help="Optional single-model pilot; otherwise use the existing five-model matrix")
    prep.add_argument("--judge", default="claude-code/claude-fable-5-1")
    gen = commands.add_parser("generate")
    gen.add_argument("cohorts", type=Path)
    gen.add_argument("--env-file", type=Path, default=ROOT / ".env")
    summ = commands.add_parser("summarize")
    summ.add_argument("cohorts", type=Path)
    summ.add_argument("results", type=Path)
    summ.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            load_dotenv(args.env_file)
            inventory = audit_asset(args.asset_class)
            if not inventory["grounding"]["open_form_eligible"]:
                parser.error("No matching live asset inventory for open-form generation")
            manifest = prepare(args.asset_class, args.source, args.output_dir, limit=args.limit, seed=args.seed,
                               backend=args.backend, generation_model=args.generation_model, agent=args.agent,
                               execution_model=args.execution_model, judge=args.judge, inventory=inventory)
            print(f"Selected {manifest['selected_count']} of {manifest['existing_count']} existing scenarios; generation plan saved.")
        elif args.command == "generate":
            import asyncio
            from scenarios.generator.service import generate_scenarios

            load_dotenv(args.env_file)
            config = GeneratorConfig.model_validate_json((args.cohorts / "generation-config.json").read_text())
            run = asyncio.run(generate_scenarios(config, output_dir=args.cohorts / "synthetic"))
            if not run.complete:
                parser.exit(1, "Generation incomplete; saved partial output cannot be benchmarked.\n")
            print(f"Generated {len(run.result.scenarios)} synthetic scenarios at {run.output_path}")
        else:
            summarize_comparison(args.cohorts, args.results, args.output)
            print(f"Comparison summary saved at {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cohort preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
