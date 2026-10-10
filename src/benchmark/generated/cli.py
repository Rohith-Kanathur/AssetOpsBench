"""Run and judge generated scenarios using their declared evaluation capabilities."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time

from dotenv import dotenv_values

from .auth import prepare_auth, private_json
from .judge import judge_case
from .report import load_case, write_report
from . import sandbox

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "general-execution": {"stirrup": "litellm_proxy/openai/gpt-5.6-luna"},
    "mcp-only": {"stirrup": "litellm_proxy/openai/gpt-5.6-luna"},
}
KEYS = ("ZAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY",
        "LITELLM_BASE_URL", "TOKENROUTER_API_KEY", "TOKENROUTER_BASE_URL",
        "AI_GATEWAY_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-") or "case"


def runner_models(runners, mode):
    """Expand one model or a model list per runner into distinct executions."""
    supported = {"stirrup"} | ({"codex", "claude-code", "zcode"} if mode == "general-execution"
                              else {"openai-agent", "claude-agent"})
    if not isinstance(runners, dict) or not runners or set(runners) - supported:
        raise ValueError(f"{mode} supports these runners: {', '.join(sorted(supported))}")
    pairs = []
    for runner, models in runners.items():
        models = [models] if isinstance(models, str) else models
        if (not isinstance(models, list) or not models
                or any(not isinstance(model, str) or not model.strip() for model in models)):
            raise ValueError(f"{runner} needs a model name or a nonempty list of model names")
        if len(set(models)) != len(models):
            raise ValueError(f"Duplicate models for {runner}")
        pairs.extend((runner, model) for model in models)
    return pairs


def evaluation_credentials(values):
    credentials = {name: os.environ.get(name) or values.get(name) for name in KEYS
                   if os.environ.get(name) or values.get(name)}
    if credentials.get("AI_GATEWAY_API_KEY") and not credentials.get("LITELLM_API_KEY"):
        credentials.update(LITELLM_API_KEY=credentials["AI_GATEWAY_API_KEY"],
                           LITELLM_BASE_URL="https://ai-gateway.vercel.sh/v1")
    return credentials


def default_runners(credentials):
    """Prefer available router credits; keep the selected route fixed per run."""
    if credentials.get("TOKENROUTER_API_KEY") and credentials.get("TOKENROUTER_BASE_URL"):
        model = "tokenrouter/openai/gpt-5.6-luna"
    elif credentials.get("LITELLM_API_KEY") and credentials.get("LITELLM_BASE_URL"):
        model = DEFAULTS["general-execution"]["stirrup"]
    else:
        model = "openai/gpt-5.6-luna"
    return {"stirrup": model}


def stirrup_case(case, model, endpoints, timeout, credentials, settings):
    """Run the same API harness per model, isolated from generator credentials."""
    private_json(case / "config/mcp-host.json", {"mcpServers": endpoints})
    environment = {key: value for key, value in os.environ.items() if key in {
        "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
        "DOCKER_HOST", "DOCKER_CONTEXT"}}
    environment.update(credentials, PYTHONPATH=str(ROOT), OTEL_SDK_DISABLED="true")
    native = case / "native"
    native.mkdir(exist_ok=True)
    command = [sys.executable, "-m", "benchmark.generated.stirrup_worker", "--model", model,
               "--mcp-config", str(case / "config/mcp-host.json"), "--question-file",
               str(case / "config/question.txt"), "--workspace", str(case / "workspace"),
               "--output", str(native / "result.json"), "--timeout", str(timeout)]
    for name, value in settings.items():
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    try:
        with (case / "execution.log").open("w") as log:
            process = subprocess.Popen(command, cwd=case / "workspace", env=environment,
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                process.wait(timeout=timeout + 90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if not (native / "result.json").exists():
            raise RuntimeError("Stirrup exited without a result")
        return json.loads((native / "result.json").read_text())
    finally:
        container = native / "code-container.json"
        if container.exists():
            identity = json.loads(container.read_text()).get("id", "")
            if re.fullmatch(r"[0-9a-f]{64}", identity):
                subprocess.run(["docker", "rm", "--force", identity], capture_output=True, timeout=30)


def coding_case(case, runner, model, timeout, credentials=None):
    harness = runner if runner in {"codex", "zcode"} else "claude"
    zai = harness == "zcode" or harness == "claude" and model.startswith("zai/")
    environment = os.environ.copy()
    if zai:
        key = (credentials or {}).get("ZAI_API_KEY") or environment.get("ZAI_API_KEY")
        if not key:
            raise ValueError("ZAI_API_KEY is required for this z.ai runner")
        environment["ZAI_API_KEY"] = key
    auth = case / "auth"
    if zai:
        auth.mkdir(mode=0o700, exist_ok=True)
    else:
        prepare_auth(auth, harness)
    (case / "native").mkdir(exist_ok=True)
    configuration = json.loads((case / "compose.json").read_text())
    configuration["services"]["agent"] = {
        "image": "assetops-zcode-evaluation:local" if harness == "zcode" else sandbox.IMAGE,
        "working_dir": "/workspace", "user": f"{os.getuid()}:{os.getgid()}",
        "environment": {"HOME": "/tmp", "PYTHONPATH": "/opt/agent", "ASSETOPS_EXECUTION_ISOLATED": "1"},
        "volumes": [f"{case / 'workspace'}:/workspace", f"{case / 'config'}:/config:ro",
                    f"{auth}:/auth:ro", f"{case / 'native'}:/results",
                    f"{ROOT / 'agent/coding_agent'}:/opt/agent/coding_agent:ro"],
        "init": True, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"],
    }
    private_json(case / "compose.json", configuration)
    arguments = ["run", "--rm", "-T"]
    if zai:
        arguments += ["--env", "ZAI_API_KEY"]
    elif os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") and harness == "claude":
        arguments += ["--env", "CLAUDE_CODE_OAUTH_TOKEN"]
    arguments += ["agent", "python", "-m", "coding_agent", "--harness", harness,
                  "--workspace", "/workspace", "--mcp-config", "/config/mcp.json",
                  "--question-file", "/config/question.txt", "--output-dir", "/results",
                  "--auth-home", f"/auth/.{harness}", "--model", model,
                  "--reasoning-effort", "xhigh" if harness == "codex" else "high",
                  "--timeout", str(timeout)]
    try:
        with (case / "execution.log").open("w") as log:
            try:
                sandbox.compose(case, *arguments, env=environment,
                                stdout=log, stderr=log, timeout=timeout + 90)
            except subprocess.CalledProcessError:
                if not (case / "native/result.json").exists():
                    raise
        return json.loads((case / "native/result.json").read_text())
    finally:
        shutil.rmtree(auth, ignore_errors=True)


def sdk_case(case, runner, model, endpoints, timeout, credentials):
    private_json(case / "config/mcp-host.json", {"mcpServers": endpoints})
    environment = {**os.environ, **credentials, "PYTHONPATH": str(ROOT), "OTEL_SDK_DISABLED": "true"}
    environment.pop("AGENT_TRAJECTORY_DIR", None)
    (case / "native").mkdir(exist_ok=True)
    command = [sys.executable, "-m", "benchmark.generated.sdk_worker", "--runner",
               "openai" if runner == "openai-agent" else "claude", "--model", model,
               "--mcp-config", str(case / "config/mcp-host.json"), "--question-file",
               str(case / "config/question.txt"), "--output", str(case / "native/result.json"),
               "--timeout", str(timeout)]
    auth = case / "auth" if runner == "claude-agent" else None
    try:
        if auth is not None:
            api_auth = any(environment.get(name) for name in
                           ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"))
            for prefix, key in (("litellm_proxy/", "LITELLM_API_KEY"), ("tokenrouter/", "TOKENROUTER_API_KEY")):
                api_auth |= model.startswith(prefix) and bool(environment.get(key))
            if not api_auth:
                prepare_auth(auth, "claude")
            auth.mkdir(parents=True, exist_ok=True, mode=0o700)
            environment.update(HOME=str(auth), CLAUDE_CONFIG_DIR=str(auth / ".claude"),
                               CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", ENABLE_CLAUDEAI_MCP_SERVERS="false")
            for name in ("CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_SAFE_MODE"):
                environment.pop(name, None)
        with (case / "execution.log").open("w") as log:
            result = subprocess.Popen(command, cwd=case / "workspace", env=environment,
                                      stdout=log, stderr=log, start_new_session=True)
            try:
                result.wait(timeout=timeout + 90)
            except subprocess.TimeoutExpired:
                os.killpg(result.pid, signal.SIGKILL)
                result.wait()
                raise
        path = case / "native/result.json"
        if not path.exists():
            raise RuntimeError(f"SDK process exited {result.returncode} without a result")
        return json.loads(path.read_text())
    finally:
        if auth is not None:
            shutil.rmtree(auth, ignore_errors=True)


def execute_case(root, case, scenario, runner, model, timeout, credentials, retry=False, settings=None):
    previous = json.loads((case / "result.json").read_text())
    if previous["status"] == "completed" or previous["status"] == "error" and not retry:
        judge_case(case)
        return load_case(case)
    if previous["status"] in {"error", "running"} or (case / "workspace").exists():
        if (case / "compose.json").exists():
            sandbox.compose(case, "down", "--volumes", "--remove-orphans",
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(case / "auth", ignore_errors=True)
        archived = root / "attempts" / f"{case.name}-{time.time_ns()}"
        archived.parent.mkdir(exist_ok=True)
        shutil.move(case, archived)
        case.mkdir(parents=True)
        private_json(case / "scenario.json", scenario)
    started = time.monotonic()
    record = {"scenario_id": scenario["id"], "positive": scenario.get("positive"),
              "domain": scenario["type"], "runner": runner, "model": model,
              "status": "running", "answer": "", "trajectory": {}}
    private_json(case / "result.json", record)
    try:
        with sandbox.environment(root, case, scenario) as endpoints:
            if runner == "stirrup":
                execution = stirrup_case(case, model, endpoints, timeout, credentials, settings or {})
            elif runner in {"codex", "claude-code", "zcode"}:
                execution = coding_case(case, runner, model, timeout, credentials)
            else:
                execution = sdk_case(case, runner, model, endpoints, timeout, credentials)
            record.update(execution, runner=runner, model=model)
        record["artifacts"] = sandbox.artifact_inventory(case, scenario)
    except Exception as exc:
        record.update(status="error", error=type(exc).__name__)
    record["duration_seconds"] = round(time.monotonic() - started, 3)
    private_json(case / "result.json", record)
    judge_case(case)
    return load_case(case)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--runners", help="JSON mapping runner names to model(s); default: Stirrup/Luna through available router credits")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--ids", help="Comma-separated IDs for a bounded execution smoke test")
    parser.add_argument("--snapshot", action="store_true", help="Use a validated frozen environment/database/inputs/scenarios snapshot")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    if args.max_turns < 1 or not 0 < args.max_output_tokens <= 100_000:
        parser.error("max-turns must be positive; max-output-tokens must be between 1 and 100000")
    root = args.directory.expanduser().resolve()
    if args.snapshot:
        sandbox.import_snapshot(args.generation, root)
    else:
        sandbox.snapshot(args.generation, root)
    mode = json.loads((root / "snapshot.json").read_text())["request"]["generation_mode"]
    values = dotenv_values(args.env_file) if args.env_file.exists() else {}
    credentials = evaluation_credentials(values)
    try:
        runners = json.loads(args.runners) if args.runners else default_runners(credentials)
        selections = runner_models(runners, mode)
    except ValueError as exc:
        parser.error(str(exc))
    scenarios = json.loads((root / "scenarios.json").read_text())
    selected = set(args.ids.split(",")) if args.ids else {str(s["id"]) for s in scenarios}
    if not selected <= {str(s["id"]) for s in scenarios}:
        parser.error("Unknown scenario ID")
    cohort = {"format_version": 1, "generation_mode": mode, "runners": runners,
              "scenario_ids": [s["id"] for s in scenarios]}
    settings = {name: getattr(args, name) for name in ("max_turns", "max_output_tokens", "reasoning_effort", "temperature")}
    if "stirrup" in runners:
        cohort["stirrup_settings"] = {**settings, "timeout": args.timeout}
    cohort_path = root / "cohort.json"
    if cohort_path.exists() and json.loads(cohort_path.read_text()) != cohort:
        parser.error("Evaluation cohort differs from the saved runner/model matrix; choose a new directory")
    if not cohort_path.exists() and (root / "cases").exists():
        parser.error("Existing cases have no compatible cohort manifest; choose a new directory")
    private_json(cohort_path, cohort)
    planned, seen = [], set()
    for scenario in scenarios:
        for runner, model in selections:
            case = root / "cases" / f"{slug(runner)}-{slug(model)}-{slug(scenario['id'])}"
            if case in seen:
                parser.error("Scenario or model identifiers collide after filename normalization")
            seen.add(case)
            expected = {"scenario_id": scenario["id"], "positive": scenario.get("positive"),
                        "domain": scenario["type"], "runner": runner, "model": model}
            result_path, scenario_path = case / "result.json", case / "scenario.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if any(saved.get(key) != value for key, value in expected.items()) or saved.get("status") not in {"pending", "running", "completed", "error"}:
                    parser.error(f"Incompatible saved case: {case.name}")
            if scenario_path.exists() and json.loads(scenario_path.read_text()) != scenario:
                parser.error(f"Saved scenario changed: {case.name}")
            case.mkdir(parents=True, exist_ok=True)
            private_json(scenario_path, scenario)
            if not result_path.exists():
                private_json(result_path, {**expected, "status": "pending"})
            planned.append((case, scenario, runner, model))
    asset = json.loads((root / "snapshot.json").read_text())["request"]["asset_class"]
    title = f"{asset} · Stirrup" if set(runners) == {"stirrup"} else f"{asset} · {mode}"
    def report():
        write_report(root, [load_case(case) for case, *_ in planned], title=title)
    report()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(execute_case, root, case, scenario, runner, model,
                               args.timeout, credentials, args.retry_failed, settings): (case, runner, model, scenario["id"])
                   for case, scenario, runner, model in planned if str(scenario["id"]) in selected}
        for future in as_completed(futures):
            _, runner, model, sid = futures[future]
            result = future.result()
            print(f"{runner} / {model} / {sid}: execution {result['status']}; judge {result['grading']['status']}", flush=True)
            report()
    print(f"Results: {root / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
