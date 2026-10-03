"""Run Codex and the existing MCP servers inside a disposable Docker environment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import shutil
import subprocess

from dotenv import dotenv_values

from .harnesses import DEFAULT_MODEL, DEFAULT_REASONING, DEFAULT_TIER, HARNESSES


HERE = Path(__file__).parent
IMAGE = "assetops-scenario-generation:local"


class IncompleteGeneration(ValueError):
    pass


def configure(destination: Path, auth_home: Path, kaggle_home: Path) -> Path:
    """Mount only the prepared workspace and client credentials, never the host repo."""
    workspace = destination / "workspace"
    if not (destination / "baseline.json").exists():
        raise ValueError("Prepare a workspace first")
    if not (auth_home / "auth.json").is_file():
        raise ValueError("Codex file-based login is required; run codex login first")
    compose_path = destination / "compose.json"
    if compose_path.exists():
        return compose_path
    environment = {
        "COUCHDB_URL": "http://database:5984",
        "COUCHDB_USERNAME": "generation",
        "COUCHDB_PASSWORD": secrets.token_hex(16),
    }
    mounts = [f"{workspace.resolve()}:/workspace"]
    mounts.append(f"{(auth_home / 'auth.json').resolve()}:/root/.codex/auth.json:ro")
    if kaggle_home.is_dir():
        mounts.append(f"{kaggle_home.resolve()}:/root/.kaggle:ro")
    name = "aob-generation-" + hashlib.sha256(str(destination.resolve()).encode()).hexdigest()[:10]
    compose = {
        "name": name,
        "services": {
            "database": {
                "image": "couchdb:3.5",
                "entrypoint": ["tini", "--", "sh", "-c",
                    "printf '[couchdb]\\nsingle_node=true\\n' > /opt/couchdb/etc/local.d/generation.ini; "
                    "exec /docker-entrypoint.sh /opt/couchdb/bin/couchdb"],
                "environment": {"COUCHDB_USER": environment["COUCHDB_USERNAME"],
                                "COUCHDB_PASSWORD": environment["COUCHDB_PASSWORD"]},
                "volumes": ["database:/opt/couchdb/data"],
                "healthcheck": {"test": ["CMD", "curl", "-fsS", "http://localhost:5984/_up"],
                                "interval": "2s", "timeout": "2s", "retries": 30},
            },
            "agent": {
                "image": IMAGE, "working_dir": "/workspace",
                "environment": environment, "volumes": mounts,
                "init": True, "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "depends_on": {"database": {"condition": "service_healthy"}},
            },
        },
        "volumes": {"database": {}},
    }
    compose_path.write_text(json.dumps(compose, indent=2) + "\n")
    compose_path.chmod(0o600)
    return compose_path


def compose(destination: Path, *arguments: str, **kwargs):
    return subprocess.run(
        ["docker", "compose", "-f", str(destination / "compose.json"), *arguments],
        check=True, **kwargs,
    )


def build() -> None:
    subprocess.run(["docker", "build", "-t", IMAGE, str(HERE / "runtime")], check=True)


def ensure_image() -> None:
    result = subprocess.run(["docker", "image", "inspect", IMAGE],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        build()


def start(destination: Path) -> None:
    """Initialize the normal fixtures once, keeping data from later sessions."""
    ensure_image()
    compose(destination, "up", "-d", "--wait", "database")
    marker = destination / "initialized.json"
    if not marker.exists():
        compose(destination, "run", "--rm", "-T", "agent", "python", "-m", "couchdb.init_data")
        marker.write_text(json.dumps({"manifest": "src/couchdb/scenarios_data/default/manifest.json"}) + "\n")


def run(destination: Path, model: str = DEFAULT_MODEL, followup: str | None = None, *,
        harness: str = "codex", reasoning_effort: str = DEFAULT_REASONING,
        service_tier: str = DEFAULT_TIER, env_file: Path | None = None) -> None:
    from .progress import write_index

    command = HARNESSES[harness](model, reasoning_effort, service_tier)
    workspace = destination / "workspace"
    for name in ("profile.md", "generate.md"):
        (workspace / name).write_text((HERE / "prompts" / name).read_text())
    shutil.copytree(HERE / "references", workspace / "references", dirs_exist_ok=True)
    environment = research_environment(env_file or Path.cwd() / ".env")
    prompt = followup or (
        "Generate scenarios for the asset class and scope in request.json. "
        "Read profile.md and inspect the available environment, then follow generate.md. "
        "Use the domain examples in references/examples.json. "
        "Run the shared checks and address findings before finishing. "
        "Save the completed scenarios and a concise output/README.md."
    )
    logs = destination / "logs"
    logs.mkdir(exist_ok=True)
    save_status(destination, "running")
    meta_path, metadata = None, None
    try:
        start(destination)
        for attempt in range(3):
            sequence = max((int(p.stem.split("-")[-1]) for p in logs.glob("codex-*.jsonl")), default=0) + 1
            metadata = {"request": json.loads((workspace / "request.json").read_text()),
                        "harness": harness, "requested_model": model,
                        "reasoning_effort": reasoning_effort, "service_tier": service_tier,
                        "started_at": now(), "process_status": "running", "validation_status": "pending",
                        "semantic_scholar_authenticated": bool(environment.get("SEMANTIC_SCHOLAR_API_KEY")),
                        "prompt_hashes": {n: hashlib.sha256((workspace / n).read_bytes()).hexdigest()
                                          for n in ("profile.md", "generate.md", "references/examples.json")}}
            meta_path = logs / f"run-{sequence}.json"
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
            save_status(destination, "running", attempt=sequence)
            write_index(destination)
            try:
                with (logs / f"codex-{sequence}.jsonl").open("w") as out, \
                     (logs / f"codex-{sequence}.stderr").open("w") as err:
                    compose(destination, *execution_arguments(destination),
                            "--env", "SEMANTIC_SCHOLAR_API_KEY", "agent", *command,
                            input=prompt, text=True, env=environment, stdout=out, stderr=err)
            except BaseException as exc:
                metadata.update(process_status="failed", exit_code=getattr(exc, "returncode", None),
                                error_type=type(exc).__name__, finished_at=now())
                meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
                raise
            metadata.update(process_status="succeeded", exit_code=0, finished_at=now())
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
            save_status(destination, "checking", attempt=sequence)
            report = check(destination)
            review_path = logs / f"review-{sequence}.json"
            review_path.write_text(json.dumps(report, indent=2) + "\n")
            if (destination / "review.stderr").exists():
                shutil.copyfile(destination / "review.stderr", logs / f"review-{sequence}.stderr")
            metadata["validation_status"] = "failed" if report["errors"] else "passed"
            metadata["review_file"] = str(review_path.relative_to(destination))
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
            if not report["errors"]:
                save_status(destination, "complete", attempt=sequence,
                            counts={k: report.get(k) for k in ("positive", "negative")})
                write_index(destination)
                return
            save_status(destination, "repairing" if attempt < 2 else "incomplete",
                        attempt=sequence, errors=report["errors"])
            write_index(destination)
            prompt = (
                "Continue the saved scenario generation. Read profile.md, generate.md, request.json "
                "and output/review.json. Address the reported errors in the existing artifacts and "
                "environment without changing the requested budget. Preserve source evidence. "
                "Run the shared checks again and update output/README.md with any remaining gaps."
            )
        raise IncompleteGeneration("Generation remains incomplete after two repair attempts; see review.json")
    except IncompleteGeneration:
        raise
    except BaseException as exc:
        if meta_path and metadata:
            if metadata["process_status"] == "running":
                metadata.update(process_status="failed", finished_at=now())
            elif metadata["validation_status"] == "pending":
                metadata["validation_status"] = "error"
            metadata["error_type"] = type(exc).__name__
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        save_status(destination, "failed", error_type=type(exc).__name__)
        write_index(destination)
        raise


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_status(destination: Path, status: str, **details) -> None:
    pending = destination / "status.tmp"
    pending.write_text(json.dumps({"status": status, "updated_at": now(), **details}, indent=2) + "\n")
    pending.replace(destination / "status.json")


def research_environment(env_file: Path) -> dict[str, str]:
    """Forward the research key; exported values take precedence over dotenv."""
    environment = os.environ.copy()
    name = "SEMANTIC_SCHOLAR_API_KEY"
    if name not in environment:
        value = dotenv_values(env_file).get(name) if env_file.is_file() else None
        if value:
            environment[name] = value
    return environment


def execution_arguments(destination: Path) -> list[str]:
    support = destination / "tooling/scenarios/generation"
    support.mkdir(parents=True, exist_ok=True)
    for path in HERE.glob("*.py"):
        (support / path.name).write_bytes(path.read_bytes())
    logs = destination / "logs"
    logs.mkdir(exist_ok=True)
    return ["run", "--rm", "-T", "--volume", f"{destination / 'tooling'}:/opt/generation:ro",
            "--env", "PYTHONPATH=/opt/generation:/workspace/src", "--volume", f"{logs}:/run-logs",
            "--env", "SCENARIO_RESEARCH_LOG=/run-logs/research.jsonl"]


def check(destination: Path) -> dict:
    """Run the same read-only checker used by the generation agent."""
    start(destination)
    exit_code = 0
    with (destination / "review.json").open("w") as out, (destination / "review.stderr").open("w") as err:
        try:
            compose(destination, *execution_arguments(destination), "agent", "python", "-m",
                    "scenarios.generation.review", "--workspace", "/workspace", "--stage", "all",
                    stdout=out, stderr=err, timeout=300)
        except subprocess.CalledProcessError as exc:
            exit_code = exc.returncode
    try:
        report = json.loads((destination / "review.json").read_text())
        if not isinstance(report.get("errors"), list):
            raise ValueError("Malformed report")
        if exit_code and not report["errors"]:
            report["errors"].append(f"Checker exited with status {exit_code}; see review.stderr")
    except (ValueError, AttributeError):
        report = {"errors": ["Checker failed to produce a report; see review.stderr"]}
    (destination / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "workspace/output/review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def stop(destination: Path) -> None:
    """Stop services while retaining the generated environment for review."""
    compose(destination, "down")
