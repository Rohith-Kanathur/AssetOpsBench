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
from .environment import BASELINE_ENV, policy


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
        data = json.loads(compose_path.read_text())
        mounts = data['services']['agent']['volumes']
        current = f"{auth_home.resolve()}:/root/.codex"
        updated = [current if ':/root/.codex' in m else m for m in mounts]
        if mounts != updated:
            data['services']['agent']['volumes'] = updated
            compose_path.write_text(json.dumps(data, indent=2) + '\n')
        return compose_path
    environment = {
        "COUCHDB_URL": "http://database:5984",
        "COUCHDB_USERNAME": "generation",
        "COUCHDB_PASSWORD": secrets.token_hex(16),
    }
    mounts = [f"{workspace.resolve()}:/workspace"]
    request_path = workspace / "request.json"
    if request_path.exists() and policy(json.loads(request_path.read_text())) == "existing":
        mounts.append(f"{(workspace / 'src').resolve()}:/workspace/src:ro")
        environment[BASELINE_ENV] = "/opt/generation/environment-baseline.json"
    if (destination / "seed.json").is_file():
        mounts.append(f"{(workspace / 'data/seed-database').resolve()}:/workspace/data/seed-database:ro")
    mounts.append(f"{auth_home.resolve()}:/root/.codex")
    if kaggle_home.is_dir():
        mounts.append(f"{kaggle_home.resolve()}:/run/kaggle-auth:ro")
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
    if kaggle_home.is_dir():
        # OAuth refresh writes credentials; keep those writes inside the container.
        compose["services"]["agent"]["entrypoint"] = [
            "/bin/sh", "-ec",
            'umask 077; mkdir -p /root/.kaggle; '
            'cp -R /run/kaggle-auth/. /root/.kaggle/; exec "$@"',
            "--",
        ]
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
    """Initialize the prepared seed or default data once, retaining it on resume."""
    request_path = destination / "workspace/request.json"
    request = json.loads(request_path.read_text()) if request_path.exists() else {}
    expected_seed = request.get("seed")
    if expected_seed:
        seed_path = destination / "seed.json"
        if not seed_path.is_file() or json.loads(seed_path.read_text()).get("sha256") != expected_seed.get("sha256"):
            raise ValueError("Prepared seed is missing or differs from the saved request")
    marker = destination / "initialized.json"
    if marker.exists() and expected_seed and json.loads(marker.read_text()).get("seed") != expected_seed["sha256"]:
        raise ValueError("Initialized database does not match the requested seed")
    ensure_image()
    compose(destination, "up", "-d", "--wait", "database")
    if not marker.exists():
        if (destination / "seed.json").is_file():
            from .seed import verify
            seed = verify(destination)
            compose(destination, *execution_arguments(destination), "agent", "python", "-m",
                    "scenarios.generation.seed", "restore", "/workspace/data/seed-database")
            initialized = {"seed": seed["sha256"]}
        else:
            compose(destination, "run", "--rm", "-T", "agent", "python", "-m", "couchdb.init_data")
            initialized = {"manifest": "src/couchdb/scenarios_data/default/manifest.json"}
        marker.write_text(json.dumps(initialized) + "\n")
    baseline_path = destination / "environment-baseline.json"
    if (request_path.exists() and policy(json.loads(request_path.read_text())) == "existing"
            and not baseline_path.exists()):
        result = compose(destination, *execution_arguments(destination), "agent", "python", "-m",
                         "scenarios.generation.environment", capture_output=True, text=True)
        baseline = json.loads((destination / "baseline.json").read_text())
        if (destination / "seed.json").is_file():
            seed = json.loads((destination / "seed.json").read_text())
            baseline["files"].update(seed["files"])
        baseline["databases"] = json.loads(result.stdout)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")


def run(destination: Path, model: str = DEFAULT_MODEL, followup: str | None = None, *,
        harness: str = "codex", reasoning_effort: str = DEFAULT_REASONING,
        service_tier: str = DEFAULT_TIER,
        env_file: Path | None = None) -> None:
    from .progress import write_index

    command = HARNESSES[harness](model, reasoning_effort, service_tier)
    workspace = destination / "workspace"
    for name in ("profile.md", "generate.md"):
        (workspace / name).write_text((HERE / "prompts" / name).read_text())
    from .environment import write_guidance as write_environment_guidance
    write_environment_guidance(workspace)
    from .modes import write_guidance
    write_guidance(workspace)
    request_path = destination / "request.json"
    requested = json.loads((workspace / "request.json").read_text())
    if request_path.exists() and json.loads(request_path.read_text()) != requested:
        raise ValueError("Workspace request differs from the original generation request")
    request_path.write_text(json.dumps(requested, indent=2) + "\n")
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
                        "prompt": prompt, "command": command,
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
            if json.loads((workspace / "request.json").read_text()) != requested:
                raise ValueError("Generation changed the requested asset, budget, seed or environment policy")
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
                            counts=report.get("counts", {}), scenario_count=report.get("scenario_count", 0))
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


def run_with_pool(destination, model=DEFAULT_MODEL, followup=None, *,
                  harness='codex', reasoning_effort=DEFAULT_REASONING,
                  service_tier=DEFAULT_TIER, env_file=None, pool=None):
    """One author at a time; fresh native sessions when a subscription fails."""
    import fcntl
    import re
    from uuid import uuid4
    from agent.codex_accounts import Pool, identity, save
    pool = pool or Pool(model=model, reasoning=reasoning_effort, tier=service_tier)
    pool.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (pool.root / 'author.lock').open('a+') as author_lock:
        try:
            fcntl.flock(author_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another subscription-backed author is already running') from None
        pool.preflight()
        attempted = set()
        for _ in range(len(pool.accounts)):
            with pool.lease(exclude=attempted) as account:
                attempted.add(account['id'])
                auth = pool.root / 'sessions' / uuid4().hex
                auth.mkdir(parents=True, mode=0o700)
                source = Path(account['home']) / 'auth.json'
                save(auth / 'auth.json', json.loads(source.read_text()))
                configure(destination, auth, Path.home() / '.kaggle')
                try:
                    return run(destination, model, followup, harness=harness,
                               reasoning_effort=reasoning_effort, service_tier=service_tier,
                               env_file=env_file)
                except subprocess.CalledProcessError:
                    logs = sorted((destination / 'logs').glob('codex-*.*'), key=lambda p: p.stat().st_mtime)
                    text = '\n'.join(p.read_text()[-4000:] for p in logs[-2:])
                    if not re.search(r'rate.limit|usage.limit|quota|token.*expir|refresh.*token|unauthorized|401', text, re.I):
                        raise
                    # Persist any refresh before app-server reads this account.
                    if identity(auth / 'auth.json')[0] != account['email']:
                        raise RuntimeError('Author login identity changed unexpectedly')
                    save(source, json.loads((auth / 'auth.json').read_text()))
                    try:
                        recovered = pool.reset_if_exhausted(account)
                    except Exception:
                        recovered = False
                    if not recovered:
                        pool.quarantine(account)
                    save(auth / 'auth.json', json.loads(source.read_text()))
                    # run() writes each native attempt separately; it never resumes a session.
                finally:
                    if identity(auth / 'auth.json')[0] != account['email']:
                        raise RuntimeError('Author login identity changed unexpectedly')
                    save(source, json.loads((auth / 'auth.json').read_text()))
        raise RuntimeError('No further authoring subscription available')


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
    baseline = destination / "environment-baseline.json"
    if baseline.exists():
        shutil.copyfile(baseline, destination / "tooling/environment-baseline.json")
    logs = destination / "logs"
    logs.mkdir(exist_ok=True)
    research_log = logs / "research.jsonl"
    research_log.touch(exist_ok=True)
    return ["run", "--rm", "-T", "--volume", f"{destination / 'tooling'}:/opt/generation:ro",
            "--env", "PYTHONPATH=/opt/generation:/workspace/src", "--volume", f"{logs}:/run-logs:ro",
            "--volume", f"{research_log}:/run-logs/research.jsonl",
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
