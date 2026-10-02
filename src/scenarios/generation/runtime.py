"""Run Codex and the existing MCP servers inside a disposable Docker environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import subprocess

from .harnesses import DEFAULT_MODEL, DEFAULT_REASONING, DEFAULT_TIER, HARNESSES


HERE = Path(__file__).parent
IMAGE = "assetops-scenario-generation:local"


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
        service_tier: str = DEFAULT_TIER) -> None:
    command = HARNESSES[harness](model, reasoning_effort, service_tier)
    workspace = destination / "workspace"
    for name in ("profile.md", "generate.md"):
        (workspace / name).write_text((HERE / "prompts" / name).read_text())
    prompt = followup or (
        "Generate scenarios for the asset class and scope in request.json. "
        "Read profile.md and inspect the available environment, then follow generate.md. "
        "Save the completed scenarios and a concise output/README.md."
    )
    start(destination)
    logs = destination / "logs"
    logs.mkdir(exist_ok=True)
    sequence = len(list(logs.glob("codex-*.jsonl"))) + 1
    metadata = {"request": json.loads((workspace / "request.json").read_text()),
                "harness": harness, "requested_model": model,
                "reasoning_effort": reasoning_effort, "service_tier": service_tier,
                "prompt_hashes": {n: hashlib.sha256((workspace / n).read_bytes()).hexdigest()
                                  for n in ("profile.md", "generate.md")}}
    (logs / f"run-{sequence}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (logs / f"codex-{sequence}.jsonl").open("w") as out, \
         (logs / f"codex-{sequence}.stderr").open("w") as err:
        compose(destination, "run", "--rm", "-T", "agent", *command,
                input=prompt, text=True, stdout=out, stderr=err)


def stop(destination: Path) -> None:
    """Stop services while retaining the generated environment for review."""
    compose(destination, "down")
