"""Fresh database and MCP environment per scenario; answers stay outside the agent."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import time
import tempfile

import requests

from scenarios.generation import runtime
from .auth import private_json
from .tools import SERVERS

HERE = Path(__file__).parent.resolve()
IMAGE = "assetops-scenario-evaluation:local"
EXCLUDED = {"evidence", "exports", "__pycache__", ".pytest_cache", ".git", ".env"}


def checked_file(root, relative):
    path = Path(relative)
    target = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Input escapes the generation workspace: {relative}")
    if not target.is_file() or any((root / Path(*path.parts[:i])).is_symlink()
                                   for i in range(1, len(path.parts) + 1)):
        raise ValueError(f"Input is not a regular file: {relative}")
    return target


def file_hashes(root, folders, *, exclude=()):
    hashes = {}
    for folder in folders:
        base = root / folder
        if base.is_symlink():
            raise ValueError(f"Snapshot contains a symlink: {folder}")
        paths = base.rglob("*") if base.is_dir() else [base]
        for path in paths:
            if any(part in exclude for part in path.relative_to(root).parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Snapshot contains a symlink: {path.relative_to(root)}")
            if path.is_file():
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def generation_hashes(workspace, scenarios):
    paths = ["src", "data", "request.json", "output/scenarios.json", "output/requirements.txt"]
    inputs = [relative for row in scenarios for relative in row.get("execution", {}).get("input_files", [])]
    for relative in inputs:
        checked_file(workspace, relative)
    paths += inputs
    return file_hashes(workspace, paths, exclude=EXCLUDED)


def snapshot(generation, destination):
    """Freeze source, explicit inputs and database; reject changed saved snapshots."""
    generation, destination = Path(generation).resolve(), Path(destination).resolve()
    workspace = generation / "workspace"
    folders = ("environment", "database", "inputs", "scenarios.json")
    if (destination / "snapshot.json").exists():
        saved = json.loads((destination / "snapshot.json").read_text())
        if saved.get("format_version") != 1:
            raise ValueError("Incompatible evaluation snapshot; choose a new directory")
        if saved["generation"] != str(generation):
            raise ValueError("Evaluation directory belongs to another generation")
        if file_hashes(destination, folders) != saved["files"]:
            raise ValueError("Evaluation snapshot files changed; choose a new directory")
        scenarios = json.loads((destination / "scenarios.json").read_text())
        if generation_hashes(workspace, scenarios) != saved["generation_files"]:
            raise ValueError("Generation source or inputs changed after the evaluation snapshot")
        return
    state = json.loads((generation / "status.json").read_text())
    if state.get("status") != "complete":
        raise ValueError("Generation must pass its checks before evaluation")
    request = json.loads((workspace / "request.json").read_text())
    if request.get("generation_mode") not in {"mcp-only", "general-execution"}:
        raise ValueError("Generation has no declared evaluation mode")
    original = generation / "request.json"
    if original.exists() and json.loads(original.read_text()) != request:
        raise ValueError("Generation request changed after execution")
    scenarios = json.loads((workspace / "output/scenarios.json").read_text())
    source_hashes = generation_hashes(workspace, scenarios)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Nonempty evaluation directory has no compatible snapshot")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=destination) as temporary:
        staging = Path(temporary)
        frozen = staging / "environment"
        frozen.mkdir()
        for name in ("src", "data"):
            shutil.copytree(workspace / name, frozen / name,
                            ignore=shutil.ignore_patterns(*EXCLUDED))
        requirements = workspace / "output/requirements.txt"
        if requirements.exists():
            shutil.copyfile(requirements, frozen / "requirements.txt")
        inputs = staging / "inputs"
        outputs = {item["path"] for row in scenarios for item in row.get("execution", {}).get("output_files", [])}
        for scenario in scenarios:
            for relative in scenario.get("execution", {}).get("input_files", []):
                source = checked_file(workspace, relative)
                if (relative in outputs or relative in {"request.json", "profile.md", "generate.md"}
                        or any(part in EXCLUDED for part in Path(relative).parts)
                        or relative.startswith(("output/", "scripts/", "references/", "src/"))):
                    raise ValueError(f"Evaluation input contains generation evidence or code: {relative}")
                target = inputs / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        runtime.start(generation)
        database = staging / "database"
        database.mkdir()
        runtime.compose(generation, "run", "--rm", "-T", "--volume", f"{database}:/snapshot",
                        "--volume", f"{HERE / 'database.py'}:/snapshot_database.py:ro",
                        "agent", "python", "/snapshot_database.py", "export", "/snapshot",
                        stdout=subprocess.DEVNULL)
        if generation_hashes(workspace, scenarios) != source_hashes:
            raise ValueError("Generation changed while preparing the evaluation snapshot")
        (staging / "scenarios.json").write_text(json.dumps(scenarios, indent=2) + "\n")
        private_json(staging / "snapshot.json", {"format_version": 1, "generation": str(generation),
                     "request": request, "baseline": json.loads((generation / "baseline.json").read_text())["baseline"],
                     "generation_files": source_hashes, "files": file_hashes(staging, folders)})
        for path in staging.iterdir():
            path.replace(destination / path.name)


def import_snapshot(source, destination):
    """Import an explicitly validated human or synthetic cohort without rewriting it.

    The input manifest defines the complete public inputs. Reference answers and
    validation evidence stay outside that directory and the agent workspace.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    folders = ("environment", "database", "inputs", "scenarios.json")
    hashes = file_hashes(source, folders)
    if not hashes or hashes != manifest.get("files"):
        raise ValueError("Validated snapshot files do not match their manifest")
    if manifest.get("validation_evidence_is_agent_visible") is True:
        raise ValueError("Validation answers must be separate from agent inputs")
    rows = json.loads((source / "scenarios.json").read_text())
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Snapshot needs nonempty scenarios with unique IDs")
    public_inputs = sorted(key.removeprefix("inputs/") for key in hashes if key.startswith("inputs/"))
    saved = {"format_version": 1, "imported_snapshot": str(source),
             "request": {"asset_class": manifest.get("asset_class", "Validated scenarios"),
                         "generation_mode": "general-execution"},
             "shared_input_files": public_inputs, "files": hashes}
    if (destination / "snapshot.json").exists():
        if json.loads((destination / "snapshot.json").read_text()) != saved or file_hashes(destination, folders) != hashes:
            raise ValueError("Evaluation snapshot changed; choose a new directory")
        return
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Nonempty evaluation directory has no compatible snapshot")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=destination) as temporary:
        staging = Path(temporary)
        for folder in folders:
            src = source / folder
            if src.is_dir():
                shutil.copytree(src, staging / folder)
            else:
                shutil.copyfile(src, staging / folder)
        if file_hashes(staging, folders) != hashes or file_hashes(source, folders) != hashes:
            raise ValueError("Snapshot changed while importing")
        for item in staging.iterdir():
            item.replace(destination / item.name)
    private_json(destination / "snapshot.json", saved)


def compose(case, *args, **kwargs):
    return subprocess.run(["docker", "compose", "-f", str(case / "compose.json"), *args],
                          check=True, **kwargs)


def prepare_case(root, case, scenario):
    case.mkdir(parents=True, exist_ok=True)
    workspace = case / "workspace"
    if workspace.exists():
        raise ValueError("Case workspace already exists; archive the prior attempt before execution")
    workspace.mkdir()
    snapshot_path = root / "snapshot.json"
    metadata = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    public_inputs = sorted(set(metadata.get("shared_input_files", [])) |
                           set(scenario.get("execution", {}).get("input_files", [])))
    for relative in public_inputs:
        source = checked_file(root / "inputs", relative)
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    tools_workspace = case / "tools-workspace"
    shutil.copytree(root / "environment", tools_workspace, dirs_exist_ok=True)
    # Exercised outputs are solutions, never initial state for a fresh execution.
    for row in json.loads((root / "scenarios.json").read_text()):
        for output in row.get("execution", {}).get("output_files", []):
            path = (tools_workspace / output["path"]).resolve()
            if not path.is_relative_to(tools_workspace.resolve()):
                raise ValueError("Generated output escapes tool workspace")
            if path.is_file():
                path.unlink()
    (workspace / "artifacts").mkdir(exist_ok=True)
    config = case / "config"
    config.mkdir(exist_ok=True)
    private_json(config / "input-files.json", public_inputs)
    (config / "question.txt").write_text(scenario["text"])
    private_json(config / "mcp.json", {"mcpServers": {
        name: {"type": "http", "url": f"http://tools:{8100 + i}/mcp"}
        for i, name in enumerate(SERVERS)}})
    environment = {"COUCHDB_URL": "http://database:5984", "COUCHDB_USERNAME": "evaluation",
                   "COUCHDB_PASSWORD": secrets.token_hex(16), "PYTHONPATH": "/environment/src",
                   "TSFM_WORKDIR": "/workspace/artifacts", "PYTHONUNBUFFERED": "1"}
    name = "aob-eval-" + hashlib.sha256(str(case.resolve()).encode()).hexdigest()[:12]
    private_json(case / "compose.json", {
        "name": name, "services": {
            "database": {
                "image": "couchdb:3.5", "environment": {
                    "COUCHDB_USER": environment["COUCHDB_USERNAME"], "COUCHDB_PASSWORD": environment["COUCHDB_PASSWORD"]},
                "entrypoint": ["tini", "--", "sh", "-c", "printf '[couchdb]\\nsingle_node=true\\n' > /opt/couchdb/etc/local.d/evaluation.ini; exec /docker-entrypoint.sh /opt/couchdb/bin/couchdb"],
                "healthcheck": {"test": ["CMD", "curl", "-fsS", "http://localhost:5984/_up"],
                                "interval": "1s", "timeout": "2s", "retries": 40}},
            "tools": {
                "image": IMAGE, "working_dir": "/workspace", "environment": environment,
                "volumes": [f"{workspace}:/workspace", f"{tools_workspace}:/environment:ro",
                            f"{HERE}:/support:ro", f"{root / 'database'}:/snapshot:ro"],
                "ports": [f"127.0.0.1::{8100 + i}" for i in range(len(SERVERS))],
                "command": ["sh", "-c", "set -e; if [ -s /environment/requirements.txt ]; then uv pip install --system -r /environment/requirements.txt; fi; exec python /support/tools.py"],
                "init": True, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"]},
        }})


@contextmanager
def environment(root, case, scenario):
    prepare_case(root, case, scenario)
    try:
        compose(case, "down", "--volumes", "--remove-orphans", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        compose(case, "up", "-d", "--wait", "database", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        compose(case, "run", "--rm", "-T", "tools", "python", "/support/database.py", "restore", "/snapshot",
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        compose(case, "up", "-d", "tools", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        endpoints = {}
        for i, name in enumerate(SERVERS):
            port = compose(case, "port", "tools", str(8100 + i), capture_output=True, text=True).stdout.strip().rsplit(":", 1)[-1]
            endpoints[name] = {"type": "http", "url": f"http://127.0.0.1:{port}/mcp"}
        deadline = time.monotonic() + 180
        pending = set(endpoints)
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                try:
                    if requests.get(endpoints[name]["url"], timeout=1).status_code in {400, 405, 406}:
                        pending.remove(name)
                except requests.RequestException:
                    pass
            if pending:
                time.sleep(1)
        if pending:
            raise RuntimeError(f"MCP servers did not start: {sorted(pending)}")
        yield endpoints
    finally:
        try:
            with (case / "tools.log").open("w") as log:
                compose(case, "logs", "--no-color", "tools", stdout=log, stderr=subprocess.DEVNULL)
        finally:
            compose(case, "down", "--volumes", "--remove-orphans", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def artifact_inventory(case, scenario):
    """Attach bounded file evidence for judging; never treat a printed path as a file."""
    inputs = set(scenario.get("execution", {}).get("input_files", []))
    recorded_inputs = case / "config/input-files.json"
    if recorded_inputs.exists():
        inputs.update(json.loads(recorded_inputs.read_text()))
    artifacts = []
    root = case / "workspace"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        relative = str(path.relative_to(root))
        if relative in inputs:
            continue
        content = path.read_bytes()
        artifacts.append({"location": "workspace", "path": relative, "bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest(),
                          "preview": content[:1000].decode("utf-8", errors="replace")})
    return artifacts
