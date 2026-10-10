"""Environment preparation policy, independent of evaluation capabilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

POLICIES = ("extend", "existing")
DEFAULT_POLICY = "extend"
BASELINE_ENV = "SCENARIO_ENVIRONMENT_BASELINE"


def policy(request):
    value = request.get("environment_policy", DEFAULT_POLICY)
    if value not in POLICIES:
        raise ValueError("Request: invalid environment_policy")
    return value


def write_guidance(workspace: Path):
    selected = policy(json.loads((workspace / "request.json").read_text()))
    guidance = (Path(__file__).parent / "prompts" / f"environment-{selected}.md").read_text()
    path = workspace / "profile.md"
    path.write_text(path.read_text().replace("{{environment_guidance}}", guidance))
    with (workspace / "generate.md").open("a") as handle:
        handle.write(f"\nEnvironment policy: `{selected}`. Apply its preparation and data rules in `profile.md`.\n")


def load_baseline():
    path = os.environ.get(BASELINE_ENV)
    if not path:
        raise ValueError("Existing environment requires the harness's protected baseline")
    return json.loads(Path(path).read_text())


def audit_source(workspace, baseline):
    from .workspace import PACKAGES
    from .seed import digest as file_digest

    errors = []
    for name, digest in baseline["files"].items():
        path = workspace / name
        if path.is_symlink() or not path.is_file() or file_digest(path) != digest:
            errors.append(f"Existing environment source changed: {name}")
    for package in PACKAGES:
        for path in (workspace / package).rglob("*"):
            if any(p in {"__pycache__", ".pytest_cache"} for p in path.parts):
                continue
            name = str(path.relative_to(workspace))
            if (path.is_file() or path.is_symlink()) and name not in baseline["files"]:
                errors.append(f"Existing environment source added: {name}")
    return errors


def database_state():
    """Hash logical input documents; credentials and document bodies stay private."""
    import requests
    from servers.tsfm.core.tasks import TASKS

    outputs = {"tsfm_runs"} | {task.result_collection for task in TASKS.values()}
    url = os.environ["COUCHDB_URL"].rstrip("/")
    auth = (os.environ["COUCHDB_USERNAME"], os.environ["COUCHDB_PASSWORD"])

    def get(path, **params):
        response = requests.get(url + path, auth=auth, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    state = {}
    for name in sorted(get("/_all_dbs")):
        if name.startswith("_") or name in outputs:
            continue
        rows = get(f"/{name}/_all_docs", include_docs="true")["rows"]
        docs = [{k: v for k, v in row["doc"].items() if k != "_rev"}
                for row in rows if "doc" in row and not row["id"].startswith("_design/")]
        digest = hashlib.sha256(json.dumps(docs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        state[name] = {"sha256": digest, "records": len(docs)}
    return state


def audit_database(baseline, current):
    expected = baseline["databases"]
    return [f"Existing environment input database changed: {name}"
            for name in sorted(expected.keys() | current.keys()) if expected.get(name) != current.get(name)]


if __name__ == "__main__":
    print(json.dumps(database_state(), indent=2))
