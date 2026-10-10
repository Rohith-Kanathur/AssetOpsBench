"""Load only database records and public inputs from a prepared snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Seed path escapes its directory: {name}")
    path = root / relative
    if any((root / Path(*relative.parts[:i])).is_symlink()
           for i in range(1, len(relative.parts) + 1)) or not path.is_file():
        raise ValueError(f"Seed input is not a regular file: {name}")
    return path


def describe(source):
    """Verify data hashes without reading or copying questions and answers."""
    source = Path(source).expanduser().resolve()
    manifest = json.loads(checked_file(source, "manifest.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Seed needs a manifest.json with file hashes")
    if manifest.get("validation_evidence_is_agent_visible") is True:
        raise ValueError("Seed inputs must exclude validation answers")
    expected = {name: value for name, value in manifest["files"].items()
                if name.startswith(("database/", "inputs/"))}
    if not any(name.startswith("database/") for name in expected):
        raise ValueError("Seed has no database exports")
    actual = {}
    for folder in ("database", "inputs"):
        base = source / folder
        if base.is_symlink() or not base.is_dir():
            raise ValueError(f"Seed needs a regular {folder}/ directory")
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Seed contains a symlink: {path.relative_to(source)}")
            if path.is_file():
                name = str(path.relative_to(source))
                if folder == "database":
                    if path.parent != base or not re.fullmatch(r"[a-z][a-z0-9_$()+-]*\.json", path.name):
                        raise ValueError(f"Invalid seed database export: {name}")
                else:
                    relative = path.relative_to(base)
                    if (relative.parts[0] in {"src", "output", "scripts", "references", "auth"}
                            or any(part.startswith(".") for part in relative.parts)
                            or relative.name in {"request.json", "profile.md", "generate.md", "seed-manifest.json"}
                            or relative.is_relative_to("data/seed-database")):
                        raise ValueError(f"Seed input uses a reserved path: {name}")
                actual[name] = digest(path)
    if actual != expected:
        raise ValueError("Seed database/input files do not match their manifest")
    identity = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    return {"sha256": identity, "files": actual}


def prepare(source, destination, description):
    """Copy verified data into a new workspace, keeping it independent of source."""
    source = Path(source).expanduser().resolve()
    workspace = destination / "workspace"
    copied = {}
    for name, expected in description["files"].items():
        relative = ("data/seed-database/" + name.removeprefix("database/")
                    if name.startswith("database/") else name.removeprefix("inputs/"))
        target = workspace / relative
        if target.exists():
            raise ValueError(f"Seed would overwrite an existing workspace file: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked_file(source, name), target)
        if digest(target) != expected:
            raise ValueError(f"Seed changed while copying: {name}")
        copied[relative] = expected
    # This data-only manifest provides traceable roots for the generator's sources.
    manifest = workspace / "data/seed-manifest.json"
    manifest.write_text(json.dumps({"sha256": description["sha256"], "files": copied}, indent=2) + "\n")
    copied[str(manifest.relative_to(workspace))] = digest(manifest)
    (destination / "seed.json").write_text(json.dumps({"sha256": description["sha256"], "files": copied}, indent=2) + "\n")


def verify(destination):
    saved = json.loads((destination / "seed.json").read_text())
    for name, expected in saved["files"].items():
        if digest(checked_file(destination / "workspace", name)) != expected:
            raise ValueError(f"Prepared seed changed: {name}")
    return saved


def restore(directory):
    """Restore the private generation database; retries replace partial imports."""
    import requests

    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("Seed has no database exports")
    session = requests.Session()
    session.auth = (os.environ["COUCHDB_USERNAME"], os.environ["COUCHDB_PASSWORD"])
    base = os.environ["COUCHDB_URL"].rstrip("/")

    def request(method, path, **kwargs):
        response = session.request(method, base + path, timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()

    for name in request("GET", "/_all_dbs"):
        if not name.startswith("_"):
            request("DELETE", "/" + name)
    for path in paths:
        docs = json.loads(path.read_text())
        if not isinstance(docs, list) or any(not isinstance(doc, dict) for doc in docs):
            raise ValueError(f"Seed database must contain a list of documents: {path.name}")
        request("PUT", "/" + path.stem)
        for start in range(0, len(docs), 500):
            results = request("POST", f"/{path.stem}/_bulk_docs", json={"docs": docs[start:start + 500]})
            if any("error" in result for result in results):
                raise ValueError(f"Failed to restore seed database: {path.stem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore",))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    restore(args.directory)
