"""Prepare and audit a committed snapshot of the environment packages."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess


PACKAGES = ("src/servers", "src/mcphub", "src/llm", "src/couchdb")


def prepare(repository: Path, destination: Path, ref: str = "HEAD") -> dict:
    """Export package files from a commit without changing the checkout."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Choose a new generation directory: {destination}")
    baseline = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=repository,
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", "--full-tree", baseline, "--", *PACKAGES],
        cwd=repository,
    )
    entries = []
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        mode, kind, object_id = metadata.split()
        name = raw_name.decode("utf-8", errors="surrogateescape")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not any(path.is_relative_to(package) for package in PACKAGES)
            or ".git" in path.parts
            or ".env" in path.parts
            or kind != b"blob"
            or mode not in {b"100644", b"100755"}
        ):
            raise ValueError(f"Unsupported source member: {name}")
        entries.append((name, object_id, 0o755 if mode == b"100755" else 0o644))
    if not entries:
        raise ValueError("No committed environment files in the selected repository revision")

    # Read blobs directly: archive attributes can omit or rewrite committed files.
    blobs = io.BytesIO(subprocess.check_output(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        input=b"".join(object_id + b"\n" for _, object_id, _ in entries),
    ))
    workspace = destination / "workspace"
    workspace.mkdir(parents=True)
    copied, modes = {}, {}
    for name, object_id, mode in entries:
        header = blobs.readline().split()
        if len(header) != 3 or header[:2] != [object_id, b"blob"]:
            raise ValueError(f"Unable to read source member: {name}")
        size = int(header[2])
        data = blobs.read(size)
        if len(data) != size or blobs.read(1) != b"\n":
            raise ValueError(f"Incomplete source member: {name}")
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
        copied[name] = hashlib.sha256(data).hexdigest()
        modes[name] = mode
    audit = {
        "baseline": baseline,
        "source_ref": ref,
        "packages": list(PACKAGES),
        "files": copied,
        "modes": modes,
    }
    (destination / "baseline.json").write_text(json.dumps(audit, indent=2) + "\n")
    (workspace / "output").mkdir()
    (workspace / "data").mkdir()
    return audit


def audit_baseline(destination: Path) -> list[str]:
    """Check the exported files, modes and absence of unexpected files."""
    audit = json.loads((destination / "baseline.json").read_text())
    workspace = destination / "workspace"
    if workspace.is_symlink():
        return ["Workspace is a symlink"]
    errors = []
    for name, digest in audit["files"].items():
        path = workspace / name
        if (
            path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            or path.stat().st_mode & 0o777 != audit.get("modes", {}).get(name, 0o644)
        ):
            errors.append(f"Changed baseline file: {name}")
    actual = {
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    for name in sorted(actual - set(audit["files"])):
        errors.append(f"Unexpected baseline file: {name}")
    if (workspace / ".git").exists() or (workspace / ".git").is_symlink():
        errors.append("Git history present")
    return errors
