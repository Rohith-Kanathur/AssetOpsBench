import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scenarios.generation.workspace import PACKAGES, audit_baseline, prepare


def git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "workspace-test@example.invalid")
    git(repo, "config", "user.name", "Workspace test")
    git(repo, "config", "commit.gpgsign", "false")
    files = {
        "src/servers/fmsr/main.py": b"def predict_health_index(): pass\n",
        "src/servers/fmsr/artifacts/Transformer_model.pkl": b"\x80\x04model\x00\xff",
        "src/servers/fmsr/tests/test_tools.py": b"def test_tool(): pass\n",
        "src/mcphub/tools.sh": b"#!/bin/sh\nexit 0\n",
        "src/llm/prompts.txt": b"Existing asset tools and prompts\n",
        "src/couchdb/data/Transformer.json": b'{"asset_class": "Transformer"}\n',
        "src/couchdb/scenarios_data/default/manifest.json": b'{"scenarios": []}\n',
        "src/couchdb/loader.py": b"def load(): pass\n",
        "src/couchdb/data/export.txt": b"$Format:%H$\n",
        "src/couchdb/.gitattributes": b"data/Transformer.json export-ignore\ndata/export.txt export-subst\n",
        "src/scenarios/failure_mapping/example.json": b"{}\n",
        ".env": b"PRIVATE_TEST_VALUE=example\n",
        "docs/reference.md": b"Repository documentation\n",
    }
    for name, data in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (repo / "src/mcphub/tools.sh").chmod(0o755)
    git(repo, "add", "--force", ".")
    git(repo, "commit", "-qm", "Environment source and fixtures")
    return repo, files


def test_snapshot_retains_committed_source_data_models_and_tests(repository, tmp_path):
    repo, files = repository
    baseline = git(repo, "rev-parse", "HEAD")
    (repo / "src/servers/fmsr/main.py").write_text("Uncommitted source change\n")
    (repo / "src/couchdb/data/private-local.json").write_text("{}\n")
    before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")

    target = tmp_path / "generation"
    audit = prepare(repo, target)
    workspace = target / "workspace"
    expected = {name: data for name, data in files.items() if name.startswith(PACKAGES)}
    assert audit["baseline"] == baseline
    assert audit["source_ref"] == "HEAD"
    assert audit["packages"] == list(PACKAGES)
    assert audit["files"] == {
        name: hashlib.sha256(data).hexdigest() for name, data in expected.items()
    }
    assert json.loads((target / "baseline.json").read_text()) == audit
    for name, data in expected.items():
        assert (workspace / name).read_bytes() == data
    assert (workspace / "src/mcphub/tools.sh").stat().st_mode & 0o777 == 0o755
    assert (workspace / "output").is_dir()
    assert (workspace / "data").is_dir()
    assert not (workspace / ".git").exists()
    assert not (workspace / ".env").exists()
    assert audit_baseline(target) == []
    assert git(repo, "rev-parse", "HEAD") == baseline
    assert git(repo, "status", "--porcelain=v1", "--untracked-files=all") == before
    assert (repo / "src/servers/fmsr/main.py").read_text() == "Uncommitted source change\n"
    with pytest.raises(FileExistsError):
        prepare(repo, target)


def test_selected_ref_is_exported_without_switching_checkout(repository, tmp_path):
    repo, files = repository
    selected = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "environment-baseline", selected)
    path = repo / "src/servers/fmsr/main.py"
    path.write_text("New committed source\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "New environment source")
    head = git(repo, "rev-parse", "HEAD")

    audit = prepare(repo, tmp_path / "selected", ref="environment-baseline")
    assert audit["baseline"] == selected
    assert audit["source_ref"] == "environment-baseline"
    assert (tmp_path / "selected/workspace/src/servers/fmsr/main.py").read_bytes() == files[
        "src/servers/fmsr/main.py"
    ]
    assert git(repo, "rev-parse", "HEAD") == head
    assert path.read_text() == "New committed source\n"
    assert prepare(repo, tmp_path / "head")["baseline"] == head
    assert (tmp_path / "head/workspace/src/servers/fmsr/main.py").read_text() == "New committed source\n"


@pytest.mark.parametrize("change", ["contents", "remove", "mode", "symlink"])
def test_audit_detects_changed_baseline_files(repository, tmp_path, change):
    repo, _ = repository
    target = tmp_path / "generation"
    prepare(repo, target)
    name = "src/servers/fmsr/main.py"
    path = target / "workspace" / name
    if change == "contents":
        path.write_text("changed")
    elif change == "mode":
        path.chmod(0o755)
    else:
        path.unlink()
        if change == "symlink":
            path.symlink_to(repo / name)
    assert audit_baseline(target) == [f"Changed baseline file: {name}"]


def test_audit_detects_added_files_and_git_history(repository, tmp_path):
    repo, _ = repository
    target = tmp_path / "generation"
    prepare(repo, target)
    workspace = target / "workspace"
    (workspace / "extra.txt").write_text("unexpected")
    (workspace / ".git").mkdir()
    assert audit_baseline(target) == ["Unexpected baseline file: extra.txt", "Git history present"]


def test_symlink_source_member_is_rejected_before_writing(repository, tmp_path):
    repo, _ = repository
    (repo / "src/couchdb/link").symlink_to("../../.env")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Unsupported source symlink")
    target = tmp_path / "generation"
    with pytest.raises(ValueError, match="Unsupported source member: src/couchdb/link"):
        prepare(repo, target)
    assert not target.exists()


def test_submodule_source_member_is_rejected_before_writing(repository, tmp_path):
    repo, _ = repository
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{git(repo, 'rev-parse', 'HEAD')},src/servers/external")
    git(repo, "commit", "-qm", "Unsupported submodule")
    target = tmp_path / "generation"
    with pytest.raises(ValueError, match="Unsupported source member: src/servers/external"):
        prepare(repo, target)
    assert not target.exists()


def test_private_environment_in_package_is_rejected(repository, tmp_path):
    repo, _ = repository
    (repo / "src/llm/.env").write_text("PRIVATE_TEST_VALUE=example\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Unsupported private environment file")
    target = tmp_path / "generation"
    with pytest.raises(ValueError, match="Unsupported source member: src/llm/.env"):
        prepare(repo, target)
    assert not target.exists()


def test_invalid_ref_does_not_create_destination(repository, tmp_path):
    repo, _ = repository
    target = tmp_path / "generation"
    with pytest.raises(subprocess.CalledProcessError):
        prepare(repo, target, ref="missing-reference")
    assert not target.exists()


def test_empty_package_export_does_not_create_destination(repository, tmp_path):
    repo, _ = repository
    git(repo, "rm", "-rq", *PACKAGES)
    git(repo, "commit", "-qm", "No environment packages")
    target = tmp_path / "generation"
    with pytest.raises(ValueError, match="No committed environment files"):
        prepare(repo, target)
    assert not target.exists()
