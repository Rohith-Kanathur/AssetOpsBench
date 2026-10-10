"""Build a complete judge-only evidence copy without run identity metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .auth import private_json


VERSION = "blinded-full-evidence-v2"
SCENARIO_FIELDS = ("text", "characteristic_form", "expected_answer")
RESULT_FIELDS = ("status", "answer", "trajectory", "artifacts", "native_tool_artifacts",
                 "error", "timed_out", "tool_names_called")
IDENTITY_FIELDS = {"model", "model_id", "model_name", "requested_model", "response_model",
                   "provider", "runner", "harness"}
METADATA_FIELDS = IDENTITY_FIELDS | {
    "scenario_id", "run_id", "session_id", "trajectory_id", "case_dir",
    "source_ids", "origin", "cohort", "scenario_source", "authored_by",
    "positive", "negative", "missing_evidence", "stirrup_version",
}
# Tool payloads are evidence, not runner metadata. In particular, a forecasting
# tool's `model` argument must survive even though runner-level `model` is hidden.
PAYLOAD_FIELDS = {"input", "output", "arguments", "result", "content", "text", "message"}
MODEL_FAMILIES = {"gpt", "claude", "gemini", "deepseek", "qwen", "llama", "gemma", "glm", "kimi", "grok"}


def workspace_files(case: Path) -> list[Path]:
    root = case / "workspace"
    if root.is_symlink():
        raise ValueError("Judge evidence workspace cannot be a symlink")
    paths = sorted(root.rglob("*")) if root.exists() else []
    if any(path.is_symlink() for path in paths):
        raise ValueError("Judge evidence cannot follow workspace symlinks")
    if any(not path.is_file() and not path.is_dir() for path in paths):
        raise ValueError("Judge evidence requires regular files and directories")
    return [path for path in paths if path.is_file()]


def workspace_hashes(case: Path) -> dict[str, str]:
    return {str(path.relative_to(case / "workspace")): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in workspace_files(case)}


class Redactor:
    def __init__(self, case: Path, scenario: dict, execution: dict):
        self.identities = set()

        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in IDENTITY_FIELDS and isinstance(item, str) and item:
                        self.identities.add(item)
                    elif key not in PAYLOAD_FIELDS:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(execution)
        for value in list(self.identities):
            # Route prefixes, provider, model ID and filename-safe forms.
            self.identities.update(value.split("/"))
            self.identities.add(re.sub(r"[^A-Za-z0-9_.-]+", "-", value))
            name = value.rsplit("/", 1)[-1].lower()
            family = name.split("-", 1)[0]
            if family in MODEL_FAMILIES:
                # Self-descriptions can use only the family or omit a variant.
                parts = name.split("-")
                self.identities.update("-".join(parts[:i]) for i in range(1, len(parts)))
                if family == "gpt":
                    self.identities.add("ChatGPT")
        self.identities.discard("")
        self.pattern = (re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(
            re.escape(value) for value in sorted(self.identities, key=len, reverse=True)) +
            r")(?![A-Za-z0-9])", re.IGNORECASE) if self.identities else None)
        self.paths = {str(case.resolve()): "/evidence"}
        if isinstance(execution.get("case_dir"), str):
            self.paths[execution["case_dir"]] = "/evidence"
        # Do not replace ordinary words if a caller uses a directory such as
        # `case` or `human`. The complete absolute path is still neutralized.
        self.case_names = {case.name} if (any(char.isdigit() for char in case.name) or
                                         re.search(r"(?:human|synthetic)[_-]", case.name, re.IGNORECASE)) else set()
        sid = str(scenario.get("id", ""))
        if re.match(r"(?:human|synthetic)[_-]", sid, re.IGNORECASE):
            self.case_names.add(sid)

    def filename(self, name: str) -> str:
        name = self.pattern.sub("[MODEL]", name) if self.pattern else name
        return re.sub(r"(^|/)(?:human|synthetic)(?=[_.-])", r"\1cohort", name,
                      flags=re.IGNORECASE)

    def text(self, value: str) -> str:
        for original, replacement in sorted(self.paths.items(), key=lambda item: -len(item[0])):
            value = value.replace(original, replacement)
        for name in sorted(self.case_names, key=len, reverse=True):
            if name and not name.isdecimal():
                value = re.sub(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                               "case", value, flags=re.IGNORECASE)
        return self.pattern.sub("[MODEL]", value) if self.pattern else value

    def payload(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.payload(item) for item in value]
        if isinstance(value, dict):
            return {self.text(key): self.payload(item) for key, item in value.items()}
        return value

    def trajectory(self, value):
        if isinstance(value, list):
            return [self.trajectory(item) for item in value]
        if not isinstance(value, dict):
            return self.payload(value)
        return {key: self.payload(item) if key in PAYLOAD_FIELDS else self.trajectory(item)
                for key, item in value.items() if key not in METADATA_FIELDS and not
                (key == "source" and isinstance(item, str) and item.lower() in {"human", "synthetic"})}


def prepare_view(case: Path, destination: Path, scenario: dict, execution: dict) -> tuple[dict, dict]:
    """Retain all turns and artifact files; only the separate copy is transformed."""
    case = Path(case)
    redactor = Redactor(case, scenario, execution)
    files = workspace_files(case)
    paths = {}
    for source in files:
        relative = str(source.relative_to(case / "workspace"))
        blinded = redactor.filename(relative)
        if blinded in paths.values():
            raise ValueError("Blinded artifact paths collide")
        paths[relative] = blinded
    redactor.paths.update({original: blinded for original, blinded in paths.items() if original != blinded})
    destination.mkdir(parents=True)
    (destination / "workspace").mkdir()
    manifest = {}
    for source in files:
        relative = str(source.relative_to(case / "workspace"))
        target = destination / "workspace" / paths[relative]
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source.read_bytes()
        try:
            value = redactor.text(original.decode("utf-8")).encode("utf-8")
        except UnicodeDecodeError:
            # Do not corrupt binary scientific outputs to hide identity strings.
            # Short family aliases can occur by chance in compressed bytes.
            identifiers = {name for name in redactor.identities if len(name) >= 6} | set(redactor.paths) | {
                name for name in redactor.case_names if name and not name.isdecimal()}
            if any(identity.lower().encode("utf-8") in original.lower() for identity in identifiers):
                raise ValueError("A binary artifact contains an identity; a reviewed blinded copy is required")
            value = original
        target.write_bytes(value)
        target.chmod(0o600)
        manifest[relative] = {"path": paths[relative], "original_sha256": hashlib.sha256(original).hexdigest(),
                              "blinded_sha256": hashlib.sha256(value).hexdigest()}
    visible_scenario = {"id": "case", **{key: redactor.payload(scenario[key])
                        for key in SCENARIO_FIELDS if key in scenario}}
    visible_execution = {"scenario_id": "case", **{key: redactor.trajectory(execution[key])
                         if key == "trajectory" else redactor.payload(execution[key])
                         for key in RESULT_FIELDS if key in execution}}
    # The judge's inventory describes the copied files, not the originals.
    for original, visible in zip(execution.get("artifacts", []), visible_execution.get("artifacts", [])):
        if isinstance(original, dict) and original.get("location") == "workspace" and original.get("path") in manifest:
            entry = manifest[original["path"]]
            content = (destination / "workspace" / entry["path"]).read_bytes()
            visible.update(path=entry["path"], bytes=len(content), sha256=entry["blinded_sha256"])
            if "preview" in original:
                visible["preview"] = content[:1000].decode("utf-8", errors="replace")
    private_json(destination / "scenario.json", visible_scenario)
    private_json(destination / "result.json", visible_execution)
    private_json(destination.parent / "blinding.json", {
        "version": VERSION, "original_case": str(case.resolve()),
        "scenario_id": scenario.get("id"), "model": execution.get("model"),
        "runner": execution.get("runner", execution.get("harness")),
        "files": manifest,
        "note": "Original logs and artifacts are unchanged. This mapping is never mounted for the judge.",
    })
    return visible_scenario, visible_execution
