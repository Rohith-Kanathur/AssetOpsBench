"""Build frozen environment dependencies once, not in every running case."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def prepared_image(root, base):
    requirements = Path(root) / "environment/requirements.txt"
    if not requirements.exists() or not requirements.read_bytes().strip():
        return base
    identity = subprocess.run(["docker", "image", "inspect", base, "--format", "{{.Id}}"],
                              check=True, capture_output=True, text=True).stdout.strip()
    content = requirements.read_bytes()
    digest = hashlib.sha256(identity.encode() + b"\0" + content).hexdigest()
    target = "assetops-evaluation-deps:" + digest[:24]
    cache = Path.home() / ".cache/assetopsbench/runtime-images"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (cache / (digest + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        found = subprocess.run(["docker", "image", "inspect", target, "--format", "{{json .Config.Labels}}"],
                               capture_output=True, text=True)
        if not found.returncode:
            labels = json.loads(found.stdout) or {}
            if labels.get("assetops.dependencies") != digest:
                raise RuntimeError("Cached dependency image identity mismatch")
            return target
        with tempfile.TemporaryDirectory(prefix="assetops-image-") as temporary:
            context = Path(temporary)
            (context / "requirements.txt").write_bytes(content)
            (context / "Dockerfile").write_text(
                "ARG BASE\nFROM ${BASE}\nCOPY requirements.txt /opt/assetops-requirements.txt\n"
                "RUN uv pip install --system -r /opt/assetops-requirements.txt\n"
                f"LABEL assetops.dependencies={digest}\n")
            with (cache / (digest + ".build.log")).open("w") as log:
                result = subprocess.run(["docker", "build", "--build-arg", f"BASE={base}",
                    "--tag", target, str(context)], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Dependency image build failed; see {cache / (digest + '.build.log')}")
        return target
