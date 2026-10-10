"""Generate genuine Harbor task packages with independent completion verifiers."""

from pathlib import Path

from harbor.models.task.config import TaskConfig
import tomllib

from .trajectory import write

VERIFIER = '''#!/bin/sh
set -eu
mkdir -p /logs/verifier
python3 - <<'PYVERIFY'
import json
from pathlib import Path
try:
    outcome = json.loads(Path('/workspace/outcome.json').read_text())
    completed = outcome.get('completed') is True
except (OSError, ValueError):
    outcome, completed = {}, False
reward = {'stage_completed': float(completed)}
if outcome.get('stage') == 'judging' and type(outcome.get('benchmark_pass')) is bool:
    reward['benchmark_pass'] = float(outcome['benchmark_pass'])
Path('/logs/verifier/reward.json').write_text(json.dumps(reward))
PYVERIFY
'''


def create_task(root, name, spec):
    task = Path(root) / 'tasks' / name
    if task.exists():
        raise ValueError(f'Task already exists: {task}')
    (task / 'environment').mkdir(parents=True)
    (task / 'tests').mkdir()
    timeout = spec['timeout']
    (task / 'task.toml').write_text(f'''schema_version = "1.4"
[task]
name = "assetops/{name}"
version = "1.0.0"
authors = []
keywords = ["assetops", "agents", "reproducibility"]
[metadata]
category = "pipeline"
[agent]
timeout_sec = {timeout}
[verifier]
timeout_sec = 30
[environment]
build_timeout_sec = 180
cpus = 1
memory_mb = 512
storage_mb = 1024
network_mode = "public"
''')
    TaskConfig.model_validate(tomllib.loads((task / 'task.toml').read_text()))
    (task / 'instruction.md').write_text(
        f"Run the AssetOpsBench {spec['stage']} stage declared in stage.json.\n\n"
        "Use the trusted benchmark.harbor.agent:PipelineAgent adapter. It launches the existing "
        "isolated runner on the host and saves the actual native logs plus ATIF trajectory. "
        "The Harbor verifier checks stage completion; the judging task separately reports "
        "benchmark_pass from the unchanged six-criterion rubric.\n")
    (task / 'environment/Dockerfile').write_text('FROM python:3.12-slim\nWORKDIR /workspace\n')
    verifier = task / 'tests/test.sh'
    verifier.write_text(VERIFIER)
    verifier.chmod(0o755)
    write(task / 'stage.json', spec)
    return task
