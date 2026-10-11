"""A fresh Codex subscription session with a read-only blinded evidence mount."""
from contextlib import suppress
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4

from agent.codex_accounts import (MODEL, REASONING, TIER, identity, save,
                                  credential_snapshot, merge_credentials)
from agent.coding_agent.trajectory import parse
from llm.base import LLMBackend


class CodexJudge(LLMBackend):
    def __init__(self, audit: Path, account: dict, model=MODEL, timeout=600,
                 reasoning=REASONING, tier=TIER):
        self.audit, self.account = Path(audit), account
        self._model_id, self.timeout = model, timeout
        self.reasoning, self.tier = reasoning, tier

    def generate(self, prompt, temperature=0):
        from .judge import SYSTEM, IMAGE, CRITERIA
        audit = self.audit
        audit.mkdir(parents=True, exist_ok=True)
        (audit / 'prompt.txt').write_text(prompt)
        name = 'assetops-judge-' + uuid4().hex[:12]
        save(audit / 'container.json', {'name': name})
        with TemporaryDirectory(prefix='assetops-judge-') as temporary:
            root = Path(temporary)
            auth = root / 'auth'
            auth.mkdir(mode=0o700)
            original_auth = credential_snapshot(self.account)
            save(auth / 'auth.json', original_auth)
            evidence = (audit / 'evidence').resolve()
            schema = {'type': 'object', 'properties': {key: {'type': 'boolean'} for key in CRITERIA},
                      'required': [*CRITERIA, 'suggestions'], 'additionalProperties': False}
            schema['properties']['suggestions'] = {'type': 'string'}
            (evidence / 'schema.json').write_text(json.dumps(schema))
            # Keep case/model names out of mount metadata too. Hard links reuse
            # the blinded file bytes while retaining a neutral mount source.
            evidence_home = root / 'evidence'
            def link_or_copy(source, destination):
                try:
                    return os.link(source, destination)
                except OSError:
                    return shutil.copy2(source, destination)
            shutil.copytree(evidence, evidence_home, copy_function=link_or_copy)
            command = ['docker', 'run', '--rm', '--init', '-i', '--name', name,
                       '--user', f'{os.getuid()}:{os.getgid()}', '--read-only',
                       '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                       '--tmpfs', '/tmp', '--workdir', '/evidence',
                       '--mount', f'type=bind,src={auth},dst=/auth',
                       '--env', 'HOME=/tmp', '--env', 'CODEX_HOME=/auth',
                       '--mount', f'type=bind,src={evidence_home},dst=/evidence,readonly',
                       '--entrypoint', 'codex', IMAGE, 'exec', '--ignore-user-config',
                       '--ephemeral', '--skip-git-repo-check', '--json', '--color', 'never',
                       # Docker enforces the read-only filesystem and evidence mount.
                       # A nested bwrap sandbox cannot create namespaces on Docker Desktop.
                       '--sandbox', 'danger-full-access', '-c', 'approval_policy="never"',
                       '-c', 'cli_auth_credentials_store="file"', '-c', 'forced_login_method="chatgpt"',
                       '-c', 'web_search="disabled"', '-c', 'agents.enabled=false',
                       '-c', f'model_reasoning_effort={json.dumps(self.reasoning)}',
                       '-c', f'service_tier={json.dumps(self.tier)}',
                       '--model', self._model_id, '--output-schema', '/evidence/schema.json', '-']
            try:
                with (audit / 'events.jsonl').open('w') as output, (audit / 'stderr.log').open('w') as errors:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                               stderr=errors, text=True, start_new_session=True)
                    try:
                        process.communicate(SYSTEM + '\n\n' + prompt, timeout=self.timeout)
                    except BaseException:
                        subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                        raise
            finally:
                # Refresh tokens belong to the same leased subscription and survive
                # this isolated session. No credentials go into evidence or ATIF.
                if identity(auth / 'auth.json')[0] != self.account['email']:
                    raise RuntimeError('Judge login identity changed unexpectedly')
                merge_credentials(self.account, original_auth,
                                  json.loads((auth / 'auth.json').read_text()))
        parsed = parse((audit / 'events.jsonl').read_text(), 'codex')
        parsed.update(runner='codex', model=self._model_id,
                      settings={'reasoning_effort': self.reasoning, 'service_tier': self.tier})
        save(audit / 'result.json', parsed)
        if process.returncode or not parsed['completed'] or not parsed['answer'].strip():
            detail = str(parsed.get('error') or (audit / 'stderr.log').read_text()[-1000:])
            raise RuntimeError('Codex judge failed: ' + detail[:1000])
        return parsed['answer']
