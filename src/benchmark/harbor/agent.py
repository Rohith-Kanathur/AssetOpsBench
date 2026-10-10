"""Harbor agent adapter around the existing isolated AssetOpsBench runners.

The adapter is trusted host orchestration, like Harbor's own Docker launcher.
Evaluated models still run with the original workspace/database isolation.
"""

import asyncio
import os
from pathlib import Path
import signal
import sys

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities

from .stage import cleanup, collect


class PipelineAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)

    def __init__(self, *args, spec_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_path = Path(spec_path).resolve()

    @staticmethod
    def name():
        return 'assetops-pipeline'

    def version(self):
        return '0.1.0'

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        process = None
        try:
            with (self.logs_dir / 'stage.log').open('w') as log:
                process = await asyncio.create_subprocess_exec(sys.executable, '-m',
                    'benchmark.harbor.stage', str(self.spec_path), stdout=log, stderr=log,
                    start_new_session=True)
                await process.wait()
        finally:
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
                except ProcessLookupError:
                    pass
            # Run synchronous Docker cleanup off the event loop.
            cleanup_error = None
            try:
                await asyncio.to_thread(cleanup, self.spec_path)
            except Exception as exc:
                cleanup_error = type(exc).__name__
            outcome = collect(self.spec_path, self.logs_dir)
            if cleanup_error:
                outcome['cleanup_error'] = cleanup_error
            context.metadata = outcome
            await environment.upload_file(self.logs_dir / 'outcome.json', '/workspace/outcome.json')
        if process.returncode:
            raise RuntimeError(f'Pipeline stage exited {process.returncode}; see agent/stage.log')
