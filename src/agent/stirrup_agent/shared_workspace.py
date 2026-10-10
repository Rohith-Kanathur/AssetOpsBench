"""Use the same per-case directory for Stirrup code and domain MCP tools."""

import json
import os
from pathlib import Path

from anyio import to_thread
import docker
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider


class SharedWorkspaceDockerProvider(DockerCodeExecToolProvider):
    """Preserve case inputs and outputs while removing only the code container."""

    def __init__(self, image: str, workspace: Path, *, container_record: Path | None = None):
        super().__init__(image, is_dockerfile=False)
        self.workspace = Path(workspace).resolve()
        self.container_record = container_record

    async def __aenter__(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._temp_dir = self.workspace
        self._client = await to_thread.run_sync(docker.from_env)
        try:
            image = await self._prepare_image()
            self._container = await to_thread.run_sync(lambda: self._client.containers.run(
                image, command="tail -f /dev/null", detach=True,
                volumes={str(self.workspace): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace", remove=False,
                user=f"{os.getuid()}:{os.getgid()}", environment={"HOME": "/tmp"},
                cap_drop=["ALL"], security_opt=["no-new-privileges:true"],
                labels={"assetops.role": "stirrup-code"},
            ))
            if self.container_record is not None:
                self.container_record.write_text(json.dumps({"id": self._container.id}) + "\n")
            return self.get_code_exec_tool()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._container is not None:
                await self._fix_file_ownership()
        finally:
            # The superclass removes its temporary directory. This directory is
            # instead the persistent case evidence, shared with the MCP servers.
            self._temp_dir = None
            await super().__aexit__(exc_type, exc_val, exc_tb)
