"""Bound new admissions by measured host and Docker memory; never stop workers."""
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import psutil

from .auth import private_json


def memory_snapshot():
    host = psutil.virtual_memory().available / 1048576
    pressure = 1
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                capture_output=True, text=True, timeout=5)
        pressure = int(result.stdout.strip()) if result.returncode == 0 else None
    # Read the existing service rather than starting a container for each sample.
    container = os.environ.get("ASSETOPS_MEMORY_CONTAINER")
    if not container:
        found = subprocess.run(["docker", "ps", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, timeout=5)
        container = next((line.split()[0] for line in found.stdout.splitlines()
            if len(line.split()) == 2 and line.split()[1].startswith(
                ('assetops-judge-', 'aob-shared-'))), None) if found.returncode == 0 else None
    command = (["docker", "exec", container, "cat", "/proc/meminfo"] if container else
        ["docker", "run", "--rm", "--network", "none", "--read-only",
         "--cap-drop", "ALL", "--entrypoint", "cat", "couchdb:3.5", "/proc/meminfo"])
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("Docker memory is unavailable")
    values = {line.split(':')[0]: int(line.split()[1]) / 1024
              for line in result.stdout.splitlines() if ':' in line}
    # Attribute memory already committed by newly started judge containers.
    # If Docker's stats API is unavailable, retain the full startup reserve.
    judges = {}
    try:
        import docker
        client = docker.from_env(timeout=5)
        try:
            for worker in client.containers.list(filters={
                    'label': f'assetops.admission-owner={os.getpid()}'}):
                stats = client.api.stats(worker.id, stream=False, one_shot=True)['memory_stats']
                judges[worker.id] = max(0, stats['usage'] -
                    stats.get('stats', {}).get('inactive_file', 0)) / 1048576
        finally:
            client.close()
    except Exception:
        judges = None
    return {"host_available_mib": host, "guest_available_mib": values['MemAvailable'],
            "host_pressure": pressure, "swapout": psutil.swap_memory().sout,
            "judge_memory_mib": judges}


class Admission:
    def __init__(self, sample=memory_snapshot, clock=time.monotonic, state_path=None):
        self.sample, self.clock = sample, clock
        self.guard = threading.Lock()
        self.last_sample = float('-inf')
        self.snapshot = None
        self.reservations = []
        self.seen_workers = None
        self.reason = 'not_sampled'
        self.high_paging_samples = 0
        self.state_path = Path(state_path) if state_path else None
        self.last_write = float('-inf')

    def _record(self, now, kind, allowed, needed):
        if self.state_path and now - self.last_write >= 1:
            private_json(self.state_path, {'at': time.time(), 'kind': kind, 'allowed': allowed,
                'reason': self.reason, 'snapshot': self.snapshot,
                'startup_reserve_mib': self._reserved(),
                'startup_budget_mib': sum(r['size'] for r in self.reservations),
                'next_worker_mib': needed, 'host_floor_mib': 2048 if self.snapshot and
                    self.snapshot['host_pressure'] == 2 else 1536, 'guest_floor_mib': 1536})
            self.last_write = now
        return allowed

    def _observe_workers(self):
        workers = self.snapshot.get('judge_memory_mib')
        if workers is None:
            return
        if self.seen_workers is not None:
            pending = [r for r in self.reservations if r['kind'] == 'judge' and not r['worker']]
            for worker, reservation in zip(sorted(set(workers) - self.seen_workers), pending):
                reservation['worker'] = worker
        self.seen_workers = (self.seen_workers or set()) | set(workers)

    def _reserved(self):
        workers = (self.snapshot or {}).get('judge_memory_mib')
        return sum(max(0, r['size'] - (workers.get(r['worker'], r['size'])
                       if workers is not None and r['worker'] else 0))
                   for r in self.reservations)

    def allow(self, kind):
        """Account for recently admitted workers before their memory appears in stats."""
        with self.guard:
            now = self.clock()
            needed = {'judge': 640, 'execution': 768}[kind]
            if now - self.last_sample >= 5:
                previous = self.snapshot
                try:
                    self.snapshot = self.sample()
                    elapsed = max(now - self.last_sample, .001)
                    rate = (max(0, self.snapshot['swapout'] - previous['swapout']) /
                            1048576 / elapsed) if previous else 0.
                    self.high_paging_samples = self.high_paging_samples + 1 if rate >= 4 else 0
                    self.snapshot['swapout_mib_per_second'] = rate
                    # Tiny background paging is normal on macOS. Stop admissions
                    # for a large burst or sustained paging, not a single page.
                    self.snapshot['paging'] = rate >= 32 or self.high_paging_samples >= 2
                    if self.snapshot['host_pressure'] == 2 and rate >= 1:
                        self.snapshot['paging'] = True
                    self._observe_workers()
                except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError):
                    self.snapshot = None
                self.last_sample = now
            self.reservations = [r for r in self.reservations if now - r['at'] < 15]
            if self.snapshot is None:
                self.reason = 'memory_unavailable'
                return self._record(now, kind, False, needed)
            if self.snapshot['host_pressure'] not in (1, 2) or self.snapshot['paging']:
                self.reason = 'host_pressure_or_paging'
                return self._record(now, kind, False, needed)
            reserve = self._reserved()
            host_floor = 2048 if self.snapshot['host_pressure'] == 2 else 1536
            if (self.snapshot['host_available_mib'] - reserve - needed < host_floor or
                    self.snapshot['guest_available_mib'] - reserve - needed < 1536):
                self.reason = 'reserved_headroom'
                return self._record(now, kind, False, needed)
            self.reservations.append({'at': now, 'size': needed, 'kind': kind, 'worker': None})
            self.reason = 'admitted'
            return self._record(now, kind, True, needed)


shared_admission = Admission(state_path=os.environ.get('ASSETOPS_ADMISSION_STATE'))
