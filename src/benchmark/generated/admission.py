"""Bound new admissions by measured host and Docker memory; never stop workers."""
import platform
import subprocess
import threading
import time

import psutil


def memory_snapshot():
    host = psutil.virtual_memory().available / 1048576
    pressure = 1
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                capture_output=True, text=True, timeout=5)
        pressure = int(result.stdout.strip()) if result.returncode == 0 else None
    # This cached image runs only cat, with no network, volumes, or model calls.
    result = subprocess.run(["docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--entrypoint", "cat", "couchdb:3.5", "/proc/meminfo"],
        capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("Docker memory is unavailable")
    values = {line.split(':')[0]: int(line.split()[1]) / 1024
              for line in result.stdout.splitlines() if ':' in line}
    return {"host_available_mib": host, "guest_available_mib": values['MemAvailable'],
            "host_pressure": pressure, "swapout": psutil.swap_memory().sout}


class Admission:
    def __init__(self, sample=memory_snapshot, clock=time.monotonic):
        self.sample, self.clock = sample, clock
        self.guard = threading.Lock()
        self.last_sample = float('-inf')
        self.snapshot = None
        self.reservations = []
        self.reason = 'not_sampled'

    def allow(self, kind):
        """Account for recently admitted workers before their memory appears in stats."""
        with self.guard:
            now = self.clock()
            if now - self.last_sample >= 5:
                previous = self.snapshot
                try:
                    self.snapshot = self.sample()
                    self.snapshot['paging'] = bool(previous and self.snapshot['swapout'] > previous['swapout'])
                except (OSError, ValueError, subprocess.SubprocessError, RuntimeError):
                    self.snapshot = None
                self.last_sample = now
            self.reservations = [(at, size) for at, size in self.reservations if now - at < 10]
            if self.snapshot is None:
                self.reason = 'memory_unavailable'
                return False
            if self.snapshot['host_pressure'] not in (1, 2) or self.snapshot['paging']:
                self.reason = 'host_pressure_or_paging'
                return False
            reserve = sum(size for _, size in self.reservations)
            needed = 768  # Covers observed judge peaks and execution startup overhead.
            available = min(self.snapshot['host_available_mib'], self.snapshot['guest_available_mib'])
            if available - reserve - needed < 1536:
                self.reason = 'reserved_headroom'
                return False
            self.reservations.append((now, needed))
            self.reason = 'admitted'
            return True


shared_admission = Admission()
