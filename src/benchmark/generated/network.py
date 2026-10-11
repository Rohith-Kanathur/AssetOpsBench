"""Small private bridges avoid exhausting Docker's large default address pools."""
from contextlib import contextmanager
import fcntl
import ipaddress
import json
from pathlib import Path
import subprocess

from .auth import private_json


def available_subnets(networks):
    used = [ipaddress.ip_network(row['Subnet']) for network in networks
            for row in (network.get('IPAM', {}).get('Config') or []) if row.get('Subnet')]
    for candidate in ipaddress.ip_network('10.253.0.0/16').subnets(new_prefix=28):
        if not any(candidate.version == item.version and candidate.overlaps(item) for item in used):
            yield str(candidate)


@contextmanager
def isolated_network(case):
    """Allocate under a cross-process lock; retain one separate bridge per case."""
    config_path = case / 'compose.json'
    config = json.loads(config_path.read_text())
    name = config['name'] + '_isolated'
    cache = Path.home() / '.cache/assetopsbench'
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (cache / 'network-allocation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ids = subprocess.check_output(['docker', 'network', 'ls', '-q'], text=True).split()
        networks = json.loads(subprocess.check_output(['docker', 'network', 'inspect', *ids], text=True)) if ids else []
        for subnet in available_subnets(networks):
            created = subprocess.run(['docker', 'network', 'create', '--driver', 'bridge',
                '--subnet', subnet, '--label', 'assetops.managed=true', name], capture_output=True, text=True)
            if created.returncode == 0:
                break
            if 'overlap' not in created.stderr.lower():
                raise RuntimeError('Docker network creation failed: ' + created.stderr.strip())
        else:
            raise RuntimeError('No free isolated Docker subnet')
    try:
        config['networks'] = {'default': {'external': True, 'name': name}}
        private_json(config_path, config)
        private_json(case / 'network.json', {'name': name, 'subnet': subnet, 'private_bridge': True})
        yield
    finally:
        removed = subprocess.run(['docker', 'network', 'rm', name], capture_output=True, text=True)
        if removed.returncode:
            # Never disconnect or stop another active container to force removal.
            private_json(case / 'network-cleanup-error.json', {'name': name, 'error': removed.stderr.strip()})
