import json
from itertools import islice
from types import SimpleNamespace

from benchmark.generated import network


def test_small_subnets_skip_overlaps_and_do_not_use_default_pools():
    existing = [{'IPAM': {'Config': [{'Subnet': '10.253.0.0/27'}, {'Subnet': '172.18.0.0/16'}]}}]
    subnets = list(islice(network.available_subnets(existing), 12))
    assert len(set(subnets)) == 12
    assert subnets[0] == '10.253.0.32/28'


def test_separate_network_is_released_without_forced_disconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(network.Path, 'home', lambda: tmp_path)
    commands = []
    def output(command, **kwargs):
        return '' if command[-2:] == ['ls', '-q'] else '[]'
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='network-id', stderr='')
    monkeypatch.setattr(network.subprocess, 'check_output', output)
    monkeypatch.setattr(network.subprocess, 'run', run)
    (tmp_path / 'compose.json').write_text(json.dumps({'name': 'case-one', 'services': {}}))
    with network.isolated_network(tmp_path):
        config = json.loads((tmp_path / 'compose.json').read_text())
        assert config['networks']['default'] == {'external': True, 'name': 'case-one_isolated'}
        assert json.loads((tmp_path / 'network.json').read_text())['subnet'] == '10.253.0.0/28'
    assert commands[-1] == ['docker', 'network', 'rm', 'case-one_isolated']
    assert not any('--force' in command for command in commands)
