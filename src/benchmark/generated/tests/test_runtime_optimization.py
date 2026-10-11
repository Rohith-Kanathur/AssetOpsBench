"""Isolation regressions for the memory optimizations; no model requests."""
import json
import subprocess
import sys
import threading

import pytest
import requests

from benchmark.generated.shared_database import Gateway


def test_importing_stirrup_does_not_import_other_harnesses():
    script = "import sys; import agent.stirrup_agent.runner; assert not any(n in sys.modules for n in ('agent.claude_agent.runner','agent.deep_agent.runner','agent.openai_agent.runner','agent.plan_execute.runner')); from agent import AgentResult, ClaudeAgentRunner; assert AgentResult and ClaudeAgentRunner"
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.fixture
def gateway(monkeypatch):
    gateway = Gateway("http://unused", ("admin", "secret"))
    calls = []

    def backend(method, path, **kwargs):
        calls.append((method, path, kwargs))
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(list(backend.databases) if path == "/_all_dbs" else {"ok": True}).encode()
        response.headers["Content-Type"] = "application/json"
        return response
    backend.databases = []
    monkeypatch.setattr(gateway, 'request', backend)
    thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    thread.start()
    yield gateway, backend, calls
    gateway.shutdown(); gateway.server_close(); thread.join()


def test_gateway_namespaces_every_request_and_hides_other_databases(gateway):
    server, backend, calls = gateway
    a, b = server.register(), server.register()
    backend.databases = [a['namespace']+'workorder', b['namespace']+'workorder', '_users']
    url = f'http://127.0.0.1:{server.server_port}'
    auth = (a['username'], a['password'])
    assert requests.get(url+'/_all_dbs', auth=auth).json() == ['workorder']
    requests.post(url+'/workorder/_find', auth=auth, json={'selector': {}})
    assert calls[-1][1] == '/'+a['namespace']+'workorder/_find'
    requests.get(url+'/'+b['namespace']+'workorder/secret', auth=auth)
    assert calls[-1][1].startswith('/'+a['namespace'])
    assert requests.get(url+'/workorder/secret', auth=('evaluation','wrong')).status_code == 401


@pytest.mark.parametrize('path', ['/_users', '/_replicate', '/_node/_local/_config',
    '/workorder/_security', '/workorder/%2e%2e/_users', '/workorder%2f..%2f_users'])
def test_gateway_rejects_global_routes_and_traversal(gateway, path):
    server, _, calls = gateway
    lease = server.register()
    response = requests.get(f'http://127.0.0.1:{server.server_port}'+path,
                            auth=(lease['username'],lease['password']))
    assert response.status_code == 403
    assert calls == []


def test_gateway_revokes_a_finished_cases_credentials(gateway):
    server, _, _ = gateway
    lease = server.register()
    server.unregister(lease)
    response = requests.get(f'http://127.0.0.1:{server.server_port}/_all_dbs',
                            auth=(lease['username'],lease['password']))
    assert response.status_code == 401


def test_cookie_authentication_is_also_scoped_and_revoked(gateway):
    server, _, calls = gateway
    a, b = server.register(), server.register()
    url = f'http://127.0.0.1:{server.server_port}'
    session = requests.Session()
    assert session.post(url+'/_session', auth=(a['username'],a['password']),
                        json={'name': a['username'], 'password': a['password']}).status_code == 200
    assert session.cookies['AuthSession'] not in {a['password'], b['password']}
    session.get(url+'/workorder/doc')
    assert calls[-1][1] == '/'+a['namespace']+'workorder/doc'
    server.unregister(a)
    assert session.get(url+'/workorder/doc').status_code == 401


def test_memory_admission_reserves_startups_and_stops_on_paging():
    from benchmark.generated.admission import Admission
    clock = [0.]
    sample = {'host_available_mib': 4000, 'guest_available_mib': 4000,
              'host_pressure': 1, 'swapout': 0}
    admission = Admission(lambda: dict(sample), lambda: clock[0])
    assert admission.allow('execution')
    assert admission.allow('judge')
    assert admission.allow('judge')
    assert not admission.allow('execution')
    clock[0] = 11
    assert admission.allow('judge')
    sample['swapout'] = 100
    clock[0] = 17
    assert not admission.allow('judge')
    assert admission.reason == 'host_pressure_or_paging'


def test_execution_queue_does_not_wait_for_judges(monkeypatch, tmp_path):
    from benchmark.generated import pipeline
    import time
    first_judge_started, last_executed = threading.Event(), threading.Event()
    observed = []
    class Pool:
        def preflight(self):
            return [{'health':'ready','remaining':90}]*5
    class Admission:
        def allow(self, _): return True
    def judges(cases, *, case_source, source_done, **kwargs):
        while not source_done() or len(observed)<3:
            for case in case_source():
                first_judge_started.set()
                assert last_executed.wait(2), 'Execution worker was occupied by judging'
                observed.append(case)
            time.sleep(.005)
    def execute(item):
        if item[0].name == 'third': last_executed.set()
        time.sleep(.01)
        return {'status':'completed'}
    monkeypatch.setattr(pipeline,'judge_cases',judges)
    pipeline.run([(tmp_path/name,) for name in ['first','second','third']], execute, lambda:None,
                 execution_jobs=1, judge_jobs=5, admission=Admission(), pool=Pool())
    assert len(observed)==3 and first_judge_started.is_set()
