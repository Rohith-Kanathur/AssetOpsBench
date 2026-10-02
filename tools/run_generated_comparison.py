"""Run the configured targets concurrently on one captured database snapshot."""
import hashlib
import gzip
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from benchmark.database_audit import serve
from benchmark.measurement import write_json


def target_suite(config, spec):
    """A cohort may use its own suite while retaining the common model/judge."""
    path = Path(spec.get('suite') or config['suite'])
    return path if path.is_absolute() else ROOT / path


def main():
    os.chdir(ROOT)
    load_dotenv(ROOT/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'benchmarks/generated-comparison.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'generated/comparisons/transformer')
    parser.add_argument('--expected-snapshot-sha256')
    parser.add_argument('--snapshot-file',type=Path,help='Reuse a captured snapshot, or save it on first use')
    parser.add_argument('--repetition-index',type=int)
    args=parser.parse_args()
    config = json.loads(args.config.read_text())
    from benchmark.generated_suite_runner import completed_scenarios, validate_executor
    # Validate every suite before cloning databases or starting any worker.
    for spec in config['targets']:
        validate_executor(spec['agent'])
        completed_scenarios(target_suite(config, spec))
    output=args.output_dir.resolve()
    if any(list((output/spec['name']/'measurements').glob('*.json')) for spec in config['targets']):
        parser.error('Output already contains measured runs; choose an empty --output-dir for a fresh comparison.')
    base = os.environ['COUCHDB_URL'].rstrip('/')
    auth = (os.environ.get('COUCHDB_USERNAME','admin'),os.environ.get('COUCHDB_PASSWORD','password'))
    client = httpx.Client(auth=auth,timeout=120)
    if args.snapshot_file and args.snapshot_file.exists():
        snapshot=json.loads(gzip.decompress(args.snapshot_file.read_bytes()))
    else:
        response = client.get(base+'/_all_dbs'); response.raise_for_status()
        databases = [d for d in response.json() if not d.startswith(('_','eval_'))]
        snapshot = {}
        for db in databases:
            response = client.get(base+'/'+db+'/_all_docs',params={'include_docs':'true','attachments':'true'})
            response.raise_for_status()
            docs = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
            for doc in docs: doc.pop('_rev',None)
            snapshot[db] = docs
    digest = hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
    if args.expected_snapshot_sha256 and digest != args.expected_snapshot_sha256:
        parser.error('Database snapshot differs from the reference repetition; no runs were launched.')
    if args.snapshot_file and not args.snapshot_file.exists():
        args.snapshot_file.parent.mkdir(parents=True,exist_ok=True)
        args.snapshot_file.write_bytes(gzip.compress(json.dumps(snapshot).encode(),mtime=0))
    stamp = str(int(time.time()))
    snapshot_stamp=str(int(args.snapshot_file.stat().st_mtime)) if args.snapshot_file else stamp
    workers = []
    servers = []
    for index,spec in enumerate(config['targets']):
        name=spec['name']; target=output/name
        suite=target_suite(config,spec)
        target.mkdir(parents=True,exist_ok=True)
        prefix='eval_'+name.lower().replace('-','_')+'_'+stamp+'_'+uuid4().hex[:12]+'_'
        for db,docs in snapshot.items():
            client.put(base+'/'+prefix+db).raise_for_status()
            for offset in range(0,len(docs),500):
                response=client.post(base+'/'+prefix+db+'/_bulk_docs',json={'docs':docs[offset:offset+500]})
                response.raise_for_status()
                if any('error' in r for r in response.json()): raise RuntimeError('snapshot clone failed')
        audit=target/'database_audit'
        active_run_file = target/'active-run.txt'
        active_run_file.write_text('preflight')
        server=serve(base,auth,prefix,audit,0,run_id_file=active_run_file)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        servers.append(server)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONUNBUFFERED':'1',
             'BENCHMARK_DB_PROXY_URL':f'http://127.0.0.1:{server.server_port}',
             'BENCHMARK_DB_AUDIT_DIR':str(audit),
             'BENCHMARK_DB_RUN_ID_FILE':str(active_run_file)}
        if args.repetition_index is not None:
            env['BENCHMARK_REPETITION_INDEX']=str(args.repetition_index)
        policy={'snapshot_sha256':digest,'namespace':prefix,'snapshot_captured_at':snapshot_stamp,
                'namespace_created_at':stamp,
                'reset_policy':'same initial snapshot per model; persisted within model in suite order',
                'database_count':len(snapshot),'document_count':sum(map(len,snapshot.values()))}
        policy.update(cohort=spec.get('cohort'),suite=str(suite))
        write_json(target/'environment.json',policy)
        command=[sys.executable,'-m','benchmark.generated_suite_runner',str(suite),
                 '--output-dir',str(output),'--name',name,'--agent',spec['agent'],'--model-id',spec['model_id'],
                 '--judge-model',config['judge'],'--concurrency-level',str(len(config['targets']))]
        if config.get('allow_same_model_judge'):
            command+=['--allow-self-judge']
        if spec.get('reasoning_effort'): command+=['--reasoning-effort',spec['reasoning_effort']]
        # Exercise the actual CouchDB client used by IoT before any model runs.
        import couchdb3
        probe = couchdb3.Database(os.environ.get('IOT_DBNAME','iot'),url=env['BENCHMARK_DB_PROXY_URL'],user=auth[0],password=auth[1])
        if not probe or probe.info().get('doc_count',0) <= 0:
            raise RuntimeError('IoT database preflight failed')
        workers.append((name,command,env,target,suite))
        print(name,'snapshot ready',flush=True)
    def run(worker):
        name,command,env,target,suite=worker
        # Preserve every failed attempt and resume only unfinished scenarios.
        for attempt in range(1,len(completed_scenarios(suite)[0])*3+4):
            with (target/f'suite-attempt-{attempt}.log').open('w') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                print(name,'started',process.pid,flush=True)
                code=process.wait()
            if code==0:
                print(name,'complete',flush=True)
                return
            print(name,'failed suite attempt',attempt,'exit',code,flush=True)
        print(name,'stopped after exhausting invocation retries',flush=True)
    threads=[threading.Thread(target=run,args=(worker,)) for worker in workers]
    for thread in threads: thread.start()
    judges=[]
    for name,_,env,target,suite in workers:
        log=(target/'live-grading.log').open('w')
        judges.append((subprocess.Popen([sys.executable,str(ROOT/'tools/grade_live_comparison.py'),
            '--target',str(target),'--suite',str(suite),'--judge',config['judge']],
            env=env,stdout=log,stderr=subprocess.STDOUT),log))
    for thread in threads: thread.join()
    for judge,log in judges:
        judge.wait();log.close()
    for server in servers: server.shutdown();server.server_close()
    from benchmark.comparison_report import render
    render(output,output/'comparison.html',config=config)


if __name__=='__main__':main()
