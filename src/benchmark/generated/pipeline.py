"""Independent, memory-admitted execution and judging queues."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import queue
import threading
import time

from agent.codex_accounts import Pool
from .admission import shared_admission
from .repeated_judge import judge_cases


def run(planned, execute, report, *, execution_jobs=12, judge_jobs=14,
        judge=True, admission=shared_admission, pool=None):
    """Execution workers never wait for judges; each queue keeps its own cap."""
    ready = queue.Queue()
    executions_done = threading.Event()

    def completed_cases():
        result = []
        while True:
            try:
                result.append(ready.get_nowait())
            except queue.Empty:
                return result

    def grade():
        return judge_cases([], pool=pool, jobs=judge_jobs, preflight=False,
            case_source=completed_cases, source_done=executions_done.is_set,
            on_complete=lambda *_: report(), admission=admission)

    if judge:
        pool = pool or Pool(sessions_per_account=2)
        rows = pool.preflight()
        if sum(a.get('health') in {'ready', 'exhausted'} and
               (a.get('remaining', 0) > 0 or (a.get('resets') or 0) > 0) for a in rows) < 5:
            raise RuntimeError('Five distinct usable judge subscriptions are required before execution')
    with ThreadPoolExecutor(max_workers=1) as judge_executor:
        judging = judge_executor.submit(grade) if judge else None
        try:
            with ThreadPoolExecutor(max_workers=execution_jobs) as executor:
                pending, active = deque(planned), {}
                while pending or active:
                    if judging is not None and judging.done():
                        judging.result()  # Do not spend on more executions after a judge failure.
                    while pending and len(active) < execution_jobs and admission.allow('execution'):
                        item = pending.popleft()
                        active[executor.submit(execute, item)] = item
                    if not active:
                        time.sleep(.2)
                        continue
                    completed, _ = wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in completed:
                        item = active.pop(future)
                        result = future.result()
                        if judge and result.get('status') == 'completed':
                            ready.put(item[0])
                        report()
        finally:
            executions_done.set()
        if judging is not None:
            judging.result()
    report()
