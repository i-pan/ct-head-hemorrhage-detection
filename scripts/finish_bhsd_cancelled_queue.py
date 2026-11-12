"""Observe remaining 25-pass job after cancelling 50-pass jobs; launches no training.

The original controller is SIGSTOPped because it keeps its queue in memory.
Only terminate that controller after preserved training processes have exited,
so systemd cannot terminate preserved jobs through its original service cgroup.
"""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'logs/bhsd_convnext_schedules_20261110/queue.json'
CONTROLLER = 1335729


def alive(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ')[1][0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    assert b'run_bhsd_convnext_schedules.py' in Path(f'/proc/{CONTROLLER}/cmdline').read_bytes()
    assert Path(f'/proc/{CONTROLLER}/stat').read_text().split(') ')[1][0] == 'T'
    while True:
        records = json.loads(QUEUE.read_text())
        pending = False
        changed = False
        for r in records:
            if r['status'] != 'running':
                continue
            assert r['passes'] == 25
            if alive(r['pid']):
                pending = True
                continue
            status = Path(r['output']) / 'status.json'
            s = json.loads(status.read_text()) if status.exists() else {}
            complete = s.get('status') == 'complete' and s.get('epoch') == 25
            r.update(status='complete' if complete else 'failed', finished_unix=time.time(), completion_verified_by='saved endpoint status after process exit')
            changed = True
        if changed:
            tmp = QUEUE.with_suffix('.observer.tmp')
            tmp.write_text(json.dumps(records, indent=2))
            tmp.replace(QUEUE)
        if not pending:
            os.kill(CONTROLLER, signal.SIGKILL)
            print('Preserved jobs exited; retired cancelled queue controller.', flush=True)
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
