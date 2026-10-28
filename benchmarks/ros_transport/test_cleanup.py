"""Exercise real shell process groups on error and external termination."""
import os
from pathlib import Path
import signal
import subprocess
import time
import pytest

CLEANUP = Path(__file__).with_name('cleanup.sh').resolve()


def live(pid):
    try:
        # A killed child may briefly remain a zombie until init reaps it.
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] != 'Z'
    except FileNotFoundError:
        return False


def reclaim_groups(directory):
    for path in directory.glob('*.pgid'):
        try:
            pgid = int(path.read_text().strip())
            if pgid > 1 and pgid != os.getpgrp():
                os.killpg(pgid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass


@pytest.mark.parametrize('trigger', ['failure', 'term', 'interrupt'])
def test_descendants_stop_on_failure_and_signal(tmp_path, trigger):
    script = tmp_path / 'driver.sh'
    script.write_text('''#!/bin/bash
set -euo pipefail
source "$1"
for which in sub router pub; do
  setsid bash -c 'sleep 300 & echo $! > "$1"; wait' _ "$2/$which.child" &
  echo $! > "$2/$which.pgid"
  case "$which" in sub) sub_group=$!;; router) router_group=$!;; pub) pub_group=$!;; esac
done
for which in sub router pub; do while [[ ! -s "$2/$which.child" ]]; do sleep .01; done; done
printf 'ready\\n' > "$2/ready"
if [[ "$3" == failure ]]; then while [[ ! -e "$2/fail" ]]; do sleep .01; done; false; fi
while true; do sleep 0.05; done
''')
    process = subprocess.Popen(['bash', str(script), str(CLEANUP), str(tmp_path), trigger])
    try:
        for _ in range(500):
            if (tmp_path/'ready').exists(): break
            if process.poll() is not None: pytest.fail('driver exited before starting children')
            time.sleep(.01)
        else: pytest.fail('child startup timed out')
        children = [int((tmp_path/f'{which}.child').read_text()) for which in ('sub','router','pub')]
        assert all(live(p) for p in children)
        if trigger == 'failure':
            (tmp_path/'fail').touch()
        else:
            os.kill(process.pid, signal.SIGTERM if trigger == 'term' else signal.SIGINT)
        assert process.wait(timeout=5) != 0
        for _ in range(100):
            if not any(live(p) for p in children): break
            time.sleep(.01)
        assert not any(live(p) for p in children)
    finally:
        reclaim_groups(tmp_path)
        if process.poll() is None:
            process.kill()
            process.wait()


def test_actual_runner_cleans_subscriber_after_publisher_failure(tmp_path):
    transport = tmp_path/'transport'
    transport.write_text('''#!/bin/bash
if [[ $1 == sub ]]; then
  ps -o pgid= -p $$ > "$OUT/sub.pgid"
  sleep 300 & echo $! > "$OUT/child.pid"
  wait
else
  exit 23
fi
''')
    transport.chmod(0o755)
    env = dict(os.environ, OUT=str(tmp_path), ARM_ORDER='fast', SECONDS_PER_ARM='1', WARMUP_SECONDS='0')
    runner = Path(__file__).with_name('run.sh')
    try:
        result = subprocess.run(['bash', str(runner)], env=env, timeout=10, capture_output=True, text=True)
        assert result.returncode == 23
        child = int((tmp_path/'child.pid').read_text())
        for _ in range(100):
            if not live(child): break
            time.sleep(.01)
        assert not live(child)
    finally:
        reclaim_groups(tmp_path)


def test_reclaims_group_if_driver_cannot_run_trap(tmp_path):
    driver = tmp_path/'stuck.sh'
    driver.write_text('''#!/bin/bash
setsid bash -c 'sleep 300 & echo $! > "$1/child.pid"; wait' _ "$1" &
echo $! > "$1/sub.pgid"
while true; do sleep .05; done
''')
    process = subprocess.Popen(['bash', str(driver), str(tmp_path)])
    try:
        for _ in range(500):
            if (tmp_path/'child.pid').exists() and (tmp_path/'sub.pgid').exists(): break
            time.sleep(.01)
        else: pytest.fail('child startup timed out')
        child = int((tmp_path/'child.pid').read_text())
        assert live(child)
        process.kill()
        process.wait(timeout=5)
        reclaim_groups(tmp_path)
        for _ in range(100):
            if not live(child): break
            time.sleep(.01)
        assert not live(child)
    finally:
        reclaim_groups(tmp_path)
        if process.poll() is None:
            process.kill()
            process.wait()
