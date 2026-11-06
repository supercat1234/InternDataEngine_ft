"""Exercise the actual parallel shell launcher with fake workers, without Isaac/GPU."""
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKER = r'''
import json, os
from pathlib import Path
import signal, subprocess, sys, time
out = Path(os.environ['OUT_DIR'])
out.mkdir(parents=True, exist_ok=True)
seed = int(os.environ['SEED'])
(out / 'call.json').write_text(json.dumps(dict(seed=seed, args=sys.argv[1:], pid=os.getpid(),
                                             pgid=os.getpgrp(), sid=os.getsid(0))))
if os.environ.get('FAKE_HOLD'):
    child = subprocess.Popen([sys.executable, '-c',
        "import os,signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)", str(out / 'child.pid')])
    while True: time.sleep(.1)
# Both workers must enter before either can finish. Serial launching will fail.
deadline = time.monotonic() + 4
while len(list(out.parent.glob('inst_*/call.json'))) < int(os.environ['INSTANCES']):
    if time.monotonic() > deadline: sys.exit(9)
    time.sleep(.02)
if os.environ.get('FAKE_WAIT_GPU'):
    while not (out.parent / 'gpu_monitor.pid').exists():
        if time.monotonic() > deadline: sys.exit(10)
        time.sleep(.02)
if seed == int(os.environ.get('FAKE_FAIL_SEED', '-1')): sys.exit(7)
for index in range(int(sys.argv[1])):
    episode = out / ('episode_' + str(index))
    episode.mkdir()
    (episode / 'meta_info.pkl').write_bytes(b'fake metadata for launcher tests')
    print('Task is successful', flush=True)
'''


class CupParallelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cup parallel test ')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        scripts = self.repo / 'scripts'
        scripts.mkdir()
        self.launcher = scripts / 'run_cup_plate_parallel.sh'
        shutil.copyfile(ROOT / 'scripts/run_cup_plate_parallel.sh', self.launcher)
        (self.repo / 'fake_worker.py').write_text(WORKER)
        (scripts / 'run_cup_plate_desk.sh').write_text(
            '#!/usr/bin/env bash\nexec ' + __import__('shlex').quote(sys.executable) + ' ' +
            __import__('shlex').quote(str(self.repo / 'fake_worker.py')) + ' "$@"\n')
        self.output = self.repo / 'output with spaces'
        self.env = dict(os.environ, INSTANCES='2', PER_INSTANCE='3', BASE_SEED='42',
                        OUT_ROOT=str(self.output), WATCH_VRAM='0', LAUNCH_DELAY='0')

    def run_script(self, *args, env=None):
        return subprocess.run(['bash', str(self.launcher), *args], env=env or self.env,
                              cwd='/tmp', capture_output=True, text=True, timeout=12)

    def test_dry_run_creates_no_output_and_invalid_controls_fail(self):
        result = self.run_script('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SEED=42', result.stdout)
        self.assertIn('SEED=43', result.stdout)
        self.assertFalse(self.output.exists())
        for change in [dict(INSTANCES='0'), dict(BASE_SEED='4294967295'), dict(BASE_SEED='18446744073709551616'),
                       dict(WATCH_VRAM='2'), dict(LAUNCH_DELAY='-1')]:
            self.assertEqual(self.run_script('--dry-run', env={**self.env, **change}).returncode, 2)
        self.assertEqual(self.run_script('--random_seed=1').returncode, 2)

    def test_workers_run_concurrently_with_isolated_arguments_and_outputs(self):
        result = self.run_script('--task.note=value with spaces')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for index, seed in [(1, 42), (2, 43)]:
            out = self.output / f'inst_{index}'
            call = json.loads((out / 'call.json').read_text())
            self.assertEqual(call['seed'], seed)
            self.assertEqual(call['args'], ['3', f'--name={out}/engine', '--task.note=value with spaces'])
            self.assertEqual(call['pid'], call['pgid'])
            self.assertEqual(call['pid'], call['sid'])
        with (self.output / 'summary.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([row['saved_episodes'] for row in rows], ['3', '3'])
        self.assertEqual([row['task_successes'] for row in rows], ['3', '3'])
        self.assertEqual([row['exit_code'] for row in rows], ['0', '0'])
        self.assertEqual(self.run_script().returncode, 2)  # must not overwrite this run

    def test_one_failed_worker_does_not_stop_another_and_returns_failure(self):
        result = self.run_script(env={**self.env, 'FAKE_FAIL_SEED': '43'})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        with (self.output / 'summary.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([row['exit_code'] for row in rows], ['0', '7'])
        self.assertEqual([row['saved_episodes'] for row in rows], ['3', '0'])

    def test_gpu_monitor_uses_detected_capacity_and_exits_with_workers(self):
        binary = self.repo / 'bin'
        binary.mkdir()
        monitor = binary / 'nvidia-smi'
        monitor.write_text(f'#!{sys.executable}\n' +
            "import os,time\nfrom pathlib import Path\n" +
            "print('2026/10/28 19:00:00, 0, 32768, 12345, 80', flush=True)\n" +
            "(Path(os.environ['OUT_ROOT'])/'gpu_monitor.pid').write_text(str(os.getpid()))\n" +
            "time.sleep(60)\n")
        monitor.chmod(0o755)
        result = self.run_script(env={**self.env, 'WATCH_VRAM': '1', 'FAKE_WAIT_GPU': '1',
                                     'PATH': str(binary) + os.pathsep + self.env['PATH']})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('12345 / 32768 MiB', result.stdout)
        self.assertIn('32768', (self.output / 'gpu_watch.csv').read_text())
        pid = int((self.output / 'gpu_monitor.pid').read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_interrupt_cleans_up_term_resistant_grandchild(self):
        for sig, code in [(signal.SIGINT, 130), (signal.SIGTERM, 143)]:
            with self.subTest(signal=sig):
                output = self.repo / f'signal_{code}'
                env = {**self.env, 'OUT_ROOT': str(output), 'INSTANCES': '1', 'FAKE_HOLD': '1'}
                proc = subprocess.Popen(['bash', str(self.launcher)], env=env, cwd='/tmp',
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        start_new_session=True)
                child_pid = None
                try:
                    child_file = output / 'inst_1/child.pid'
                    deadline = time.monotonic() + 4
                    while not child_file.exists() and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(child_file.exists())
                    child_pid = int(child_file.read_text())
                    proc.send_signal(sig)
                    self.assertEqual(proc.wait(timeout=8), code)
                    stat = Path(f'/proc/{child_pid}/stat')
                    if stat.exists():
                        self.assertEqual(stat.read_text().split(') ', 1)[1][0], 'Z')
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    if child_pid is not None:
                        try: os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError: pass


if __name__ == '__main__':
    unittest.main()
