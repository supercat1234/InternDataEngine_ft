#!/usr/bin/env python3
"""Run TOTAL attempts per worker, with a continuing scene RNG stream per seed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import urllib.request
from zoneinfo import ZoneInfo
import yaml
from eval_reporting import allocate_output, discard_attempt, ABORT_REASONS
from scene_random_stream import initial_state, write_state

REPO = Path(__file__).resolve().parents[2]


def plan_jobs(total, workers, base_seed):
    if total < 1 or workers < 1 or base_seed < 0 or base_seed + workers - 1 > 4294967295:
        raise ValueError('Require positive total/workers and seeds within uint32')
    return [[{'attempt': worker * total + episode + 1, 'worker': worker + 1,
              'episode': episode + 1, 'seed': base_seed + worker}
             for episode in range(total)] for worker in range(workers)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total', type=int, default=100, help='Attempts PER worker (TOTAL); batch count = total * workers')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--config', default='local_configs/cup_plate_pi05_eval/eval.yaml')
    parser.add_argument('--url', default='http://127.0.0.1:8097')
    parser.add_argument('--episode-timeout', type=float, default=600, help='Wall time per attempt, including initialization/cleanup')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = plan_jobs(args.total, args.workers, args.base_seed)
    requested = args.total * args.workers
    if not args.episode_timeout > 0:
        parser.error('episode-timeout must be positive')
    os.chdir(REPO)
    if args.dry_run:
        print(json.dumps({'per_instance': args.total, 'total': requested, 'workers': len(jobs),
                          'seed_mode': 'fixed_worker_continuing_scene_stream', 'plan': jobs}, indent=2))
        return
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.url.rstrip('/') + '/health', timeout=10) as response:
        health = json.load(response)
    if health.get('action_dim') != 14 or not health.get('absolute_actions'):
        raise ValueError('Start the cup/plate 14D model service first')
    settings = yaml.safe_load(Path(args.config).read_text())
    if Path(health['checkpoint']).resolve() != Path(settings['checkpoint']).resolve():
        raise ValueError('Model service checkpoint differs from eval config')
    root = Path(settings['output_root']).resolve()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    batch = root / 'batches' / f'{stamp}_{os.getpid()}'
    batch.mkdir(parents=True, exist_ok=False)
    (batch / 'plan.json').write_text(json.dumps({'args': vars(args), 'per_instance': args.total,
        'total': requested, 'seed_mode': 'fixed_worker_continuing_scene_stream',
        'service': health, 'plan': jobs}, indent=2))
    lock = threading.Lock()
    stopped = threading.Event()
    processes = {}
    rows = []
    discarded = 0
    fields = ['attempt', 'worker', 'episode', 'seed', 'success', 'strict_success', 'reason', 'exit_code', 'output_dir']

    def kill_group(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=8)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    def stop(signum, frame):
        stopped.set()
        with lock:
            active = list(processes.values())
        for process in active:
            kill_group(process)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f'BATCH_READY per_instance={args.total} total={requested} workers={len(jobs)} batch={batch}', flush=True)

    def worker(sequence):
        nonlocal discarded
        stream_path = batch / f'worker_{sequence[0]["worker"]}_scene_rng.json'
        write_state(stream_path, initial_state(sequence[0]['seed']))
        for job in sequence:
            if stopped.is_set():
                break
            output = allocate_output(root)
            cmd = ['bash', str(REPO / 'scripts/run_cup_plate_pi05_eval.sh'), '--headless',
                   '--config', args.config, '--url', args.url, '--seed', str(job['seed']),
                   '--episode-index', str(job['episode']), '--scene-rng-state', str(stream_path),
                   '--output', str(output), '--reserved-output']
            if args.randomize:
                cmd.append('--randomize')
            if args.max_steps is not None:
                cmd.extend(['--max-steps', str(args.max_steps)])
            print(f'[start] worker={job["worker"]} episode={job["episode"]}/{args.total} attempt={job["attempt"]} seed={job["seed"]} output={output}', flush=True)
            code, failure = -1, None
            try:
                with (batch / f'attempt_{job["attempt"]:03d}.log').open('w') as log:
                    with lock:
                        if stopped.is_set():
                            discard_attempt(output)
                            break
                        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        processes[job['worker']] = process
                    try:
                        code = process.wait(timeout=args.episode_timeout)
                    except subprocess.TimeoutExpired:
                        failure = 'wall_timeout'
                        kill_group(process)
                        code = process.returncode
            except Exception as exc:
                failure = f'launch_error: {exc}'
            finally:
                with lock:
                    processes.pop(job['worker'], None)
            result_path = output / 'result.json'
            try:
                result = json.loads(result_path.read_text()) if result_path.exists() else None
            except (OSError, ValueError):
                result = None
            excluded = stopped.is_set() or code != 0 or failure is not None or result is None or result.get('reason') in ABORT_REASONS
            if excluded:
                discard_attempt(output)
                (batch / f'attempt_{job["attempt"]:03d}.log').unlink(missing_ok=True)
                with lock:
                    discarded += 1
                    publish_summary()
                print(f'[discard] worker={job["worker"]} episode={job["episode"]}; excluded from success rate', flush=True)
                continue
            row = {**job, 'success': bool(result.get('success')) and code == 0 and failure is None,
                   'reason': failure or result.get('reason', 'unknown'), 'exit_code': code, 'output_dir': str(output)}
            row['strict_success'] = row['success'] and bool(result.get('strict_success'))
            with lock:
                rows.append(row)
                with (batch / 'summary.csv').open('a', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                publish_summary()
            print(f'[done] attempt={job["attempt"]} success={row["success"]} strict_success={row["strict_success"]} reason={row["reason"]}', flush=True)

    def publish_summary():
        successes = sum(x['success'] for x in rows)
        strict_successes = sum(x['strict_success'] for x in rows)
        (batch / 'result.json').write_text(json.dumps({'requested': requested, 'finished': len(rows),
            'discarded': discarded, 'successes': successes,
            'success_rate': successes / len(rows) if rows else None,
            'strict_successes': strict_successes,
            'strict_success_rate': strict_successes / len(rows) if rows else None}, indent=2))

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for future in [pool.submit(worker, sequence) for sequence in jobs]:
            future.result()
    with lock:
        publish_summary()
    print(f'BATCH_DONE completed={len(rows)}/{requested} successes={sum(x["success"] for x in rows)} results={batch}', flush=True)

if __name__ == '__main__':
    main()
