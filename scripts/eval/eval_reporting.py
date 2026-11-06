"""Per-run artifacts and append-only evaluation index (no Isaac imports)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ABORT_REASONS = {'error', 'interrupted', 'window_closed', 'wall_timeout', 'process_exit', 'launch_error'}


def discard_attempt(output):
    """Only called for the current, newly allocated attempt directory."""
    shutil.rmtree(output, ignore_errors=False) if Path(output).exists() else None


def concise_result(result):
    keys = ['mode', 'status', 'seed', 'episode', 'success', 'strict_success', 'reason',
            'simulation_seconds', 'success_simulation_seconds', 'post_success_seconds']
    report = {key: result[key] for key in keys if key in result}
    report['failed_checks'] = result.get('final_placement', {}).get('failed_checks', [])
    report['strict_failed_checks'] = result.get('final_placement', {}).get('strict_failed_checks', [])
    return report


@contextmanager
def capture_log(path):
    """Capture Python and native stdout/stderr, while retaining terminal output."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    # Keep the log sink alive while the evaluation handles foreground-group
    # Ctrl+C/SIGTERM; otherwise cleanup prints can fail with a broken pipe.
    tee = subprocess.Popen(['tee', str(path)], stdin=subprocess.PIPE, start_new_session=True)
    try:
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)
        tee.stdin.close()
        tee.wait(timeout=10)


def save_report(output, result, records, settings, durations, started, index_root):
    output = Path(output)
    if result['reason'] in ABORT_REASONS:
        discard_attempt(output)
        return None
    videos = {name: str(output / f'{name}.mp4') for name in
              ('TOP_CAM', 'LEFT_WRIST_CAM', 'RIGHT_WRIST_CAM')
              if (output / f'{name}.mp4').is_file() and (output / f'{name}.mp4').stat().st_size > 0}
    mode = result['mode']
    result.setdefault('strict_success', False)
    status = ('error' if result['reason'] == 'error' else
              'interrupted' if result['reason'] in ('interrupted', 'window_closed') else
              'diagnostic' if mode != 'closed_loop' else
              'success' if result['success'] else 'not_completed')
    result.update(status=status, checkpoint=settings['checkpoint'], seed=settings['seed'],
                  output_dir=str(output), fps=settings['fps'],
                  simulation_seconds=result['steps'] / settings['fps'],
                  wall_seconds=time.perf_counter()-started,
                  inference_count=len(durations),
                  mean_inference_seconds=sum(durations)/len(durations) if durations else None,
                  returned_home=bool(records[-1]['home']) if records else False,
                  videos=videos, video_frames=len(records) if videos else 0,
                  finished_at=datetime.now(timezone.utc).isoformat())
    (output / 'trajectory.json').write_text(json.dumps(records))
    report = concise_result(result)
    (output / 'result.json').write_text(json.dumps(report, indent=2))
    (output / 'progress.json').write_text(json.dumps(report, indent=2))
    # Full per-run result remains authoritative; CSV is a convenient shared index.
    fields = ['output_dir', 'mode', 'status', 'success', 'strict_success', 'reason', 'steps', 'simulation_seconds',
              'wall_seconds', 'inference_count', 'mean_inference_seconds', 'returned_home',
              'seed', 'checkpoint', 'finished_at', 'error', 'recording_error']
    index = Path(index_root) / 'summary.csv'
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open('a+', newline='') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        existing_fields = next(csv.reader(stream), None)
        if existing_fields and existing_fields != fields:
            # Upgrade an existing index under its lock, retaining old rows and
            # unknown columns. Old strict-success values remain unknown/blank.
            fields += [name for name in existing_fields if name not in fields]
            if existing_fields != fields:
                previous = list(csv.DictReader(stream, fieldnames=existing_fields))
                stream.seek(0)
                stream.truncate()
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(previous)
        stream.seek(0, os.SEEK_END)
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow({key: result.get(key, '') for key in fields})
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
    return report


def allocate_output(root, day=None):
    """Atomically reserve date_sequence, including concurrent invocations."""
    from zoneinfo import ZoneInfo
    import re
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    day = day or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    pattern = re.compile(rf'{day}_(\d+)$')
    number = max((int(match[1]) for p in root.iterdir()
                  if (match := pattern.fullmatch(p.name))), default=0) + 1
    while True:
        path = root / f'{day}_{number:03d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def finalize_videos(output, names):
    """Make broadly playable H.264/yuv420p MP4s, then inspect actual metadata."""
    metadata = {}
    for name in names:
        path = Path(output) / f'{name}.mp4'
        encoded = path.with_suffix('.encoding.mp4')
        try:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(path),
                            '-an', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
                            '-preset', 'fast', '-crf', '20', '-movflags', '+faststart', str(encoded)],
                           check=True, capture_output=True, timeout=120)
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=codec_name,nb_frames,duration,width,height,pix_fmt', '-of', 'json', str(encoded)], timeout=30))
            stream = probe['streams'][0]
            if int(stream.get('nb_frames', 0)) < 1:
                raise ValueError('Encoded video has no frames')
            # Verify a frame really decodes, in addition to probing the container.
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(encoded),
                            '-frames:v', '1', '-f', 'null', '-'], check=True, capture_output=True, timeout=30)
            encoded.replace(path)
            metadata[name] = {'path': str(path), **stream}
        except Exception:
            encoded.unlink(missing_ok=True)
            raise
    return metadata
