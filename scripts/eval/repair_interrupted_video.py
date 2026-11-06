#!/usr/bin/env python3
"""Recover unfinalized OpenCV mp4v payload using matching reference codec data.

Writes to a separate output directory and drops the final incomplete video packet.
The reference must match the original codec, dimensions and frame rate.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=30)
    args = parser.parse_args()
    ref = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=extradata,codec_name,width,height', '-show_data', '-of', 'json', str(args.reference)]))['streams'][0]
    if ref['codec_name'] != 'mpeg4':
        raise ValueError('Reference must use original mp4v/mpeg4 codec')
    header = bytes.fromhex(''.join(line.split(':', 1)[1].split('  ')[0].replace(' ', '')
                                  for line in ref['extradata'].strip().splitlines()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'source': str(args.run_dir), 'reference': str(args.reference),
              'incomplete_final_packet_discarded': True, 'videos': {}}
    for name in ('TOP_CAM', 'LEFT_WRIST_CAM', 'RIGHT_WRIST_CAM'):
        source = args.run_dir / f'{name}.mp4'
        data = source.read_bytes()
        payload = data[data.index(b'mdat')+4:]
        cut = payload.rfind(bytes.fromhex('000001b6'))
        if cut <= 0:
            raise ValueError(f'No recoverable MPEG4 frames: {source}')
        # Drop a GOP header belonging to that incomplete final frame too.
        gop = payload.rfind(bytes.fromhex('000001b3'), 0, cut)
        previous = payload.rfind(bytes.fromhex('000001b6'), 0, cut)
        if gop > previous:
            cut = gop
        raw = args.output_dir / f'{name}.m4v'
        raw.write_bytes(header + payload[:cut])
        target = args.output_dir / f'{name}.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-y', '-fflags', '+genpts',
            '-r', str(args.fps), '-f', 'm4v', '-i', str(raw), '-c:v', 'libx264', '-threads', '1',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
        metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,duration,codec_name,width,height',
            '-of', 'json', str(target)]))['streams'][0]
        report['videos'][name] = metadata
        print(name, metadata, flush=True)
        raw.unlink()
    (args.output_dir / 'video_recovery.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
