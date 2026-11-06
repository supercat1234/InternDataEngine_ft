"""Streaming H.264 recorder whose MP4 index survives interrupted simulation."""
from pathlib import Path
import subprocess
import numpy as np


class StreamingVideoWriter:
    def __init__(self, path, fps, width=256, height=256):
        self.path = Path(path)
        self.shape = (height, width, 3)
        self.frames = 0
        self.closed = False
        self.log = self.path.with_suffix('.encoder.log').open('wb')
        self.process = subprocess.Popen([
            'ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-threads', '1', '-preset', 'fast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-g', str(max(1, round(fps))), '-bf', '0',
            '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
            '-frag_duration', '1000000', '-flush_packets', '1', str(self.path)],
            stdin=subprocess.PIPE, stderr=self.log)

    def write(self, frame):
        frame = np.asarray(frame)
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(f'Expected RGB uint8 {self.shape}')
        if self.closed:
            raise RuntimeError('Video writer already closed')
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(f'Video encoder failed; see {self.log.name}') from exc
        self.frames += 1

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=30)
            if code:
                raise RuntimeError(f'Video encoder exit={code}; see {self.log.name}')
        finally:
            self.log.close()
