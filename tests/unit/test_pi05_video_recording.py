"""Exercise real encoding, including a recorder owner that exits without cleanup."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np

path = Path(__file__).resolve().parents[2] / 'scripts/eval/video_recording.py'
spec = importlib.util.spec_from_file_location('recording', path)
recording = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recording)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class VideoTests(unittest.TestCase):
    def test_recording_ends_five_seconds_after_ordinary_success(self):
        sys.path.insert(0,str(path.parent))
        from placement_evaluation import DEFAULT_CRITERIA,evaluate_placement
        from rollout_termination import PlacementProgress
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'ordinary_success.mp4'
            writer=recording.StreamingVideoWriter(target,30)
            progress=PlacementProgress(1800,30,.5)
            step,reason=0,None
            cup=np.eye(4)
            while step<progress.limit and reason is None:
                step+=1
                # Stable on the plate by 6.53 s, outside the strict flat footprint.
                cup[:3,3]=[.04 if step>=196 else .15,0,.005]
                placement=evaluate_placement(cup,np.eye(4),([0,0,0],[0,0,0]),
                    ([0,0,0],[0,0,0]),[0,0,.4],DEFAULT_CRITERIA)
                writer.write(np.full((256,256,3),step%256,np.uint8))
                reason=progress.update(step,placement)
            writer.release()
            self.assertEqual(progress.success_step,210)
            self.assertEqual(progress.outcome(placement),{'success':True,'strict_success':False})
            metadata=self.probe(target)
            self.assertEqual(int(metadata['nb_read_frames']),360)
            self.assertAlmostEqual(float(metadata['duration']),12.,places=3)

    def probe(self, path):
        return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,nb_read_frames,duration',
            '-of', 'json', str(path)]))['streams'][0]

    def test_normal_close_preserves_multiple_frames(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'test.mp4'
            writer = recording.StreamingVideoWriter(target, 30)
            for i in range(65):
                writer.write(np.full((256, 256, 3), i * 3, np.uint8))
            writer.release()
            metadata = self.probe(target)
            self.assertEqual(metadata['codec_name'], 'h264')
            self.assertEqual(int(metadata['nb_read_frames']), 65)
            self.assertGreater(float(metadata['duration']), 2.)

    def test_owner_abrupt_exit_still_has_playable_video(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'interrupted.mp4'
            code = f'''import sys, os, numpy as np
sys.path.insert(0, {str(path.parent)!r})
from video_recording import StreamingVideoWriter
writer = StreamingVideoWriter({str(target)!r}, 30)
for i in range(65):
    writer.write(np.full((256,256,3), i * 3, np.uint8))
os._exit(9)
'''
            result = subprocess.run([sys.executable, '-c', code], capture_output=True)
            self.assertEqual(result.returncode, 9)
            # The independent encoder sees EOF and finishes even without release().
            deadline = time.monotonic() + 10
            metadata = None
            while time.monotonic() < deadline:
                try:
                    metadata = self.probe(target)
                    if int(metadata.get('nb_read_frames', 0)) == 65:
                        break
                except (subprocess.CalledProcessError, KeyError, ValueError):
                    pass
                time.sleep(.1)
            self.assertIsNotNone(metadata)
            self.assertEqual(int(metadata['nb_read_frames']), 65)

if __name__ == '__main__':
    unittest.main()
