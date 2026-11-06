import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

path = Path(__file__).resolve().parents[2] / 'scripts/eval/eval_reporting.py'
spec = importlib.util.spec_from_file_location('reporting', path)
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


class ReportingTests(unittest.TestCase):
    def test_existing_csv_is_upgraded_without_losing_old_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'summary.csv').write_text('finished_at,mode,success,reason,custom\nold,closed_loop,True,stable_placement,keep\n')
            result = dict(mode='closed_loop',success=True,strict_success=False,reason='stable_placement',steps=360)
            reporting.save_report(root,result,[],dict(checkpoint='model',seed=42,fps=30),[],time.perf_counter(),root)
            with (root/'summary.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['success'],'True')
            self.assertEqual(rows[0]['strict_success'],'')
            self.assertEqual(rows[0]['custom'],'keep')
            self.assertEqual(rows[1]['strict_success'],'False')

    def test_only_completed_attempts_are_saved_and_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = dict(checkpoint='/test/model', seed=42, fps=30)
            for name, success, reason in [('completed', True, 'stable_placement'), ('failed',False,'max_steps'),
                                           ('startup',False,'error'),('cancelled',False,'interrupted'),('closed',False,'window_closed')]:
                output = root / name
                output.mkdir()
                result = dict(mode='closed_loop', success=success, reason=reason, steps=15 if success else 0)
                report = reporting.save_report(output, result, [], settings, [.1, .2] if success else [], time.perf_counter(), root)
                if reason in reporting.ABORT_REASONS:
                    self.assertIsNone(report)
                    self.assertFalse(output.exists())
                    continue
                saved = json.loads((output / 'result.json').read_text())
                self.assertEqual(saved['success'], success)
                self.assertNotIn('videos',saved)
                self.assertNotIn('video_metadata',saved)
                self.assertLessEqual(len(saved),12)
                self.assertEqual(saved['status'], 'success' if success else 'not_completed')
            with (root / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]['reason'], 'max_steps')

    def test_date_sequence_concurrency(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(lambda _: reporting.allocate_output(directory, '20261103'), range(8)))
            self.assertEqual({p.name for p in outputs}, {f'20261103_{i:03d}' for i in range(1, 9)})
            self.assertEqual(reporting.allocate_output(directory, '20261104').name, '20261104_001')

    def test_replay_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = dict(mode='replay', success=True, reason='stable_placement', steps=10)
            reporting.save_report(root, result, [], dict(checkpoint='model', seed=1, fps=30), [], time.perf_counter(), root)
            self.assertEqual(result['status'], 'diagnostic')
            self.assertIsNone(result['mean_inference_seconds'])

if __name__ == '__main__':
    unittest.main()
