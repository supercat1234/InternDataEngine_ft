import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/eval'))
from batch_cup_plate_pi05 import plan_jobs


class BatchTests(unittest.TestCase):
    def test_three_workers_each_one_hundred_with_fixed_worker_seeds(self):
        plan = plan_jobs(100, 3, 42)
        self.assertEqual([len(x) for x in plan], [100,100,100])
        self.assertEqual([x[0]['seed'] for x in plan], [42,43,44])
        self.assertEqual([set(x['seed'] for x in sequence) for sequence in plan], [{42},{43},{44}])
        self.assertEqual([x['episode'] for x in plan[0]], list(range(1,101)))
        jobs = sorted(sum(plan, []), key=lambda x: x['attempt'])
        self.assertEqual([x['attempt'] for x in jobs], list(range(1,301)))

    def test_one_worker_ten_attempts_stays_on_seed_one_hundred(self):
        plan = plan_jobs(10,1,100)
        self.assertEqual(len(plan),1)
        self.assertEqual(len(plan[0]),10)
        self.assertEqual({x['seed'] for x in plan[0]}, {100})
        self.assertEqual([len(x) for x in plan_jobs(2,3,42)], [2,2,2])

    def test_batch_runs_fake_children_and_aggregates_results(self):
        import contextlib, io, json, os, shlex, signal, tempfile, threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from unittest.mock import patch
        import batch_cup_plate_pi05 as batch
        import yaml

        from types import SimpleNamespace
        opener = SimpleNamespace(open=lambda *a, **kw: io.StringIO(json.dumps(dict(action_dim=14, absolute_actions=True, checkpoint='/fake/model'))))
        old_dir = os.getcwd()
        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'scripts').mkdir()
                fake = root/'fake.py'
                fake.write_text(f"import sys\nsys.path.insert(0,{str(Path(batch.__file__).parent)!r})\n" + '''import argparse,json,pathlib,random
from scene_random_stream import capture_state,restore_state,write_state
p=argparse.ArgumentParser()
p.add_argument('--output');p.add_argument('--seed',type=int)
p.add_argument('--scene-rng-state');p.add_argument('--episode-index',type=int)
a,_=p.parse_known_args()
state=json.loads(pathlib.Path(a.scene_rng_state).read_text())
restore_state(state,a.seed)
sample=random.random()
write_state(a.scene_rng_state,capture_state(a.seed,state['scenes_started']+1))
if a.seed==43 and a.episode_index==2:
    import os,signal
    os.kill(os.getpid(),signal.SIGTERM)
pathlib.Path(a.output,'result.json').write_text(json.dumps(dict(success=a.seed%2==0,strict_success=a.seed%2==0 and a.episode_index==1,reason='test_result',sample=sample)))
''')
                (root/'scripts/run_cup_plate_pi05_eval.sh').write_text(f'exec {shlex.quote(sys.executable)} {shlex.quote(str(fake))} "$@"\n')
                config = root/'config.yaml'
                config.write_text(yaml.safe_dump(dict(checkpoint='/fake/model',output_root=str(root/'results'))))
                argv=['batch','--total','2','--workers','2','--config',str(config),'--url','http://127.0.0.1:8097']
                with patch.object(batch,'REPO',root), patch.object(sys,'argv',argv), patch.object(batch.urllib.request, 'build_opener', return_value=opener), contextlib.redirect_stdout(io.StringIO()):
                    batch.main()
                report = json.loads(next((root/'results/batches').glob('*/result.json')).read_text())
                self.assertEqual(report['finished'],3)
                self.assertEqual(report['discarded'],1)
                self.assertEqual(report['requested'],4)
                self.assertEqual(report['successes'],2)
                self.assertAlmostEqual(report['success_rate'],2/3)
                self.assertEqual(report['strict_successes'],1)
                self.assertAlmostEqual(report['strict_success_rate'],1/3)
                self.assertEqual(len(list((root/'results').glob('20*_*'))),3)
                import csv
                with next((root/'results/batches').glob('*/summary.csv')).open() as stream:
                    attempts=sorted(csv.DictReader(stream),key=lambda x:int(x['attempt']))
                self.assertEqual([(int(x['worker']),int(x['episode']),int(x['seed'])) for x in attempts],
                                 [(1,1,42),(1,2,42),(2,1,43)])
                self.assertFalse(list((root/'results/batches').glob('*/attempt_004.log')))
                import random
                for worker,seed in [(1,42),(2,43)]:
                    rng=random.Random(seed)
                    samples=[json.loads(Path(x['output_dir'],'result.json').read_text())['sample']
                             for x in attempts if int(x['worker'])==worker]
                    self.assertEqual(samples,[rng.random() for _ in range(2 if worker==1 else 1)])
                    state_path=next((root/'results/batches').glob(f'*/worker_{worker}_scene_rng.json'))
                    self.assertEqual(json.loads(state_path.read_text())['scenes_started'],2)
        finally:
            os.chdir(old_dir)
            for signum, handler in handlers.items():
                signal.signal(signum,handler)

    def test_seed_overflow_rejected(self):
        with self.assertRaises(ValueError):
            plan_jobs(2,3,4294967295)
        self.assertEqual(len(plan_jobs(10,1,4294967295)[0]),10)

if __name__ == '__main__':
    unittest.main()
