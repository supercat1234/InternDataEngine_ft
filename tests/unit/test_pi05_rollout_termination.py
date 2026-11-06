import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / 'scripts/eval/rollout_termination.py'
spec = importlib.util.spec_from_file_location('termination', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TerminationTests(unittest.TestCase):
    def test_failed_episode_stops_at_forty_five_seconds(self):
        progress=module.PlacementProgress(1350,30,.5)
        step=0
        while step<progress.limit:
            step+=1
            self.assertIsNone(progress.update(step,{'candidate':False,'strict_candidate':False}))
        self.assertEqual(step/30,45.)
        self.assertIsNone(progress.success_step)

    def test_ordinary_success_at_seven_seconds_stops_at_twelve_without_strict(self):
        progress = module.PlacementProgress(1800,30,.5)
        reason = None
        step = 0
        while step < progress.limit and reason is None:
            step += 1
            placement = {'candidate':step>=196, 'strict_candidate':False}
            reason = progress.update(step,placement)
        self.assertEqual(progress.success_step,210)
        self.assertEqual(step,360)
        self.assertEqual(reason,'post_success_timeout')
        self.assertEqual(progress.outcome(placement), {'success':True,'strict_success':False})

    def test_strict_success_is_independent_and_may_be_lost_at_end(self):
        progress = module.PlacementProgress(1800,30,.5)
        for step in range(1, 16):
            progress.update(step,{'candidate':True,'strict_candidate':True})
        self.assertEqual(progress.outcome({'candidate':True,'strict_candidate':True}),
                         {'success':True,'strict_success':True})
        self.assertEqual(progress.outcome({'candidate':True,'strict_candidate':False}),
                         {'success':True,'strict_success':False})
        self.assertEqual(progress.outcome({'candidate':False,'strict_candidate':False}),
                         {'success':False,'strict_success':False})

    def test_success_near_step_limit_adds_full_five_seconds(self):
        rule = module.RolloutTermination(100, 30)
        for step in range(1, 100):
            self.assertIsNone(rule.update(step, False))
        rule.update(100, True)
        self.assertEqual(rule.limit, 250)
        for step in range(101, 250):
            self.assertIsNone(rule.update(step, True))
        self.assertEqual(rule.update(250, True), 'post_success_timeout')

    def test_success_timer_is_not_reset_when_condition_changes(self):
        rule = module.RolloutTermination(1800, 30)
        rule.update(224, True)
        for step in range(225, 374):
            self.assertIsNone(rule.update(step, step % 2 == 0))
        self.assertEqual(rule.update(374, False), 'post_success_timeout')
        self.assertEqual(rule.success_step, 224)

    def test_no_success_preserves_original_limit(self):
        rule = module.RolloutTermination(1800, 30)
        rule.update(1800, False)
        self.assertEqual(rule.limit, 1800)
        self.assertIsNone(rule.success_step)

if __name__ == '__main__':
    unittest.main()
