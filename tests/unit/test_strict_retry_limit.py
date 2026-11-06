"""Exercise strict retry counts and propagation without simulation or GPU work."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nimbus.components.data.scene import Scene
from nimbus.components.load.base_randomizer import LayoutRandomizer, StrictRetryLimitError
from nimbus.components.planner.base_seq_planner import SequencePlanner
from nimbus_extension.components.load.env_randomizer import EnvRandomizer
from nimbus_extension.components.plan_with_render.plan_with_render import EnvPlanWithRender
from nimbus.utils.config_processor import ConfigProcessor


class FakeRandomizer(LayoutRandomizer):
    def __init__(self, *args, **kwargs):
        self.calls = 0
        super().__init__(*args, **kwargs)

    def randomize_scene(self, scene):
        self.calls += 1
        return scene


class FakePlanner(SequencePlanner):
    def generate_sequence(self):
        return None


class StrictRetryTests(unittest.TestCase):
    def make_randomizer(self, targets=1, retries=2, strict=True):
        scene = Scene(name='cup_plate')
        return FakeRandomizer(iter([scene]), targets, strict, retries)

    def test_limit_counts_extra_attempts_and_zero_stops_after_initial_failure(self):
        for retries in [0, 1, 2, 20, 40]:
            with self.subTest(retries=retries):
                randomizer = self.make_randomizer(retries=retries)
                for _ in range(retries + 1):
                    next(randomizer).update_generate_status(False)
                with self.assertRaisesRegex(StrictRetryLimitError, f'attempts_for_target={retries + 1}'):
                    next(randomizer)
                self.assertEqual(randomizer.calls, retries + 1)
                self.assertEqual(randomizer.cur_index, 1)

    def test_success_resets_budget_for_each_target(self):
        randomizer = self.make_randomizer(targets=2, retries=1)
        next(randomizer).update_generate_status(False)
        next(randomizer).update_generate_status(True)
        next(randomizer).update_generate_status(False)
        self.assertEqual(randomizer.strict_retry_count, 0)
        next(randomizer).update_generate_status(True)
        with self.assertRaises(StopIteration):
            next(randomizer)
        self.assertEqual(randomizer.calls, 4)

    def test_new_scene_resets_budget(self):
        randomizer = FakeRandomizer(iter([Scene(name='first'), Scene(name='second')]), 1, True, 1)
        next(randomizer).update_generate_status(False)
        next(randomizer).update_generate_status(True)
        self.assertEqual(next(randomizer).name, 'second')
        self.assertEqual(randomizer.strict_retry_count, 0)
        next(randomizer).update_generate_status(True)
        with self.assertRaises(StopIteration):
            next(randomizer)

    def test_nonstrict_ignores_retry_limit_and_unlimited_preserves_old_behavior(self):
        randomizer = self.make_randomizer(targets=3, retries=0, strict=False)
        for _ in range(3):
            next(randomizer).update_generate_status(False)
        with self.assertRaises(StopIteration):
            next(randomizer)
        unlimited = self.make_randomizer(retries=-1)
        for _ in range(40):
            next(unlimited).update_generate_status(False)
        next(unlimited).update_generate_status(True)
        with self.assertRaises(StopIteration):
            next(unlimited)
        self.assertEqual(unlimited.calls, 41)

    def test_bad_limits_are_rejected(self):
        for value in [-2, None, True, 2.5, '2']:
            with self.subTest(limit=value), self.assertRaises(ValueError):
                self.make_randomizer(retries=value)

    def test_env_randomizer_and_plan_with_render_propagate_limit_even_without_debug(self):
        workflow = SimpleNamespace(randomization=lambda path: True, plan_with_render=lambda: 0)
        randomizer = EnvRandomizer(iter([Scene(name='cup_plate', wf=workflow)]),
                                   random_num=1, strict_mode=True, max_retries=2)
        planner = EnvPlanWithRender(randomizer)
        with patch('nimbus_extension.components.plan_with_render.plan_with_render.is_debug_mode', return_value=False):
            for _ in range(3):
                scene, _, obs = next(planner)
                self.assertIsNone(obs)
                scene.update_generate_status(False)
            with self.assertRaises(StrictRetryLimitError):
                next(planner)

    def test_sequence_planner_also_propagates_limit_without_debug(self):
        planner = FakePlanner(self.make_randomizer(retries=0), planner_cfg=None)
        with patch('nimbus.components.planner.base_seq_planner.is_debug_mode', return_value=False):
            scene, sequence = next(planner)
            self.assertIsNone(sequence)
            scene.update_generate_status(False)
            with self.assertRaises(StrictRetryLimitError):
                next(planner)

    def test_cli_configuration_accepts_cap_and_preserves_default(self):
        processor = ConfigProcessor()
        template = str(ROOT / 'configs/simbox/de_plan_with_render_template.yaml')
        default = processor.process_config(template)
        self.assertEqual(default['load_stage']['layout_random_generator']['args']['max_retries'], -1)
        config = processor.process_config(template, cli_args=[
            '--load_stage.layout_random_generator.args.strict_mode=false',
            '--load_stage.layout_random_generator.args.max_retries=-1',
            '--load_stage.layout_random_generator.args.strict_mode=true',
            '--load_stage.layout_random_generator.args.max_retries=20'])
        args = config['load_stage']['layout_random_generator']['args']
        self.assertIs(args['strict_mode'], True)
        self.assertEqual(args['max_retries'], 20)


if __name__ == '__main__':
    unittest.main()
